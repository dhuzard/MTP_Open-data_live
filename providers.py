# Montpellier live open-data dashboard
# Copyright (C) 2026 Damien Huzard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Provider-agnostic LLM layer for the Montpellier open-data Streamlit app.

The LLM never invents facts. It only:

  (a) classify_intent — maps a citizen question onto ONE approved structured
      intent (or the literal "unsupported"), and
  (b) explain — turns an ALREADY-COMPUTED deterministic result dict into a
      short, citizen-friendly explanation, grounded strictly in that dict.

Four providers are supported: OpenAI, Anthropic, Gemini, and a Disabled
(no-key, keyword-heuristic, no-network) mode.

Design constraints honoured here:
  * All third-party SDK imports are LAZY — done INSIDE the methods that use
    them — so the app runs with only streamlit/pandas/requests installed when
    the user picks "Disabled". A missing SDK raises a clear RuntimeError only
    when that provider is actually used.
  * No network calls happen at import time.
  * Any parse/API failure in classify_intent degrades to the safe sentinel
    {"intent": "unsupported", "params": {}, "confidence": 0.0}.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Default model per provider. Overridable via get_provider(..., model=...).
DEFAULT_MODELS: Dict[str, str] = {
    "OpenAI": "gpt-4o-mini",
    "Anthropic": "claude-haiku-4-5-20251001",
    "Gemini": "gemini-2.0-flash",
}

# Safe fallback that the rest of the app treats as "I can't answer this".
_UNSUPPORTED: Dict[str, Any] = {"intent": "unsupported", "params": {}, "confidence": 0.0}


# ---------------------------------------------------------------------------
# Prompt construction (shared by all online providers)
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = (
    "You are an intent classifier for a Montpellier (France) open-data civic "
    "assistant. The assistant can ONLY answer questions about Montpellier "
    "bike-share stations, car parking, glass/recycling containers, an overall "
    "city snapshot, an ecological signal, and data-quality status.\n"
    "\n"
    "You DO NOT answer the question and you NEVER invent facts. You only pick "
    "exactly one structured intent from the allowed list (or the literal "
    "string \"unsupported\") and extract location parameters from the text.\n"
    "\n"
    "Respond with STRICT JSON ONLY — no prose, no markdown, no code fences. "
    "The JSON must be exactly:\n"
    '{"intent": <one allowed intent OR "unsupported">, '
    '"params": {<extracted params>}, '
    '"confidence": <float between 0 and 1>}\n'
    "\n"
    "Rules for params:\n"
    '  * For single-location intents, set "location" to the free-text place '
    "from the question.\n"
    '  * For trip planning, set "origin" and "destination".\n'
    "  * Prefer a known landmark name when the question matches one.\n"
    "  * If the question cannot be answered from Montpellier bike/parking/"
    'glass/eco data, return intent "unsupported", empty params, confidence 0.0.'
)


def _build_classify_user(
    question: str,
    allowed_intents: List[str],
    landmark_names: List[str],
) -> str:
    """Build the user-turn prompt passed to an online provider."""
    intents_block = "\n".join(f"  - {name}" for name in allowed_intents)
    if landmark_names:
        landmarks_block = "\n".join(f"  - {name}" for name in landmark_names)
    else:
        landmarks_block = "  (none provided)"
    return (
        "Allowed intents (use exactly one of these strings, or "
        '"unsupported"):\n'
        f"{intents_block}\n"
        "\n"
        "Known Montpellier landmarks (prefer these when extracting a "
        "location):\n"
        f"{landmarks_block}\n"
        "\n"
        f"Citizen question:\n{question}\n"
        "\n"
        "Return ONLY the strict JSON object described in the system message."
    )


def _build_explain_prompt(intent: str, params: Dict[str, Any], result: Dict[str, Any]) -> str:
    """Build the explanation prompt. `result` is serialised as JSON for the model."""
    result_json = json.dumps(result, ensure_ascii=False, default=str)
    params_json = json.dumps(params, ensure_ascii=False, default=str)
    return (
        "You explain an ALREADY-COMPUTED result to a citizen. Deterministic "
        "Python produced the result below; you must NOT add numbers, names, "
        "distances, or claims that are not present in it.\n"
        "\n"
        f"Intent: {intent}\n"
        f"Parameters: {params_json}\n"
        "\n"
        "Result (JSON, the only source of truth):\n"
        f"{result_json}\n"
        "\n"
        "Write a 2 to 4 sentence, friendly explanation in the SAME LANGUAGE as "
        "the result's summary/question. Ground every statement strictly in the "
        'result. If result["ok"] is false, simply relay result["summary"] '
        'politely. Preserve any caveats listed in result["notes"] (for example '
        "that distances are straight-line / as-the-crow-flies). Do not invent "
        "anything. Reply with the explanation text only."
    )


_EXPLAIN_SYSTEM = (
    "You rewrite a precomputed civic-data result into a short, friendly "
    "explanation for a member of the public. You never introduce facts that "
    "are not in the provided result JSON. You answer in the same language as "
    "the result."
)


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------

def _parse_intent_json(text: Optional[str], allowed_intents: List[str]) -> Dict[str, Any]:
    """Parse a model's raw output into the canonical intent dict.

    Strips ```json fences, locates the first {...} block, json.loads it, and
    normalises the shape. Returns the _UNSUPPORTED sentinel on any problem.
    """
    if not text:
        return dict(_UNSUPPORTED)

    cleaned = text.strip()

    # Strip leading/trailing code fences (```json ... ``` or ``` ... ```).
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Find the first balanced-ish {...} block (greedy to the last brace).
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return dict(_UNSUPPORTED)
    candidate = cleaned[start : end + 1]

    try:
        data = json.loads(candidate)
    except (ValueError, TypeError):
        return dict(_UNSUPPORTED)

    if not isinstance(data, dict):
        return dict(_UNSUPPORTED)

    intent = data.get("intent", "unsupported")
    if not isinstance(intent, str) or (intent != "unsupported" and intent not in allowed_intents):
        return dict(_UNSUPPORTED)

    params = data.get("params", {})
    if not isinstance(params, dict):
        params = {}

    try:
        confidence = float(data.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if intent == "unsupported":
        return {"intent": "unsupported", "params": {}, "confidence": confidence}

    return {"intent": intent, "params": params, "confidence": confidence}


def _format_disabled_explanation(result: Dict[str, Any]) -> str:
    """Deterministic, no-network explanation: summary plus any notes."""
    summary = result.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)
    lines: List[str] = [summary] if summary else []

    notes = result.get("notes", [])
    if isinstance(notes, (list, tuple)):
        for note in notes:
            note_text = note if isinstance(note, str) else str(note)
            if note_text:
                lines.append(note_text)
    elif notes:
        lines.append(str(notes))

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Provider-agnostic LLM interface used by the rest of the app."""

    name: str = "base"

    @abstractmethod
    def classify_intent(
        self,
        question: str,
        allowed_intents: List[str],
        landmark_names: List[str],
    ) -> Dict[str, Any]:
        """Classify `question` into one approved intent.

        Returns exactly:
            {"intent": <allowed intent | "unsupported">,
             "params": {...},
             "confidence": <float 0..1>}
        """
        raise NotImplementedError

    @abstractmethod
    def explain(self, intent: str, params: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Explain a precomputed deterministic `result` in citizen-friendly prose."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Disabled provider (no key, keyword heuristics, no network)
# ---------------------------------------------------------------------------

class DisabledProvider(LLMProvider):
    """Offline provider. Keyword-rule classification; relays summary on explain."""

    name = "Disabled"

    # Ordered keyword -> intent rules for single-location intents. Trip
    # planning is detected separately (needs both an origin and a destination).
    _RULES = (
        (("trajet", "itiner", "itinér"), "plan_bike_trip"),
        (("glass", "verre", "recycl"), "find_glass_container"),
        (("bike", "vélo", "velo", "bicycle", "cycl"), "find_nearest_bike_station"),
        (("park", "parking", "voiture", "car", "stationnement"), "find_nearest_parking"),
        (("ecolog", "écolog", "carbon", "carbone", "co2", "emission", "émission"),
         "summarize_ecological_signal"),
        (("quality", "stale", "fresh", "qualité", "qualite", "fiab", "à jour", "a jour"),
         "explain_data_quality"),
        (("summary", "overview", "city", "résumé", "resume", "snapshot", "ville", "aperçu"),
         "summarize_city_now"),
    )

    # Markers that split an origin from a destination in a trip phrasing.
    # Kept word-boundary and unambiguous on purpose: a bare " a " / " à " is
    # far too common in ordinary questions ("find a parking", "à Montpellier")
    # to be treated as an origin/destination separator.
    _TRIP_SPLITTERS = (" to ", " vers ", " jusqu'à ", " jusqu'a ")

    def classify_intent(
        self,
        question: str,
        allowed_intents: List[str],
        landmark_names: List[str],
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            return dict(_UNSUPPORTED)

        lowered = question.lower()

        # --- Trip detection: explicit verb, or an origin+destination pair. ---
        # Require BOTH endpoints (the strong signal), or an explicit trip verb.
        trip_origin, trip_destination = self._extract_trip(question, lowered)
        trip_verb = any(k in lowered for k in ("trajet", "itiner", "itinér"))
        if "plan_bike_trip" in allowed_intents and (
            (trip_origin and trip_destination) or trip_verb
        ):
            params: Dict[str, Any] = {}
            if trip_origin:
                params["origin"] = trip_origin
            if trip_destination:
                params["destination"] = trip_destination
            if params:
                return {"intent": "plan_bike_trip", "params": params, "confidence": 0.5}

        # --- Single-location keyword rules. ---
        for keywords, intent in self._RULES:
            if intent == "plan_bike_trip":
                continue  # handled above
            if intent not in allowed_intents:
                continue
            if any(k in lowered for k in keywords):
                location = self._extract_location(question, lowered)
                params = {"location": location} if location else {}
                return {"intent": intent, "params": params, "confidence": 0.5}

        return dict(_UNSUPPORTED)

    def explain(self, intent: str, params: Dict[str, Any], result: Dict[str, Any]) -> str:
        # No network: relay the deterministic summary plus caveats verbatim.
        return _format_disabled_explanation(result)

    # -- helpers ------------------------------------------------------------

    def _extract_location(self, question: str, lowered: str) -> str:
        """Pull the text after a 'near/at/around/près de' marker, heuristically.

        Markers are matched at word boundaries so short words like "at"/"by"
        don't fire inside other words ("what", "nearby" handled separately).
        """
        markers = (
            r"près\s+d[eu]\b", r"pres\s+d[eu]\b", r"proche\s+de\b", r"autour\s+de\b",
            r"near\b", r"nearby\b", r"around\b", r"\bat\b", r"\bby\b",
        )
        for marker in markers:
            m = re.search(marker + r"\s+(.+)$", question, flags=re.IGNORECASE)
            if m:
                tail = self._trim_location(m.group(1))
                if tail:
                    return tail
        return ""

    def _extract_trip(self, question: str, lowered: str):
        """Return (origin, destination) for a 'from X to Y' / 'de X à Y' phrasing."""
        origin = ""
        destination = ""

        # English "from X to Y".
        m = re.search(r"from\s+(.+?)\s+to\s+(.+)$", question, flags=re.IGNORECASE)
        if m:
            return self._trim_location(m.group(1)), self._trim_location(m.group(2))

        # French "de X à Y" / "de X vers Y".
        m = re.search(r"\bde\s+(.+?)\s+(?:à|a|vers)\s+(.+)$", question, flags=re.IGNORECASE)
        if m:
            return self._trim_location(m.group(1)), self._trim_location(m.group(2))

        # Generic: split on the first trip splitter token if one is present.
        for splitter in self._TRIP_SPLITTERS:
            if splitter in lowered:
                left, _, right = self._partition(question, lowered, splitter)
                left = self._trim_location(left)
                right = self._trim_location(right)
                if left and right:
                    origin, destination = left, right
                    break

        return origin, destination

    @staticmethod
    def _partition(question: str, lowered: str, splitter: str):
        idx = lowered.find(splitter)
        if idx == -1:
            return "", "", ""
        return question[:idx], splitter, question[idx + len(splitter):]

    @staticmethod
    def _trim_location(text: str) -> str:
        """Tidy an extracted location fragment."""
        if not text:
            return ""
        text = text.strip().strip("?.!,;:")
        # Drop a leading article that survived extraction.
        text = re.sub(r"^(the|la|le|les|du|de la|de|d')\s+", "", text, flags=re.IGNORECASE)
        return text.strip()


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI chat-completions backed provider (SDK imported lazily)."""

    name = "OpenAI"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS["OpenAI"]

    def _client(self):
        if not self.api_key:
            raise RuntimeError(
                "OpenAI is selected but no API key was provided. "
                "Enter an OpenAI API key to use this provider."
            )
        try:
            from openai import OpenAI  # lazy import
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "The 'openai' package is not installed. Install it with "
                "`pip install openai`, or switch the provider to Disabled."
            ) from exc
        return OpenAI(api_key=self.api_key)

    def classify_intent(
        self,
        question: str,
        allowed_intents: List[str],
        landmark_names: List[str],
    ) -> Dict[str, Any]:
        try:
            client = self._client()
            user_prompt = _build_classify_user(question, allowed_intents, landmark_names)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content
        except RuntimeError:
            raise
        except Exception:
            return dict(_UNSUPPORTED)
        return _parse_intent_json(text, allowed_intents)

    def explain(self, intent: str, params: Dict[str, Any], result: Dict[str, Any]) -> str:
        client = self._client()
        prompt = _build_explain_prompt(intent, params, result)
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": _EXPLAIN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content
        except Exception:
            return _format_disabled_explanation(result)
        if not text or not text.strip():
            return _format_disabled_explanation(result)
        return text.strip()


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Anthropic Messages-API backed provider (SDK imported lazily)."""

    name = "Anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS["Anthropic"]

    def _client(self):
        if not self.api_key:
            raise RuntimeError(
                "Anthropic is selected but no API key was provided. "
                "Enter an Anthropic API key to use this provider."
            )
        try:
            import anthropic  # lazy import
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "The 'anthropic' package is not installed. Install it with "
                "`pip install anthropic`, or switch the provider to Disabled."
            ) from exc
        return anthropic.Anthropic(api_key=self.api_key)

    def classify_intent(
        self,
        question: str,
        allowed_intents: List[str],
        landmark_names: List[str],
    ) -> Dict[str, Any]:
        try:
            client = self._client()
            user_prompt = _build_classify_user(question, allowed_intents, landmark_names)
            msg = client.messages.create(
                model=self.model,
                max_tokens=300,
                system=_CLASSIFY_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = msg.content[0].text
        except RuntimeError:
            raise
        except Exception:
            return dict(_UNSUPPORTED)
        return _parse_intent_json(text, allowed_intents)

    def explain(self, intent: str, params: Dict[str, Any], result: Dict[str, Any]) -> str:
        client = self._client()
        prompt = _build_explain_prompt(intent, params, result)
        try:
            msg = client.messages.create(
                model=self.model,
                max_tokens=400,
                system=_EXPLAIN_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
        except Exception:
            return _format_disabled_explanation(result)
        if not text or not text.strip():
            return _format_disabled_explanation(result)
        return text.strip()


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """Google Gemini backed provider via google-genai (SDK imported lazily)."""

    name = "Gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS["Gemini"]

    def _client(self):
        if not self.api_key:
            raise RuntimeError(
                "Gemini is selected but no API key was provided. "
                "Enter a Google API key to use this provider."
            )
        try:
            from google import genai  # lazy import (google-genai)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "The 'google-genai' package is not installed. Install it with "
                "`pip install google-genai`, or switch the provider to Disabled."
            ) from exc
        return genai.Client(api_key=self.api_key)

    def classify_intent(
        self,
        question: str,
        allowed_intents: List[str],
        landmark_names: List[str],
    ) -> Dict[str, Any]:
        try:
            client = self._client()
            user_prompt = _build_classify_user(question, allowed_intents, landmark_names)
            contents = f"{_CLASSIFY_SYSTEM}\n\n{user_prompt}"
            response = client.models.generate_content(model=self.model, contents=contents)
            text = response.text
        except RuntimeError:
            raise
        except Exception:
            return dict(_UNSUPPORTED)
        return _parse_intent_json(text, allowed_intents)

    def explain(self, intent: str, params: Dict[str, Any], result: Dict[str, Any]) -> str:
        client = self._client()
        prompt = _build_explain_prompt(intent, params, result)
        contents = f"{_EXPLAIN_SYSTEM}\n\n{prompt}"
        try:
            response = client.models.generate_content(model=self.model, contents=contents)
            text = response.text
        except Exception:
            return _format_disabled_explanation(result)
        if not text or not text.strip():
            return _format_disabled_explanation(result)
        return text.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(
    name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMProvider:
    """Return a provider instance for `name`.

    "Disabled" ignores key/model. The online providers default their model
    from DEFAULT_MODELS when `model` is None. A falsy api_key is allowed at
    construction time; classify/explain will raise a friendly RuntimeError if
    the provider is then actually used without a key.
    """
    key = (name or "").strip().lower()
    if key in ("disabled", "", "none", "off"):
        return DisabledProvider()
    if key == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    if key == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    if key == "gemini":
        return GeminiProvider(api_key=api_key, model=model)

    raise ValueError(
        f"Unknown provider {name!r}. Expected one of: "
        "Disabled, OpenAI, Anthropic, Gemini."
    )
