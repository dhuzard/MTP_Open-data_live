# Montpellier live open-data dashboard

A [Streamlit](https://streamlit.io) app that reads the Montpellier Méditerranée
Métropole Fiware / NGSI open-data portal and turns the documented live resources
into a citizen-friendly dashboard, a natural-language "Citizen Copilot", and a
set of deterministic public/ecological insights.

**The API is the source of truth.** When the optional LLM assistant is enabled it
only (a) routes a question to one approved intent and (b) explains an
already-computed result. It never produces the underlying numbers.

## Features

- **Live dashboard** — status cards and a map across all resources.
- **Public & ecological insights** — deterministic, LLM-free, network-free signals
  (bike availability, free docks, parking pressure, eco-counter activity, freshness,
  data-quality warnings).
- **Ask Montpellier (Citizen Copilot)** — ask in plain language (e.g. "Where can I
  find a bike near the Comédie?"). The pipeline is:
  question → LLM classifies into ONE approved intent → deterministic Python
  fetches / filters / ranks the live API data → LLM explains the verified result.
  The UI shows the exact "data used", timestamps, freshness, and confidence.
- **Data explorer**, **historical time series**, and **raw JSON** views.
- **Citizen feedback** — report a data issue (saved locally to `feedback.csv`).

### Live resources

- VéloMagg bike-station availability
- Off-street parking availability
- Parking-space inventory
- Eco-counter live counts
- Glass waste-container filling levels
- Historical time series for documented resources: VéloMagg, off-street parking,
  and eco-counters

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The default API base URL is:

```text
https://portail-api-data.montpellier3m.fr
```

The OpenAPI document used to build this app is:

```text
https://portail-api.montpellier3m.fr/doc/fiware3m.yaml
```

## Optional LLM assistant

The app runs fully without any API key in **Disabled** mode, which routes questions
with keyword heuristics and relays the deterministic summaries — no network, no cost.

To enable the assistant, pick a provider (OpenAI, Anthropic, or Gemini) in the
sidebar and paste an API key. Keys are entered at runtime in the browser session and
are never written to disk or committed. Install the relevant optional SDK(s) from
`requirements.txt` for the provider you choose.

## Project layout

| File          | Responsibility                                                        |
| ------------- | -------------------------------------------------------------------- |
| `app.py`      | Streamlit UI, API fetching/caching, tabs.                            |
| `intents.py`  | Approved deterministic intents over the live data (no LLM).          |
| `insights.py` | Deterministic public/ecological signals (no LLM, no network).        |
| `geocode.py`  | Landmark dictionary + optional Nominatim fallback geocoding.         |
| `providers.py`| Provider-agnostic LLM layer (Disabled / OpenAI / Anthropic / Gemini).|

## Notes

The waste-container time-series endpoint appears only as a commented section in the
YAML, so the app treats it as unstable and does not expose it as a formal historical
endpoint.

The app uses a 20-second cache TTL to avoid excessive repeated calls while keeping the
display close to live data. Use "Clear cache and refresh" from the sidebar for a forced
refresh.

## Contributing

Issues and pull requests are welcome. By contributing you agree that your
contributions are licensed under the project's AGPL-3.0 license (see below).

## License

Copyright (C) 2026 Damien Huzard.

This program is free software: you can redistribute it and/or modify it under the
terms of the **GNU Affero General Public License v3.0** as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the [GNU AGPL v3](LICENSE) for more details.

> Because this is an AGPL-3.0 project, if you run a modified version of it as a
> network service you must offer the corresponding source of your modified version
> to its users.
