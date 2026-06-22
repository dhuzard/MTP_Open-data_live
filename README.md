# Montpellier live open-data Streamlit dashboard

This Streamlit app reads the Montpellier Méditerranée Métropole Fiware/OpenAPI portal and displays the documented live resources:

- VéloMagg bike-station availability
- Off-street parking availability
- Parking-space inventory
- Eco-counter live counts
- Glass waste-container filling levels
- Historical time series for documented resources: VéloMagg, off-street parking, and eco-counters

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

## Notes

The waste-container time-series endpoint appears only as a commented section in the YAML, so the app treats it as unstable and does not expose it as a formal historical endpoint.

The app uses a 20-second cache TTL to avoid excessive repeated calls while keeping the display close to live data. Use “Clear cache and refresh” from the sidebar for a forced refresh.
