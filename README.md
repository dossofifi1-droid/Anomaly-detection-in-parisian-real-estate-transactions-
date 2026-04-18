# 🏠 Paris Real Estate Anomaly Detection — Streamlit App

## Files needed
- `app.py` — the Streamlit application
- `requirements.txt` — Python dependencies
- `suspects_carte.csv` — your exported suspicious transactions (from Google Drive)

## How to deploy on Streamlit Cloud (free)

1. Create a GitHub account at https://github.com
2. Create a new repository (public)
3. Upload these 3 files: `app.py`, `requirements.txt`, `suspects_carte.csv`
4. Go to https://share.streamlit.io
5. Connect your GitHub account
6. Select your repository → main branch → app.py
7. Click Deploy → your app gets a public URL in ~2 minutes

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Note: place `suspects_carte.csv` in the same folder as `app.py`
If the CSV is not found, the app loads a realistic sample dataset automatically.
