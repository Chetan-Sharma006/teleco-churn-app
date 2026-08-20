# Telco Customer Churn Predictor

A Streamlit web app that predicts whether a telecom customer is likely to churn,
built from the model in your notebook (tuned Logistic Regression, ~81% test accuracy).

## Files
- `app.py` — the Streamlit app
- `best_model.pkl` — trained model
- `scaler.pkl` — StandardScaler fitted on training data
- `feature_columns.pkl` — exact column order the model expects
- `requirements.txt` — Python dependencies

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL it prints (usually http://localhost:8501).

## Deploy for free (Streamlit Community Cloud) — recommended, ~5 minutes
1. Create a new **public** GitHub repo and push these 5 files to it (via GitHub Desktop,
   the GitHub web uploader, or `git init && git add . && git commit -m "init" && git push`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick your repo/branch, set the main file path to `app.py`.
4. Click "Deploy" — you'll get a public `https://<something>.streamlit.app` link in a couple minutes.

## Alternative: Render.com
1. Push the same files to a GitHub repo.
2. On Render, create a new "Web Service" from that repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy — Render gives you a public URL.

## Notes
- This reproduces your notebook's pipeline (same cleaning, same encoding, same
  GridSearchCV-tuned Logistic Regression) trained on the public IBM Telco Customer
  Churn dataset, since the dataset/model weren't included in the upload.
- Model performance: ~81% accuracy, 0.67 precision / 0.55 recall on the churn class.
