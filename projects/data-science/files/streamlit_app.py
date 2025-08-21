
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Upload customer data or fill the form to predict churn using a trained pipeline.")

MODEL_PATH = "churn_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found. Train and save `churn_pipeline.pkl` by running the notebook first.")
else:
    pipe = joblib.load(MODEL_PATH)
    st.success("Model loaded.")

    with st.expander("Upload CSV for batch predictions"):
        uploaded = st.file_uploader("Upload CSV with the same columns used in training (excluding 'Churn')", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            preds = pipe.predict(df)
            probs = pipe.predict_proba(df)[:,1]
            out = df.copy()
            out["churn_pred"] = preds
            out["churn_prob"] = probs
            st.write(out.head())
            st.download_button("Download Predictions", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

    st.subheader("Or enter a single customer manually")
    # Try to infer inputs from the pipeline preprocess step
    # For simplicity, we will ask the user to paste a JSON row matching training columns.
    example = st.text_area("Paste a JSON object with feature columns (excluding 'Churn')", 
        value='{}')
    if st.button("Predict from JSON") and example.strip() and os.path.exists(MODEL_PATH):
        try:
            row = pd.DataFrame([pd.read_json(example, typ="series")])
        except Exception:
            # Fallback: try eval-like parsing (not secure generally; for demo only)
            import ast
            row = pd.DataFrame([pd.Series(ast.literal_eval(example))])
        probs = pipe.predict_proba(row)[:,1]
        preds = pipe.predict(row)
        st.write({"churn_pred": int(preds[0]), "churn_prob": float(probs[0])})
