
# Customer Churn Prediction — Live Project

This is a complete, job-ready data science project scaffold used at **Shiksha Skills Institute**.

## 📦 Contents
- `churn_project.ipynb` — End-to-end notebook (EDA → ML → Evaluation → Save pipeline)
- `streamlit_app.py` — Simple Streamlit app to load the trained pipeline and make predictions
- `requirements.txt` — Dependencies
- *(You provide the dataset file)*

## 🗂 Dataset
Use the **Telco Customer Churn** dataset from Kaggle. Place the CSV in this project folder.
The notebook will look for either:
- `Telco-Customer-Churn.csv` **or**
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## ▶️ How to Run

1) Create & activate a virtual environment (recommended), then install deps:
```
pip install -r requirements.txt
```

2) Put the dataset CSV in the same folder.

3) Open and run the notebook to train and save the pipeline:
```
jupyter notebook churn_project.ipynb
```
A file `churn_pipeline.pkl` will be saved on successful training.

4) Start the Streamlit app:
```
streamlit run streamlit_app.py
```
Upload a CSV to get batch predictions or paste a JSON row for single prediction.

## ✅ Deliverables
- Trained pipeline (`churn_pipeline.pkl`)
- Notebook with EDA, model results, ROC curves, and confusion matrices
- Streamlit app for demo/live predictions
- (Optional) PPT summarizing your findings

## 💡 Tips for Intern
- Add feature engineering (e.g., tenure buckets, TotalSpend = MonthlyCharges * tenure)
- Try hyperparameter tuning
- Document business insights in README and a short PPT

Good luck and have fun!




* `churn_project.ipynb` — full notebook (EDA → ML → evaluation → saves pipeline)
* `streamlit_app.py` — Streamlit app that loads the trained pipeline and predicts churn
* `requirements.txt` — dependencies
* `README.md` — step-by-step instructions

Download everything here:

* [Download the notebook](./files/churn_project.ipynb)
* [Download the Streamlit app](./files/streamlit_app.py)
* [Download requirements.txt](./files/requirements.txt)
* [Download README.md](./files/README.md)

How to use:

1. Put the Kaggle **Telco Customer Churn** CSV in the same folder (the notebook will look for `Telco-Customer-Churn.csv` or `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. `pip install -r requirements.txt`
3. Open and run `churn_project.ipynb` to train and save `churn_pipeline.pkl`.
4. Run `streamlit run streamlit_app.py` to demo live predictions.


