
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
