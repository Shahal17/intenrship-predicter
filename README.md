# Student Placement / Internship Readiness Predictor

A beginner-friendly supervised machine learning project that predicts whether a student is placement/internship ready using academic, technical, and skill-related factors.

## Objective

Develop and compare supervised classification models that predict student placement readiness and provide useful insights about the factors that influence the prediction.

## Dataset

**Indian Student Placement Dataset 2025** by Sakhare Bharat on Kaggle:
https://www.kaggle.com/datasets/sakharebharat/indian-student-placement-dataset-2025

The dataset is synthetic and intended for education/research. It contains 12,000+ student records with attributes related to academics, skills, internships, projects, certifications, and placement outcomes.

> The application is an educational ML demonstration. It should not be used as an automated hiring or admission decision system.

## Models compared

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The model with the best F1-score is saved for prediction.

## Project structure

```text
intenrship-predicter/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── src/
    ├── download_data.py
    └── train.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shahal17/intenrship-predicter.git
cd intenrship-predicter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Kaggle dataset

```bash
python src/download_data.py
```

If Kaggle authentication is required on your machine, configure your Kaggle credentials first. You can also manually download the CSV from the dataset page and place it inside the `data/` folder.

### 5. Train and compare models

```bash
python src/train.py
```

Training creates:

- `models/best_model.joblib` — best trained ML pipeline
- `models/metadata.json` — input schema and label information
- `results/model_metrics.csv` — model comparison
- `results/feature_importance.csv` — permutation-based feature importance

### 6. Run the web app

```bash
streamlit run app.py
```

Enter a student's details and click **Predict readiness** to see the predicted result and confidence score.

## ML workflow

1. Load the placement dataset.
2. Detect the placement-status target column.
3. Remove identifier and post-placement leakage fields such as salary/package/company fields.
4. Split data into training and testing sets using stratification.
5. Impute missing values.
6. Scale numerical variables.
7. One-hot encode categorical variables.
8. Train Logistic Regression, Decision Tree, Random Forest, and SVM classifiers.
9. Compare accuracy, precision, recall, and F1-score.
10. Save the model with the highest F1-score.
11. Estimate feature importance with permutation importance.
12. Serve predictions through a Streamlit interface.

## Notes

- The training code is designed to adapt to the CSV column names found in the Kaggle dataset rather than requiring one hard-coded schema.
- Post-outcome fields such as salary/package and company information are intentionally excluded to reduce target leakage.
- Results can change if the Kaggle dataset is updated.

## Tech stack

Python, Pandas, NumPy, scikit-learn, Joblib, Streamlit, KaggleHub
