# Road Accident Severity Predictor (India)  

End-to-end machine learning project that predicts **injury severity** in road accidents  
using Indian accident data, with a deployed **Flask + HTML/CSS/JS** web app and a  
risk-based interpretation layer.

---

##  Problem Statement

Given details about a road accident (day, driver age band, weather, light conditions,  
vehicles involved, casualties, time of day), the goal is to predict **injury severity**:

- `Fatal injury`
- `Serious Injury`
- `Slight Injury`

Why this matters:

- Helps traffic authorities and road safety teams identify **high-risk patterns**
- Can support **resource planning**, awareness campaigns, or triage assistance
- Shows ability to handle **imbalanced, noisy real-world data** and **deploy a model**

---

##  Key Features

- Uses **Indian road accident dataset** (Kaggle: “Road Accident Severity in India”)
- Handles **strong class imbalance** (Slight Injury dominates) using **SMOTE**
- Compares:
  - Logistic Regression (baseline)
  - Random Forest
  - ANN (vanilla)
  - **ANN + SMOTE (final deployed model)**
- Deployed via **Flask** with:
  - HTML/CSS/JS frontend
  - JSON `/predict` API
- Adds a **risk score interpretation layer**:

  \[
  \text{risk\_score} = P(\text{Fatal}) + P(\text{Serious})
  \]

  Mapped to:
  - **Low** / **Medium** / **High** risk  
  even when the predicted class is “Slight Injury”.

---

##  Architecture Overview

**High-level flow:**

```text
Raw CSV (Kaggle)
        ↓
Data Preprocessing (cleaning, feature selection, Hour extraction)
        ↓
Train / Val / Test Split (stratified)
        ↓
Preprocessing + One-Hot Encoding
        ↓
SMOTE on Training Set (to balance classes)
        ↓
ANN Model Training (TensorFlow / Keras)
        ↓
Saved Artifacts (model.h5, preprocessor.pkl, metadata.json)
        ↓
Flask API (/predict) + Risk Score Logic
        ↓
HTML/CSS/JS Frontend (form + probability bars + risk badge)

Tech stack:

Python: 3.13 (venv)

ML / DL:

pandas, numpy

scikit-learn

imbalanced-learn (SMOTE)

tensorflow / keras

Web:

Flask

HTML5, CSS3, vanilla JavaScript (no frontend framework)

Visualization:

matplotlib, seaborn

Notebooks:

Jupyter (.ipynb)

Traffic/
├── app/
│   ├── flask_app.py          # Flask web app (API + UI)
│   ├── templates/
│   │   └── index.html        # Frontend page
│   └── static/
│       ├── css/
│       │   └── styles.css    # Modern UI styling
│       └── js/
│           └── app.js        # Calls /predict, renders results
│
├── data/
│   ├── raw/
│   │   └── road_accidents_india.csv   # Original Kaggle dataset
│   └── processed/
│       └── accidents_india_clean.csv  # Cleaned + feature-engineered data
│
├── models/
│   ├── baseline_model.pkl            # Best classical ML model (e.g. LogisticReg)
│   ├── baseline_preprocessor.pkl
│   ├── baseline_metadata.json
│   ├── ann_model.h5                  # Final ANN + SMOTE model
│   ├── ann_preprocessor.pkl
│   ├── ann_metadata.json
│   └── ann_label_encoder.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis
│   ├── 02_Model_Experiments.ipynb    # Baseline vs ANN vs ANN+SMOTE
│   └── 03_Inference_Testing.ipynb    # Manual & test-set inference + risk analysis
│
├── reports/
│   └── ann_confusion_matrix_test.png # Final ANN confusion matrix (test set)
│
├── src/
│   ├── data_preprocessing/
│   │   └── data_prep.py              # Load raw CSV, clean, save processed data
│   ├── training/
│   │   ├── train_baseline_ml.py      # Train LogisticReg + RandomForest
│   │   └── train_ann.py              # Train ANN + SMOTE (8 features)
│   ├── inference/
│   │   └── predict_single.py         # Inference helper used by Flask
│   └── evaluation/
│       ├── evaluate_ann.py           # Metrics + confusion matrix on test set
│       ├── check_predictions_distribution.py
│       └── inspect_non_slight_examples.py
│
├── requirements.txt
└── README.md
Dataset

Name: Road Accident Severity in India

Source: Kaggle

Modality: Tabular

Key columns used (final feature set):

Day_of_week

Age_band_of_driver

Sex_of_driver

Weather_conditions

Light_conditions

Number_of_vehicles_involved

Number_of_casualties

Hour (extracted from Time)

Target:

Accident_Severity (3 classes)

The dataset is highly imbalanced, with Slight Injury being the majority class.
This motivates the use of SMOTE and macro F1 as a key metric.

Setup & Installation
# 1. Clone the repository
git clone <your-repo-url>.git
cd Traffic

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

Data Preprocessing

Make sure raw dataset is placed at:

data/raw/road_accidents_india.csv


Then run:

python src/data_preprocessing/data_prep.py


This script:

Loads the raw CSV

Cleans and normalizes column names

Extracts Hour from Time (if present)

Drops unused / noisy columns

Saves the processed file to:

data/processed/accidents_india_clean.csv

 Model Training
1 Baseline classical ML
python src/training/train_baseline_ml.py


Trains:

LogisticRegression

RandomForestClassifier

Uses shared preprocessing (imputation + scaling + one-hot)

Selects best model by validation macro F1

Saves:

models/baseline_model.pkl

models/baseline_preprocessor.pkl

models/baseline_metadata.json

ANN + SMOTE (final model)
python src/training/train_ann.py


This script:

Uses only the final 8 features

Splits into train / val / test (stratified)

Fits preprocessing pipeline

Applies SMOTE on the training set

Builds an ANN in Keras:

Dense(64, ReLU) → Dropout(0.3)

Dense(32, ReLU)

Dense(16, ReLU) + Dropout

Dense(3, Softmax)

Uses early stopping on validation loss

Evaluates on validation + test

Saves:

models/ann_model.h5

models/ann_preprocessor.pkl

models/ann_metadata.json

models/ann_label_encoder.pkl

 Evaluation

Evaluate the final ANN on the test set:

python src/evaluation/evaluate_ann.py


Example results (ANN + SMOTE, 8 features):

Test accuracy ≈ 0.62

Test macro F1 ≈ 0.42

Per-class performance (example):

Fatal injury:

Precision ≈ 0.16

Recall ≈ 0.42

Serious Injury:

Precision ≈ 0.20

Recall ≈ 0.49

Slight Injury:

Precision ≈ 0.88

Recall ≈ 0.65

The confusion matrix is saved to:

reports/ann_confusion_matrix_test.png

 Risk Score Interpretation Layer

Instead of relying only on the predicted class (argmax), the app computes:

risk_score
=
𝑃
(
Fatal injury
)
+
𝑃
(
Serious Injury
)
risk_score=P(Fatal injury)+P(Serious Injury)

This is mapped to:

High if risk_score ≥ 0.7

Medium if risk_score ≥ 0.4

Low otherwise

The frontend displays:

Predicted severity class

Risk level badge (Low / Medium / High, color-coded)

Risk score as a percentage

Probability bars for each class

This is closer to how a real safety system would present results.

 Running the Web App (Flask + Frontend)

From the project root:

source venv/bin/activate
python app/flask_app.py


By default, the app runs at:

http://127.0.0.1:5000/


The UI lets you enter:

Day of week

Driver age band

Driver sex

Weather conditions

Light conditions

Number of vehicles involved

Number of casualties

Hour (0–23)

On submit:

Frontend (app/static/js/app.js) sends JSON to /predict

Flask loads:

ANN model

Preprocessor

Metadata

Backend returns:

predicted_label

probabilities

risk_score

risk_level

The frontend then displays:

Severity

Risk badge

Probability bars

📓 Notebooks

01_EDA.ipynb

Dataset overview

Missing values

Class imbalance

Feature distribution plots

Justification for final feature subset (8 features)

02_Model_Experiments.ipynb

Baseline Logistic Regression & RandomForest

ANN (vanilla)

ANN + SMOTE

Metric comparison (accuracy + macro F1)

Final model choice rationale

03_Inference_Testing.ipynb

Manual test cases

Inference on test-set rows

Risk score distribution

Risk vs true severity analysis

 Limitations & Future Work

Current limitations:

Severe class imbalance: Fatal and Serious are still much rarer than Slight.

Only 8 features are used at inference time (by design, to keep the UI simple).

ANN performance is good but not perfect; some borderline slight/serious cases are hard.

Possible improvements:

Try cost-sensitive learning or focal loss for the ANN.

Add calibrated tree-based models (Gradient Boosting / XGBoost) as additional baselines.

Engineer more features:

Night vs day binary

Weather severity index

Vehicle-to-casualty ratio

Add model explainability (e.g. SHAP values) to show which features contributed most.

Deploy to a cloud platform (Render / Railway) with CI/CD.

🎯 What This Project Demonstrates

Data cleaning and preprocessing on a real, messy, imbalanced dataset

Classical ML baselines + proper comparison with ANN

Handling of class imbalance (SMOTE)

Practical deep learning on tabular data

Model evaluation with realistic metrics (macro F1, confusion matrix)

Serving a trained model via Flask API

Building a complete frontend (HTML/CSS/JS) with risk-aware outputs

Clear project structure, documentation, and notebooks