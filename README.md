# Bankruptcy Prediction in Polish Companies

This project applies **XGBoots* to predict bankruptcy among Polish firms using financial statement data. It was developed as part of a learning exercise and extended into a full pipeline with data wrangling, resampling, model training, and evaluation.

---

## 📌 Project Overview
- **Problem:** Bankruptcy is rare but costly. Predicting it accurately can help reduce financial risk.  
- **Dataset:** Polish company bankruptcy dataset (2009).  
- **Target:** `bankrupt` (binary: 1 = bankrupt, 0 = not bankrupt).  
- **Methods:**  
  - Data wrangling & cleaning.  
  - Train/test split (80/20).  
  - Handling class imbalance with **RandomOverSampler**.  
  - **Gradient Boosting Classifier** pipeline with imputation.  
  - Hyperparameter tuning via **GridSearchCV**.  
  - Evaluation with accuracy, confusion matrix, classification report, and profit/loss perspective.  

---

## 📊 Key Results
- **Baseline Accuracy:** ~89.76% (predicting majority class).  
- **Model Test Accuracy:** ~87.17% after tuning Gradient Boosting.  
- **Business Impact:** Model enables cost/benefit evaluation by adjusting decision thresholds.  

Example output (confusion matrix on test set):

![Confusion Matrix](images/confusion_matrix.png)

---
## 📂 Repository Structure
- `data/` — zip file dataset
- `Main/` — Bankruptcy in Poland.py, model.py, model-5-4.pkl
- `notebooks` — Bankruptcy_Poland.ipynb
- `images/` — Plots and saved visualizations
- `README.md` — Project overview
- `requirements.txt` — Python dependencies

---

## ⚙️ Getting Started

Follow these steps to run the project locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/QNNHU/poland-bankruptcy-prediction.git
   cd poland-bankruptcy-prediction

2. **Create a virtual environment (recommended)**
   ```bash
    python -m venv venv
    # Activate the environment
    source venv/bin/activate    # On Mac/Linux
    venv\Scripts\activate       # On Windows

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

4. **Launch Jupyter and open the main notebook or run the Python file directly**

### ▶️ Option 1: Run the Jupyter Notebook
```bash
jupyter notebook notebooks/Bankruptcy_Poland.ipynb

### ▶️ Option 2: Run the Jupyter Notebook
python Main/Bankruptcy_Poland.py