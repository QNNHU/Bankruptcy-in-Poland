import gzip
import json
import pickle

import ipywidgets as widgets
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from ipywidgets import interact
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Import data

def wrangle(filename):
    with gzip.open(filename, "r") as read_file:
        poland_data_gz = json.load(read_file)
    df = pd.DataFrame.from_dict(poland_data_gz["data"]).set_index("company_id")
 
    return df

df = wrangle("data/poland-bankruptcy-data-2009.json.gz")
print(df.shape)
df.head()

# Split
target = "bankrupt"
X = df.drop(columns= target)
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Resample
over_sampler =  RandomOverSampler()
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()

# Build model
#Baseline
acc_baseline = y_train.value_counts(normalize = True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))

# Iterate - Grid search
clf = make_pipeline(SimpleImputer(), GradientBoostingClassifier())

params = {"simpleimputer__strategy": ["mean","median"],"gradientboostingclassifier__max_depth":range(2,5,1), 
          "gradientboostingclassifier__n_estimators": range(20,31,5)}
model = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1, verbose=1)
model.fit(X_train_over, y_train_over)

results = pd.DataFrame(model.cv_results_)
results.sort_values("rank_test_score").head(10)

# Extract the best parameter
model.best_params_

#  Evaluate
acc_train = model.score(X_train,y_train)
acc_test = model.score(X_test, y_test)

print("Training Accuracy:", round(acc_train, 4))
print("Validation Accuracy:", round(acc_test, 4))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()

# Print classification report
print(classification_report(y_test, model.predict(X_test)))

# Create an interactive dashboard that shows how company profit and losses change in relationship to the model's probability threshold.
def make_cnf_matrix(threshold):
    y_pred_proba = model.predict_proba(X_test)[:,-1]
    y_pred = y_pred_proba>threshold
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"Profit: €{tp*100_000_000}")
    print(f"Losses: €{tp*250_000_000}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar= False)
    pass


thresh_widget = widgets.FloatSlider(min =0, max=1, value =0.5, step = 0.05)

interact(make_cnf_matrix, threshold=thresh_widget);

# Communicate
# Save model
with open("Main/model-5-4.pkl","wb") as f:
    pickle.dump(model,f)

# Make prediction
# Import your module
from Main.model import make_predictions
# Generate predictions
y_test_pred = make_predictions(
    data_filepath="data/poland-bankruptcy-data-2009-mvp-features.json.gz",
    model_filepath="Main/model-5-4.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()