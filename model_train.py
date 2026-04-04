# ================================
# IMPORTS
# ================================
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ================================
# LOAD DATA
# ================================
with open('power_spectrum_1000.pkl', 'rb') as f:
    data = pickle.load(f)

X = data["power_spectrum"]
y = data["label"]


# ================================
# SHUFFLE DATA (IMPORTANT)
# ================================
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)


# ================================
# TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ================================
# MODEL
# ================================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    ))
])


# ================================
# TRAIN
# ================================
pipeline.fit(X_train, y_train)


# ================================
# EVALUATE
# ================================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ================================
# SAVE MODEL
# ================================
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel saved as model.pkl")