import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from imblearn.pipeline import Pipeline
import joblib

df = pd.read_csv("energy_data.csv")


X = df.iloc[:,:-2]
y1 = df.iloc[:,-2]
y2 = df.iloc[:,-1]

pipe_heating = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVR(C=100))
])

pipe_cooling = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVR(C=100))
])

pipe_heating.fit(X, y1)
pipe_cooling.fit(X, y2)

joblib.dump(
    {"heating": pipe_heating, "cooling": pipe_cooling},
    "energy_load_models.pkl"
)