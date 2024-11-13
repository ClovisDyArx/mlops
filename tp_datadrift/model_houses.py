"""
Create a webservice including a model. The web model should have a :

/predict endpoint. The endpoint should return the prediction And store the X prediction sent by the customer
/detect-drift endpoint. The detect drift enpoint should use the data store from production and some data from train and return wether the production data has drift or not using a method of your choice
"""

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data)
    return jsonify(prediction.tolist())


@app.route('/detect-drift', methods=['GET'])
def detect_drift():
    # Load the data
    data = pd.read_csv('data/houses.csv')
    # Split the data
    X_train, X_test = train_test_split(data, test_size=0.2)
    # Train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train.drop(columns='price'), X_train['price'])
    # Predict the test set
    y_pred = model.predict(X_test.drop(columns='price'))
    # Compute the MAE
    mae = mean_absolute_error(X_test['price'], y_pred)
    # Return the result
    if mae > 1000:
        return jsonify({'drift': True})
    else:
        return jsonify({'drift': False})
