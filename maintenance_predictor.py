import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

class MaintenancePredictor:
    def __init__(self, model_path='models/maintenance_model.pkl'):
        self.model_path = model_path
        self.model = None

    def train_anomaly_model(self, df):
        if df.empty:
            return
        X = df[['HEALTHSCORE']].dropna()
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(X)
        self.save_model(self.model_path)

    def save_model(self, path=None):
        if self.model is not None:
            joblib.dump(self.model, path or self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def predict_anomalies(self, df):
        if self.model is None:
            self.load_model()
        if self.model is None or df.empty:
            return pd.Series([1] * len(df))  # All normal
        X = df[['HEALTHSCORE']].fillna(0)
        return pd.Series(self.model.predict(X), index=df.index)

    def generate_maintenance_report(self, df):
        if self.model is None:
            self.load_model()
        if self.model is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df['anomaly'] = self.model.predict(df[['HEALTHSCORE']])
        df = df[df['anomaly'] == -1]  # Only anomalies

        report = df.groupby(['FACILITYID', 'EQUIPMENTID']).agg({
            'HEALTHSCORE': 'min',
            'TIMESTAMP': 'max'
        }).reset_index()

        report = report.rename(columns={
            'HEALTHSCORE': 'Min Health Score',
            'TIMESTAMP': 'Last Seen'
        })

        report['Priority'] = report['Min Health Score'].rank(method='first', ascending=True).astype(int)
        return report.sort_values(by='Priority')
