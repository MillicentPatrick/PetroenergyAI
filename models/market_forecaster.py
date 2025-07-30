import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

class MarketForecaster:
    def __init__(self):
        self.model = {'WTI': RandomForestRegressor(n_estimators=100, random_state=42),
                      'Brent': RandomForestRegressor(n_estimators=100, random_state=42)}

    def train_ensemble_models(self, df):
        df = df.copy()
        df['DAYOFYEAR'] = df['DATE'].dt.dayofyear
        X = df[['DAYOFYEAR']]
        y1 = df['WTIPRICE']
        y2 = df['BRENTPRICE']

        if X.empty or y1.isna().all() or y2.isna().all():
            raise ValueError("Market data is insufficient or contains NaN values.")

        self.model['WTI'].fit(X, y1)
        self.model['Brent'].fit(X, y2)

    def predict_prices(self, future_dates):
        df = pd.DataFrame({'DATE': future_dates})
        df['DAYOFYEAR'] = df['DATE'].dt.dayofyear
        df['WTIPRICE_PRED'] = self.model['WTI'].predict(df[['DAYOFYEAR']])
        df['BRENTPRICE_PRED'] = self.model['Brent'].predict(df[['DAYOFYEAR']])
        return df

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
