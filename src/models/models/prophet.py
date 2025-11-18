from prophet import Prophet
import pandas as pd

class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def train(self, X, y):
        df = pd.DataFrame({'ds': X['date'], 'y': y})
        self.model.fit(df)

    def predict(self, X):
        future = pd.DataFrame({'ds': X['date']})
        forecast = self.model.predict(future)
        return forecast['yhat']