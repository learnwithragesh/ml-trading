import pandas as pd
from src.models.models.linear import LinearModel

class Analytics:
    def __init__(self):
        pass

    def importData(self):
        print("Importing data...")
        df = pd.read_csv('data/external/NIFTY 50-9-09-2024-to-9-09-2025.csv')
        return df

    def predict(self, df, days):
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
        print("Running analytics...")
        df.set_index("Date", inplace=True)
        df["date_ordinal"] = df.index.map(lambda date: date.toordinal())
        X = df[["date_ordinal"]]
        y = df["Close"]
        model = LinearModel()
        model.train(X, y)
        # Get the last date from your DataFrame
        last_date = df.index.max()

        # Create a range of future dates (next 30 days)
        future_dates = pd.date_range(start=last_date, periods=days, freq='D')

        # Convert future dates to ordinal
        future_df = pd.DataFrame({"Date": future_dates})
        future_df["date_ordinal"] = future_df["Date"].map(pd.Timestamp.toordinal)

        # Predict future closing prices
        future_df["Predicted_Close"] = model.predict(future_df[["date_ordinal"]])

        return future_df

    def run(self):
        df = self.importData()
        df = self.predict(df, 30)
        print(df)