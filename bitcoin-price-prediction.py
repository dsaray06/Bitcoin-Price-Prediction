import pandas as pd
from prophet import Prophet


df = pd.read_csv('BTC-USD.csv')
df = df[["Date", "Close"]]
df.columns = ["ds", "y"]
print(df)

prophet = Prophet()
prophet.fit(df)

future = prophet.make_future_dataframe(periods=700)
print(future)

forecast = prophet.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(200)

from prophet.plot import plot
prophet.plot(forecast, figsize=(20, 10))
