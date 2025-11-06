from dotenv import load_dotenv
import os
import pandas as pd
import requests

load_dotenv()
av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")



# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey={av_api_key}'
r = requests.get(url)
data = r.json()
print(data["Time Series (Digital Currency Daily)"])


def structure_data(data):
    """
    Converts raw Alpha Vantage crypto daily data into a sorted closing price series.
    """
    df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float).sort_index()
    return df["4. close"]


def fetch_data(tickers):
    """
    Fetch closing prices for multiple tickers and combine them into one DataFrame.
    Each column represents one ticker.
    """
    market = "USD"
    combined_df = pd.DataFrame()

    for ticker in tickers:
        url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={ticker}&market={market}&apikey={av_api_key}"
        r = requests.get(url)
        data = r.json()

        # Extract and structure the series
        closing_price_series = structure_data(data)
        closing_price_series.name = ticker  # Set the column name

        # Combine using outer join to align by date
        combined_df = pd.concat([combined_df, closing_price_series], axis=1)

    return combined_df
