import yfinance as yf

from datetime import datetime, timedelta

import pandas as pd

import os

# Danh sách mã cổ phiếu

symbols = [

    "AAPL"

]
all_data = []
for symbol in symbols:

    stock = yf.Ticker(symbol)
    today = datetime.now()
    yesterday = today - timedelta(days=1)



    data = stock.history(

        start=yesterday.strftime("%Y-%m-%d"),

        end=today.strftime("%Y-%m-%d"),

        interval="1m"

    )



    data["Symbol"] = symbol

all_data.append(data)
final_df = pd.concat(all_data)
final_df.reset_index(inplace=True)
final_df.to_parquet("stock_data_1d.parquet", index=False)

df = pd.read_parquet("stock_data_1d.parquet")
print(df)