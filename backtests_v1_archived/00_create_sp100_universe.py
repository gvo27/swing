import os
import pandas as pd

tickers = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BK","BLK","BMY","BRK-B","C","CAT",
    "CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX",
    "DHR","DIS","DUK","EMR","EXC","F","FDX","GD","GE","GILD",
    "GM","GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","ISRG",
    "JNJ","JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ",
    "MET","META","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE",
    "NOW","NVDA","ORCL","PEP","PFE","PG","PM","PYPL","QCOM","RTX",
    "SBUX","SCHW","SO","SPG","T","TGT","TJX","TMO","TMUS","TSLA",
    "TXN","UNH","UPS","USB","V","VZ","WBA","WFC","WMT","XOM"
]

os.makedirs("universes", exist_ok=True)

df = pd.DataFrame({"ticker": tickers})
df.to_csv("universes/sp100.csv", index=False)

print("Created universes/sp100.csv")
print(f"Tickers: {len(df)}")
