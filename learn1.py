import pandas as pd

url = "https://www.fdic.gov/bank-failures/failed-bank-list"  # ← replace with actual working URL
dfs = pd.read_html(url)
print(dfs)
