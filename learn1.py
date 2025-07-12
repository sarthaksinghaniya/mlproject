import pandas as pd

url = "https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/"
dfs = pd.read_html(url)  # now works because lxml is installed
print(dfs[0])  # print first table
