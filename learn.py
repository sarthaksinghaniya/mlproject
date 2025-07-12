import pandas as pd
from io import StringIO

# Your JSON data
data = '''
{
    "employee_name": "james",
    "email": "james@gmail.com",
    "job_profile": [{"title": "team lead", "title2": "sr_dev"}]
}
'''

# Wrap in StringIO
df = pd.read_json(StringIO(data))
print(df)