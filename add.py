import pandas as pd
from io import StringIO

data = '''
{
    "employee_name": "james",
    "email": "james@gmail.com",
    "job_profile": [{"title": "team lead", "title2": "sr_dev"}]
}
'''

df = pd.read_json(StringIO(data))
# Flatten job_profile into separate columns
job_df = pd.json_normalize(df['job_profile'])
final_df = pd.concat([df.drop(columns=['job_profile']), job_df], axis=1)

print(final_df) 

final_df.to_json(orient = "records")
