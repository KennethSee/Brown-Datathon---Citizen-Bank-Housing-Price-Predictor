import numpy as np
import pandas as pd

df = pd.read_csv('sliced_data.csv')

print(df.head())

def transaction_year(year):
    year_end = year[-2::]
    if int(year_end) < 20:
        year_actual = '20' + year_end
    else:
        year_actual = '19' + year_end
    intYear = int(year_actual)
    return intYear

# obtain year of transaction date
df['transaction_date'] = df['transaction_date'].apply(transaction_year)
# print(df['transaction_date'].head())

# remove whitespace in city
df['prop_city'] = df['prop_city'].apply(lambda x: str(x).replace(' ',''))
print(df['prop_city'].unique())