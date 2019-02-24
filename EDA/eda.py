import numpy as np
import pandas as pd
# from sklearn.preprocessing import LabelBinarizer

df = pd.read_csv('training_data.csv')
print(df.census_tract.unique())
print(df.census_tract.nunique())

def transaction_year(year):
    year_end = year[-2::]
    if int(year_end) < 20:
        year_actual = '20' + year_end
    else:
        year_actual = '19' + year_end
    intYear = int(year_actual)
    return intYear


cat_var = [14,17,20,21,24,46,47,48,49,51,52,53,54,55,70]
total_var = [14,15,17,20,21,23,24,26,27,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,69,70,71,72]
noncat_var = np.setdiff1d(total_var, cat_var, assume_unique=True)

df_noncat = df.iloc[:, noncat_var]

# print(list(df_noncat))

# obtain year of transaction date
df_noncat['transaction_date'] = df_noncat['transaction_date'].apply(transaction_year)

df_cat = df.iloc[:, cat_var]

df_dummies = pd.get_dummies(df_cat)

df_clean = pd.concat([df_noncat, df_dummies],axis=1)

# remove 0 and NaN values from sale amount and zip code
df_clean.dropna(subset=['sale_amt','prop_zip_code'], how='any', inplace=True)
df_clean = df_clean[df_clean['sale_amt'] != 0]

# convert zip code values into int
df_clean['prop_zip_code'] = df_clean['prop_zip_code'].apply(lambda x: int(x))
# replace all NaN values as 0
df_clean.fillna(0, inplace=True)
# remove NaN values from year built, effective year built and stories_cd
df_clean.dropna(subset=['year_built', 'effective_year_built', 'stories_cd'], how='any', inplace=True)
df_clean['year_built'] = df_clean['year_built'].apply(lambda x: int(x))
df_clean['effective_year_built'] = df_clean['effective_year_built'].apply(lambda x: int(x))
# remove Split Entry values from stories_cd
# df_clean = df_clean[df_clean['stories_cd'] != 'Split Entry']
# df_clean['stories_cd'] = df_clean['stories_cd'].apply(lambda x: int(float(x)))

df_clean.to_csv('training_data_full_clean.csv')
print(df_clean.shape)
print(df_clean.isnull().values.any())