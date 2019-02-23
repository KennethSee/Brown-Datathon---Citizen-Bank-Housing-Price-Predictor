import numpy as np
import pandas as pd

rawData = pd.read_csv('datathon_propattributes.csv')
slicedData = rawData.loc[rawData.IsTraining == 1]

print(slicedData.shape)
slicedData.to_csv('training_data.csv')