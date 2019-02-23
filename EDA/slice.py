import numpy as np
import pandas as pd

rawData = pd.read_csv('datathon_propattributes.csv')
slicedData = rawData[:100]

slicedData.to_csv('sliced_data.csv')