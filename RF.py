import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

filePath = r'E:\chenyw\毕业论文\甘肃\lanzhou.xls'
tb = pd.read_excel(filePath, sheet_name='lanzhou')
features = tb.iloc[:, 0:9].values
est = tb.iloc[:, 9].values
print(tb.head())
# print(features[0:5])
# print(est[0:5])

rfRegress = RandomForestRegressor(n_estimators=5000, n_jobs=-1)
rfRegress.fit(features, est)
print(rfRegress.feature_importances_)
print(rfRegress.score(features, est))
