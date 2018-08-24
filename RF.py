import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

filePath = r'E:\chenyw\毕业论文\湖南\changde.xls'
tb = pd.read_excel(filePath, sheet_name='changde')
features = tb.iloc[:, 0:6].values
cls = tb.iloc[:, -1].values
print(tb.head())
# print(features[0:5])
# print(cls[0:5])
rfc = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
rfc.fit(features, cls)
print(rfc.score(features, cls))
print(rfc.feature_importances_)
