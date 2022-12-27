import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

np.random.seed(1)
path = str(os.path.dirname(os.path.realpath(__file__)))

df_x = pd.read_csv(path + '\\X_train.csv')
df_y = pd.read_csv(path + '\\Y_train.csv')
df_gray = pd.read_csv(path + '\\2022 01 28 V2 Gray Data.csv')
df_census = pd.read_csv(path + '\\2022 01 28 Census Data by Geoid.csv')

df_train = pd.merge(df_x, df_y, on = 'key')

# print(df_gray)
# df_gray = df_gray.add_prefix('NCG ')
# print(df_gray.columns.values)

# print(df_train)

df_gray.rename(columns = {'geoid_hashed': 'student_geoid_hashed'}, inplace = True)
df_train = pd.merge(df_train, df_gray, on='student_geoid_hashed', how = 'left')

df_census.rename(columns = {'geoid_hashed': 'student_geoid_hashed'}, inplace = True)
df_train = pd.merge(df_train, df_census, on='student_geoid_hashed', how = 'left')

# print(df_train.columns.values)


Y_train = np.asarray(df_train['starts'])
drop_indices = ['key', 'starts', 'student_geoid_hashed', 'nearest_campus_hashed_geoid']
drop_indices += ['student_state', 'nearest_campus_state', 'nearest_campus_id']
X_train = np.asarray(df_train.drop(drop_indices, axis = 1))
# print(X_train.shape)
# print(X_train)
# print(Y_train.shape)

X_train, X_test, Y_train, Y_test = train_test_split( X_train, Y_train, test_size=0.2, random_state=1)

# reg = LazyRegressor(verbose=1)
# models, pred = reg.fit(X_train, X_test, Y_train, Y_test)
# print(models)

# reg = XGBRegressor()
reg = LGBMRegressor()
# reg = HistGradientBoostingRegressor()
# # reg = GradientBoostingRegressor()

reg.fit(X_train, Y_train)
Y_predict = reg.predict(X_test)
score = reg.score(X_test, Y_test)
print(Y_predict)
print(score)
importances = reg.feature_importances_
print(importances)
print(type(importances))

# print(X_train)
# print(Y_train)
# print(X_test)
# print(Y_test)
