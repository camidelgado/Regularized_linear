from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import pickle
from sklearn.metrics import mean_squared_error, r2_score

url='https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv'
df=pd.read_csv(url)

df.drop_duplicates()
df.dropna()
x = df.drop(['Heart disease_prevalence', 'Heart disease_Lower 95% CI',
       'Heart disease_Upper 95% CI', 'COPD_prevalence', 'COPD_Lower 95% CI',
       'COPD_Upper 95% CI', 'diabetes_prevalence', 'diabetes_Lower 95% CI',
       'diabetes_Upper 95% CI', 'CKD_prevalence', 'CKD_Lower 95% CI',
       'CKD_Upper 95% CI'],axis=1)
y = df['ICU Beds_x']

X=pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=75)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), Lasso(alpha=2))
pipeline.fit(X_train, y_train)
print(pipeline[1].coef_, pipeline[1].intercept_)

coef_list=pipeline[1].coef_
loc=[i for i, e in enumerate(coef_list) if e != 0]

col_name=df.columns
col_name[loc]
modelo = Lasso(alpha = 0.3,normalize = True)
modelo.fit(X_train,y_train)
predicciones = modelo.predict(X_test)

alphas = modelo.alphas_
coefs = []

for alpha in alphas:
    modelo_temp = Lasso(alpha=alpha, fit_intercept=False, normalize=True)
    modelo_temp.fit(X_train, y_train)
    coefs.append(modelo_temp.coef_.flatten())

fig, ax = plt.subplots(figsize=(7, 3.84))
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_ylim([-15,None])
ax.set_xlabel('alpha')
ax.set_ylabel('coeficientes')
ax.set_title('Coeficientes del modelo en función de la regularización')

alfa_optimo = modelo.alpha_

modelo = Lasso(alpha = alfa_optimo,normalize = True)
modelo.fit(X_train,y_train)


pickle.dump(modelo, open('../models/best_model_reg_linear.pickle', 'wb'))git remote set-url origin