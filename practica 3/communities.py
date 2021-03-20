# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:34:59 2020

@author: alvar
"""

from mpl_toolkits.mplot3d import axes3d
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

seed=1
np.random.seed(seed)

"""
########################
#   LECTURA DE DATOS   #
########################
"""

def read_dataset(filename):
    data = np.genfromtxt(filename, delimiter = ",", dtype = np.double)
    return data[:, :-1], data[:, -1]

X , y = read_dataset('datos/communities.data')

"""
#############################
#   DIVISION EN CONJUNTOS   #
#############################
"""

#quito como dice en la documentacion de la BD los 5 primeros atributos
X=np.delete(X, 1, axis=1)
X=np.delete(X, 1, axis=1)
X=np.delete(X, 1, axis=1)
X=np.delete(X, 1, axis=1)
X=np.delete(X, 1, axis=1)

#barajamos las instancias
X, y = shuffle(X, y, random_state=seed)

#escojo los conjuntos de train y test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=seed)

"""
###################################
#      REPRESENTACIÓN   3D        #
###################################
"""

X_aux=np.copy(X)
y_aux=np.copy(y)

s=SimpleImputer(missing_values=np.nan, strategy='mean')
s.fit(X_aux,y_aux)
X_aux_transformed = s.transform(X_aux)

pca = PCA(n_components=3)
pca.fit(X_aux_transformed,y_aux)
X_aux_transformed = pca.transform(X_aux_transformed)

x1=[]
y1=[]
z1=[]
for i in range(len(X_aux_transformed)):
    x1.append(X_aux_transformed[i][0])
    y1.append(X_aux_transformed[i][1])
    z1.append(X_aux_transformed[i][2])
    
fig=plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.scatter3D(x1,y1,z1,label='puntos')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

"""
######################################################
#     PREPROCESAMIENTO     Y     ELECCIÓN MODELO     #
######################################################
"""


#tamaño validacion cruzada
CV=5

#creamos un vector preproc donde incluimos el preprocesado que vamos a realizar 
preproc=[("simpleimputer",SimpleImputer(missing_values=np.nan, strategy='mean')),
         ("var", VarianceThreshold(0.01)),
         ("poly",PolynomialFeatures(1)), 
         ("standardize", StandardScaler())]

# Añadimos los estimadores que vamos a utilizar y los parametros que vamos a estudiar:
#       si clase de funcion lineal o cuadrática
#       la potencia de la penalización l1 o l2 "alpha"  
#       si uso penalización l1 o l2
pipe=Pipeline(preproc + [('estimator', SGDRegressor())])
params_grid=[
              {"estimator":[SGDRegressor(max_iter=500,loss='squared_loss')],
               "estimator__penalty":['l1','l2'],
               "estimator__alpha":np.logspace(-4, 4, 3),
               "poly__degree":[1,2]
               },
               {"estimator":[LinearRegression()],
               "poly__degree":[1,2]
               },     
               {"estimator":[Ridge()],
               "poly__degree":[1,2],
               "estimator__alpha":np.logspace(-4, 4, 3),
               },
               {"estimator":[Lasso()],
               "poly__degree":[1,2],
               "estimator__alpha":np.logspace(-4, 4, 3)
               }
               # {'estimator':[Any_other_estimator_you_want],
               #  'estimator__valid_param_of_your_estimator':[valid_values]
            ]

best_clf = GridSearchCV(pipe, params_grid, scoring = 'neg_mean_squared_error',cv = CV, n_jobs = -1, verbose=1)
best_clf.fit(X_train, y_train)

print("CON PREPROCESADO: \n")

#con esto conseguimos pasar a una tabla los resultados obtenidos
results=pd.DataFrame(best_clf.cv_results_)
print("\n He estudiado estos modelos a los que asigno un índice:\n")
print(results[['params','param_poly__degree']])

input("\n--- Pulsar tecla para continuar ---\n")

print("Resultados: \n")
print(results[['rank_test_score', 'mean_fit_time','mean_test_score']])

input("\n--- Pulsar tecla para continuar ---\n")

#escojo los 10 modelos con mejor score
scores=(results.sort_values('mean_test_score',ascending=False))[['split0_test_score', 'split1_test_score','split2_test_score','split3_test_score','split4_test_score','rank_test_score']][0:10].to_numpy()

fig,ax1=plt.subplots()
for i in np.arange(len(scores)):
    ax1.plot(np.arange(CV),(scores[:, :-1])[i],label='rank '+str( (scores[:, -1])[i]) )

ax1.axis([0.0,CV+1.0,-0.075,0.0])
plt.xlabel('Conjuntos CV')
plt.ylabel('ECM')
plt.title('Gráfico ECM')
ax1.legend()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")


print("Mejor modelo:\n",best_clf.best_params_)
print("Error en training:", -100.0 * best_clf.score(X_train, y_train))
print("Error en test: ",-100.0 * best_clf.score(X_test, y_test))



























