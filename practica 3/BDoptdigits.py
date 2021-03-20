# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:06:05 2020

@author: alvar
"""

from mpl_toolkits.mplot3d import axes3d
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso, Perceptron, SGDClassifier
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

# Funcion para leer los datos
def read_dataset(name):
    data = np.loadtxt(name, delimiter=",")
    return data[:,:-1], data[:,-1]

# Lectura de los datos de entrenamiento
X_train , y_train = read_dataset('datos/optdigits.tra')

# Lectura de los datos de test
X_test , y_test = read_dataset('datos/optdigits.tes')

"""
#############################
#   DIVISION EN CONJUNTOS   #
#############################
"""

# Uno todas las instancias para luego dividir en los conjuntos indicados en el guión
# 80% train 20% test

X=np.concatenate((X_train,X_test),axis=0)
y=np.concatenate((y_train,y_test),axis=0)

#barajamos las instancias
X, y = shuffle(X, y, random_state=seed)

#escojo los conjuntos de train y test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=seed)

"""
###################################
#      REPRESENTACIÓN   HEATMAP   #
###################################
"""

X_aux=np.copy(X)
y_aux=np.copy(y)

pca = PCA(n_components=2)
pca.fit(X_aux,y_aux)
X_aux_transformed = pca.transform(X_aux)

x1=[]
y1=[]
for i in range(len(X_aux_transformed)):
    x1.append(X_aux_transformed[i][0])
    y1.append(X_aux_transformed[i][1])
    
N_bins = 50

# Construct 2D histogram from data using the 'plasma' colormap
plt.hist2d(x1, y1, bins=N_bins, density=False, cmap='plasma')

# Plot a colorbar with label.
cb = plt.colorbar()
cb.set_label('Number of entries')

# Add title and labels to plot.
plt.title('Heatmap of DataBase')
plt.xlabel('x axis')
plt.ylabel('y axis')

# Show the plot.
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

"""
######################################################
#     PREPROCESAMIENTO     Y     ELECCIÓN MODELO     #
######################################################
"""
#tamaño validacion cruzada
CV=5

#creamos un vector preproc donde incluimos el preprocesado que vamos a realizar y la regularización
preproc=[("var", VarianceThreshold(0.1)),
         ("poly",PolynomialFeatures(1)), 
         ("standardize", StandardScaler())]

#creamos un pipeline de sklearn donde añadiremos uno de los modelos a estudiar
pipe = Pipeline(steps=preproc+[('estimator', LogisticRegression())])

# Añadimos los estimadores que vamos a utilizar y los parametros que vamos a estudiar:
#       si clase de funcion lineal o cuadrática
#       la potencia de la penalización l2 "C"  
#       el solver que voy a usar: Regresion Logistica con SGD (SGDClassifier) ,lbfgs,newton  
params_grid = [ {
                'estimator':[LogisticRegression(max_iter=500)],
                'estimator__solver':['lbfgs','newton-cg'],
                'estimator__C': np.logspace(-4, 4, 3),
                'poly__degree': [1,2]
                },
                {
                'estimator': [Perceptron(random_state = seed)],
                'poly__degree': [1,2]
                }
               # {'estimator':[Any_other_estimator_you_want],
               #  'estimator__valid_param_of_your_estimator':[valid_values]

              ]

print("CON PREPROCESADO: \n")

# entrenamos con crossvalidation y sacamos el mejor con sus parámetros.
# con 5 conjuntos, accuracy y con n_jobs conseguimos que el ordenador use paralelamente los nucleos que pueda.
best_clf = GridSearchCV(pipe, params_grid, scoring = 'accuracy',cv = CV, n_jobs = -1, verbose=1)
best_clf.fit(X_train, y_train)

#con esto conseguimos pasar a una tabla los resultados obtenidos
results=pd.DataFrame(best_clf.cv_results_)
print("\n He estudiado estos modelos a los que asigno un índice:\n")
print(results[['params','param_estimator__C']])

input("\n--- Pulsar tecla para continuar ---\n")

print("Resultados: \n")
print(results[['rank_test_score', 'mean_fit_time','mean_test_score']])

input("\n--- Pulsar tecla para continuar ---\n")

scores=results[['split0_test_score', 'split1_test_score','split2_test_score','split3_test_score','split4_test_score']][0:15].to_numpy()

fig,ax1=plt.subplots()
for i in np.arange(len(scores)):
    ax1.plot(np.arange(CV),scores[i],label='modelo '+str(i))

ax1.axis([0.0,CV+1.0,0.85,1.0])
plt.xlabel('Conjuntos CV')
plt.ylabel('Accuracy')
plt.title('Gráfico Accuracy')
ax1.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Mejor modelo:\n",best_clf.best_params_)
print("Precisión en training:", 100.0 * best_clf.score(X_train, y_train))
print("Precisión en test: ",100.0 * best_clf.score(X_test, y_test))

input("\n--- Pulsar tecla para continuar ---\n")

"""
###############################
#   CON REGULARIZACIÓN LASSO  #
###############################
"""
#Añadimos Lasso
preproc=[("var", VarianceThreshold(0.1)),         
         ("lasso", SelectFromModel(Lasso())),
         ("poly",PolynomialFeatures(1)), 
         ("standardize", StandardScaler())]

#creamos un pipeline de sklearn donde añadiremos uno de los modelos a estudiar
pipe = Pipeline(steps=preproc+[('estimator', LogisticRegression())])

# Añadimos los estimadores que vamos a utilizar y los parametros que vamos a estudiar:
#       si clase de funcion lineal o cuadrática
#       la potencia de la penalización l2 "C"    
params_grid = [ {
                'estimator':[LogisticRegression(max_iter=500)],
                'estimator__solver':['lbfgs','newton-cg'],
                'estimator__C': np.logspace(-4, 4, 3),
                'poly__degree': [1,2]
                },
                {
                'estimator': [Perceptron(random_state = seed)],
                'poly__degree': [1,2]
                }
               # {'estimator':[Any_other_estimator_you_want],
               #  'estimator__valid_param_of_your_estimator':[valid_values]

              ]

print("CON PREPROCESADO Y REGULARIZACION: \n")

# entrenamos con crossvalidation y sacamos el mejor con sus parámetros.
best_clf = GridSearchCV(pipe, params_grid, scoring = 'accuracy',cv = 5, n_jobs = -1)
best_clf.fit(X_train, y_train)


#con esto conseguimos pasar a una tabla los resultados obtenidos
results=pd.DataFrame(best_clf.cv_results_)
print("\n He estudiado estos modelos a los que asigno un índice:\n")
print(results[['params','param_estimator__C']])

input("\n--- Pulsar tecla para continuar ---\n")

print("Resultados: \n")
print(results[['rank_test_score', 'mean_fit_time','mean_test_score']])

input("\n--- Pulsar tecla para continuar ---\n")

scores=results[['split0_test_score', 'split1_test_score','split2_test_score','split3_test_score','split4_test_score']][0:15].to_numpy()

fig,ax1=plt.subplots()
for i in np.arange(len(scores)):
    ax1.plot(np.arange(CV),scores[i],label='modelo '+str(i))

ax1.axis([0.0,CV+1.0,0.7,1.0])
plt.xlabel('Conjuntos CV')
plt.ylabel('Accuracy')
plt.title('Gráfico Accuracy con LASSO')
ax1.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Mejor modelo:\n",best_clf.best_params_)
print("Precisión en training:", 100.0 * best_clf.score(X_train, y_train))
print("Precisión en test: ",100.0 * best_clf.score(X_test, y_test))

input("\n--- Pulsar tecla para continuar ---\n")



