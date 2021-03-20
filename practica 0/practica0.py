"""
Álvaro Beltrán Camacho

"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as m
from sklearn.model_selection import train_test_split
"""
PRÁCTICA 1

EJERCICIO 0 - INTRODUCCIÓN A PYTHON
"""

"""
PARTE 1
"""

#Leer la base de datos Iris

from sklearn import datasets
iris = datasets.load_iris()

#Obtener las características (datos de entrada X) y la clase (y).

X, y = datasets.load_iris(return_X_y=True)

#Quedarse con las dos últimas características (2 últimas columnas de X).

Z=np.zeros((len(X),2))

for i in range(0,len(Z) ) :
    Z[i]=X[i][2:4]
    

    
#Visualizar con un Scatter Plot los datos, coloreando cada clase
#con un color diferente (con rojo, verde y azul), e indicando con
#una leyenda la clase a la que corresponde cada color

color=[' ',' ']
fig, ax = plt.subplots()
for i in range(0,len(y) ):
    
    if y[i]==0:
        color[0]='red'
        color[1]='Setosa'
    if y[i]==1:
        color[0]='green'
        color[1]='Versicolour'
    if y[i]==2:
        color[0]='blue'
        color[1]='Virginica'
    
    ax.scatter(Z[i][0], Z[i][1], c=color[0], label=color[1] if i==0 or i==50 or i==100 else "")

ax.legend()
ax.grid(True)

plt.show()
    
"""
PARTE 3
"""  

#Separar en training (80 % de los datos) y test (20 %) aleatoriamente conservando 
#la proporción de elementos en cada clase tanto en training como en test. Con esto 
#se pretende evitar que haya clases infra-representadas en entrenamiento o test.
#Primero divido por categorias X

X_0=Z[0:50]
X_1=Z[50:100]
X_2=Z[100:150]

y_0=y[0:50]
y_1=y[50:100]
y_2=y[100:150]

#ahora separo aleatoriamente

X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=0.2)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2)

#ahora concateno los vectores

X_train= np.concatenate((X_train_0,X_train_1,X_train_2),axis=0)
X_test= np.concatenate((X_test_0,X_test_1,X_test_2),axis=0)
y_train= np.concatenate((y_train_0,y_train_1,y_train_2),axis=0)
y_test= np.concatenate((y_test_0,y_test_1,y_test_2),axis=0)

#y con esto ya estaría separado en conjuntos de training y test

del X_train_0,X_train_1,X_train_2,y_train_0,y_train_1,y_train_2, X_test_0,X_test_1,X_test_2
del y_test_0,y_test_1,y_test_2,X_0,X_1,X_2,i,color,y_0,y_1,y_2


"""
PARTE 3
"""  

#Obtener 100 valores equiespaciados entre 0 y 2π
valores=np.linspace(0,2*m.pi,100)

#Obtener el valor de sin(x), cos(x) y sin(x) + cos(x) para los 100
#valores anteriormente calculados.

valores_sin,valores_cos,valores_sincos=np.zeros(100),np.zeros(100),np.zeros(100)
for i in range(0,100):
    valores_sin[i]=m.sin(valores[i])
    valores_cos[i]=m.cos(valores[i])
    valores_sincos[i]=valores_sin[i]+valores_cos[i]
    
#Visualizar las tres curvas simultáneamente en el mismo plot (con
#líneas discontinuas en negro, azul y rojo).
    
fig,grafica=plt.subplots()

grafica.plot(valores,valores_sin,c='black',linewidth=2.5, linestyle='--')
grafica.plot(valores,valores_cos,c='blue',linewidth=2.5, linestyle='--')
grafica.plot(valores,valores_sincos,c='red',linewidth=2.5, linestyle='--')

plt.show()
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    