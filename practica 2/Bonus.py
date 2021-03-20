# -*- coding: utf-8 -*-
###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

import numpy as np
import matplotlib.pyplot as plt
import math as m

np.random.seed(1)

def PaintResults(x,y,fun,a,b,tipo):
    x1_x=[]
    x2_x=[]
    x1_y=[]
    x2_y=[]
    fig, ax = plt.subplots()
    for i in range(len(x)) :
        if y[i]==1:
            x1_x.append(x[i][1])
            x1_y.append(x[i][2])
        if y[i]==-1:
            x2_x.append(x[i][1])
            x2_y.append(x[i][2])
    
    ax.scatter(x2_x, x2_y, c='red', label='4')
    ax.scatter(x1_x, x1_y, c='blue', label='8')
    
    
    xrange = np.arange(0, 1, 0.025)
    yrange = np.arange(-7, 0, 0.025)
    X, Y = np.meshgrid(xrange,yrange)
    C=[0]
    ax.contour(X,Y,fun(X,Y,a,b),C,linewidths=3,colors='green')
    ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos'+tipo)
    ax.legend()
    
    plt.show() 
    
# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION  SGD
print ('\nGRADIENTE DESCENDIENTE ESTOCÁSTICO\n')
# Funcion para calcular el error
def ErrSGD(x,y,w):
    error=0
    for i in range(len(x)):
        error += (w.dot(x[i])-y[i])**2
    error=error/len(x)
    return error

# Funcion para calcular el error
def dErr_j(x,y,w,j,tam_minibatch):
    error=0
    #creo un vector z con x e y para aleatorizar los datos escogidos
    z=np.arange(len(x))
    np.random.shuffle(z)
    
    for i in range(tam_minibatch):
        error += x[z[i]][j]*(w.dot(x[z[i]])-y[z[i]])
    error=(2*error)/tam_minibatch
    return error


# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):	
    it=0
    w=np.zeros(len(x[0]))
    while(it<max_iters ):
        w_ant=np.copy(w)
        for j in range(len(w) ):
            w[j] = w[j] - lr*dErr_j(x,y,w_ant,j,tam_minibatch)
        it+=1
    
    return w


wSGD = sgd(x,y,0.01,500,32)
def f2(x,y,a,b):
    return a[2]*y+a[1]*x+a[0]
PaintResults(x,y,f2,wSGD,0,' TRAINING')

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", ErrSGD(x,y,wSGD))
PaintResults(x_test,y_test,f2,wSGD,0,' TEST')
print ("Etest: ", ErrSGD(x_test,y_test,wSGD))

input("\n--- Pulsar tecla para continuar ---\n")


#POCKET ALGORITHM
  
print ('\n POCKET ALGORITHM\n')

def ErrP(x,y,w):
    error=0
    for i in range(len(x)):
        if(np.sign(w.dot(x[i]) )!=y[i]): 
            error += 1
    error=error/len(x)
    return error

def ajusta_PLA(x, y, max_iter, vini):
    
    w=vini
    it=0
    cambio=True
    while cambio and it<max_iter:
        cambio=False
        for i in range(len(x)):
            if np.sign(w.dot(x[i]))!=y[i]:
                for j in range(len(x[i])):
                    w[j]=w[j]+y[i]*x[i][j]
                cambio=True
        it+=1
        
    return w

def pocket(x,y,vini, max_iteration):
    w = np.copy(vini)
    best_error = ErrP(x,y,w)
    for i in range(0, max_iteration):
        w_new=ajusta_PLA(x, y, 100, np.copy(w)) 
        error=ErrP(x,y,w_new)
        if error < best_error :
            w=np.copy(w_new)
            best_error = error
    return w



    
print ('\n Usando el peso inicial de SGD\n')

wP = pocket(x,y,wSGD,20)
def f4(x,y,a,b):
    return a[2]*y+a[1]*x+a[0]
PaintResults(x,y,f4,wP,0,' TRAINING')

print ('Bondad del resultado para Pocket:\n')
print ("Ein: ", ErrP(x,y,wP))
PaintResults(x_test,y_test,f4,wP,0,' TEST')
print ("Etest: ", ErrP(x_test,y_test,wP))

input("\n--- Pulsar tecla para continuar ---\n")

#COTA SOBRE EL ERROR

print ('Cotas sobre Eout al 95% de probabilidad:\n')
print ("Usando Ein -> Eout <= ", ErrP(x,y,wP)+m.sqrt( (1/len(x) * m.log(2/0.05) ) ) )
print ("Usando Etest -> Eout <= ", ErrP(x_test,y_test,wP)+m.sqrt( (1/len(x_test) * m.log(2/0.05) ) ) )















