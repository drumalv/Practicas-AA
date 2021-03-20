# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import numpy as np
import matplotlib.pyplot as plt
import math as m
import random

#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([ 1,datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

def PaintLinearResults(x,y,w,x0w,x1w):
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
    
    ax.scatter(x1_x, x1_y, c='red', label='5')
    ax.scatter(x2_x, x2_y, c='blue', label='1')
    
    regres_x=[x0w,x1w]
    regres_y=[((w[0]/-w[2])+(w[1]/-w[2])*x0w),((w[0]/-w[2])+(w[1]/-w[2])*x1w)]
    
    ax.plot(regres_x,regres_y,linewidth=3.5,label='recta regresión',color='green')
    
    ax.legend()
    ax.set_xlabel('Intensidad')
    ax.set_ylabel('Simetría')
    
    plt.show() 
    
def PaintQuadraticResults(x,y,w,f):
    x1_x=[]
    x2_x=[]
    x1_y=[]
    x2_y=[]
    fig, ax = plt.subplots()
    x_aux=np.linspace(-1,1,100)
    y_aux=np.linspace(-1,1,100)
    X, Y=np.meshgrid(x_aux,y_aux)
    C=[0]
    ax.contour(X,Y,f(X,Y,w),C,linewidths=5,colors='green')

    for i in range(len(x)) :
        if y[i]==1:
            x1_x.append(x[i][0])
            x1_y.append(x[i][1])
        if y[i]==-1:
            x2_x.append(x[i][0])
            x2_y.append(x[i][1])
    
    ax.scatter(x1_x, x1_y, c='red', label='tipo 1')
    ax.scatter(x2_x, x2_y, c='blue', label='tipo 2')
    ax.legend()
    
    plt.show() 
	
# Funcion para calcular el error
def Err(x,y,w):
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
    
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
    w=np.zeros(len(x[0]))
    w=np.linalg.pinv(x).dot(y) 
    #calcula la pseudoinversa y la multiplica por y
    return w
	
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w = sgd(x,y,0.01,100,32)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test,y_test,w))
input("\n--- Pulsar tecla para continuar ---\n")
#pintar resultados
print ('\nGráfico para datos train de grad. descendente estocastico:\n')
PaintLinearResults(x,y,w,0.0,0.6)
print ('Gráfico para datos test de grad. descendente estocastico:\n')   
PaintLinearResults(x_test,y_test,w,0.0,0.6)     
        


input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test,y_test,w))
input("\n--- Pulsar tecla para continuar ---\n")
#pintar resultados
print ('\nGráfico para datos train de grad. descendente estocastico:\n')
PaintLinearResults(x,y,w,0.0,0.6)
print ('Gráfico para datos test de grad. descendente estocastico:\n')   
PaintLinearResults(x_test,y_test,w,0.0,0.6) 
input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))
	

def PaintResults(x,y):
    x1_x=[]
    x2_x=[]
    x1_y=[]
    x2_y=[]
    fig, ax = plt.subplots()
    for i in range(len(x)) :
        if y[i]==1:
            x1_x.append(x[i][0])
            x1_y.append(x[i][1])
        if y[i]==-1:
            x2_x.append(x[i][0])
            x2_y.append(x[i][1])
    
    ax.scatter(x1_x, x1_y, c='red', label='tipo 1')
    ax.scatter(x2_x, x2_y, c='blue', label='tipo 2')
    
    ax.legend()
    
    plt.show() 
# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]	

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')

x=simula_unif(1000, 2, 1)
x=np.transpose(x)
fig2, ax2 = plt.subplots()
ax2.scatter(x[0], x[1], c='blue', label='muestra entrenamiento')
ax2.legend()
plt.show()
x=np.transpose(x)

# b) Consideremos la función f(x1, x2) que usaremos
#para asignar una etiqueta a cada punto de la muestra anterior. Introducimos
#ruido sobre las etiquetas cambiando aleatoriamente el signo de un 10 % de las
#mismas. Pintar el mapa de etiquetas obtenido.

def f(x1,x2):
    return np.sign( ((x1-0.2)**2  + x2**2 - 0.6) )

#asignamos etiquetas
y=[]
for i in range(len(x)) :
    y.append(f(x[i][0],x[i][1]))
#cambiamos 10%
    
for i in range(m.trunc(len(x)*0.1)) :
    a=random.randint(0,len(x)-1)
    y[a]=-y[a]

PaintResults(x,y)
input("\n--- Pulsar tecla para continuar ---\n")

#c) Usando como vector de características (1, x1, x2) ajustar un modelo de regresion
#lineal al conjunto de datos generado y estimar los pesos w. Estimar el error de
#ajuste Ein usando Gradiente Descendente Estocástico (SGD).

#generar (1, x1, x2)
x_carac=[]
for i in range(len(x)) :
    x_carac.append([])
    x_carac[i].append(1)
    x_carac[i].append(x[i][0])
    x_carac[i].append(x[i][1])

w = sgd(x_carac,y,0.1,100,32)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x_carac,y,w))

#pintar resultados
x0w=(w[0]/-w[1])+(w[2]/-w[1])*1.0
x1w=(w[0]/-w[1])+(w[2]/-w[1])*-1.0
print ('\nGráfico para datos train de grad. descendente estocastico:\n')
PaintLinearResults(x_carac,y,w,x0w,x1w)
input("\n--- Pulsar tecla para continuar ---\n")
# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

Ein_media=0
Eout_media=0

for i in range(1000):
    #calculamos el train
    x=simula_unif(1000, 2, 1)
    #asignamos etiquetas
    y=[]
    for i in range(len(x)) :
        y.append(f(x[i][0],x[i][1]))
    #cambiamos 10%
    
    for i in range(m.trunc(len(x)*0.1)) :
        a=random.randint(0,len(x)-1)
        y[a]=-y[a]
        
    x_carac=[]
    for i in range(len(x)) :
        x_carac.append([])
        x_carac[i].append(1)
        x_carac[i].append(x[i][0])
        x_carac[i].append(x[i][1])
    
    w = sgd(x_carac,y,0.1,100,32)
    
    Ein_media+=Err(x_carac,y,w)
    
    #hacemos el test
    
    x=simula_unif(1000, 2, 1)
    
    #asignamos etiquetas
    y=[]
    for i in range(len(x)) :
        y.append(f(x[i][0],x[i][1]))
    #cambiamos 10%
    
    for i in range(m.trunc(len(x)*0.1)) :
        a=random.randint(0,len(x)-1)
        y[a]=-y[a]
        
    x_carac=[]
    for i in range(len(x)) :
        x_carac.append([])
        x_carac[i].append(1)
        x_carac[i].append(x[i][0])
        x_carac[i].append(x[i][1])
        
    Eout_media+=Err(x_carac,y,w)
    
Eout_media=Eout_media/1000
Ein_media=Ein_media/1000

print ('\nErrores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)
input("\n--- Pulsar tecla para continuar ---\n")

#repetir el experimento anterior ahora usando Φ2(x) = (1, x1, x2, x1*x2, x1**2, x2**2).

print ('\nCambiamos las características y repetimos el experimento:\n')

x=simula_unif(1000, 2, 1)
#asignamos etiquetas
y=[]
for i in range(len(x)) :
    y.append(f(x[i][0],x[i][1]))
#cambiamos 10%

for i in range(m.trunc(len(x)*0.1)) :
    a=random.randint(0,len(x)-1)
    y[a]=-y[a]

x_carac=[]
for i in range(len(x)) :
    x_carac.append([])
    x_carac[i].append(1)
    x_carac[i].append(x[i][0])
    x_carac[i].append(x[i][1])
    x_carac[i].append(x[i][0]*x[i][1])
    x_carac[i].append(x[i][0]**2)
    x_carac[i].append(x[i][1]**2)

w = sgd(x_carac,y,0.1,100,32)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x_carac,y,w))
def H(x1,x2,t):
    return t[0]+t[1]*x1+t[2]*x2+t[3]*x1*x2+t[4]*(x1**2)+t[5]*(x2**2)

PaintQuadraticResults(x,y,w,H)
input("\n--- Pulsar tecla para continuar ---\n")

#1000 veces


Ein_media=0
Eout_media=0

for i in range(1000):
    #calculamos el train
    x=simula_unif(1000, 2, 1)
    #asignamos etiquetas
    y=[]
    for i in range(len(x)) :
        y.append(f(x[i][0],x[i][1]))
    #cambiamos 10%
    
    for i in range(m.trunc(len(x)*0.1)) :
        a=random.randint(0,len(x)-1)
        y[a]=-y[a]
        
    x_carac=[]
    for i in range(len(x)) :
        x_carac.append([])
        x_carac[i].append(1)
        x_carac[i].append(x[i][0])
        x_carac[i].append(x[i][1])
        x_carac[i].append(x[i][0]*x[i][1])
        x_carac[i].append(x[i][0]**2)
        x_carac[i].append(x[i][1]**2)
    
    w = sgd(x_carac,y,0.1,100,32)
    
    Ein_media+=Err(x_carac,y,w)
    
    #hacemos el test
    
    x=simula_unif(1000, 2, 1)
    
    #asignamos etiquetas
    y=[]
    for i in range(len(x)) :
        y.append(f(x[i][0],x[i][1]))
    #cambiamos 10%
    
    for i in range(m.trunc(len(x)*0.1)) :
        a=random.randint(0,len(x)-1)
        y[a]=-y[a]
        
    x_carac=[]
    for i in range(len(x)) :
        x_carac.append([])
        x_carac[i].append(1)
        x_carac[i].append(x[i][0])
        x_carac[i].append(x[i][1])
        x_carac[i].append(x[i][0]*x[i][1])
        x_carac[i].append(x[i][0]**2)
        x_carac[i].append(x[i][1]**2)
        
    Eout_media+=Err(x_carac,y,w)
    
Eout_media=Eout_media/1000
Ein_media=Ein_media/1000

print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)



