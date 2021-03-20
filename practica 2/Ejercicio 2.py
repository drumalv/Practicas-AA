# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: 
"""
import numpy as np
import matplotlib.pyplot as plt
import math as m


# Fijamos la semilla
np.random.seed(1)

#con esta función aplicamos el ruido especificado en la práctica
def RuidoEtiquetas(x,y):
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for i in range(len(x)) :
        if y[i]==1:
            x1.append(x[i])
            y1.append(1)
        if y[i]==-1:
            x2.append(x[i])
            y2.append(-1)
        
    z=np.arange(len(x1))
    np.random.shuffle(z)
    for i in range(m.trunc(len(z)*0.1)) :
        y1[z[i]]=-1
    z=np.arange(len(x2))
    np.random.shuffle(z)
    for i in range(m.trunc(len(z)*0.1)) :
        y2[z[i]]=1
     
    if len(x1)==0:
        x_aux=x2
    elif len(x2)==0:
        x_aux=x1
    else :
        x_aux=np.concatenate((x1,x2),axis=0)
    if len(y1)==0:
        y_aux=y2
    elif len(y2)==0:
        y_aux=y1
    else :
        y_aux=np.concatenate((y1,y2),axis=0)
    
    return x_aux,y_aux

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

#función para pintar
def PaintResults(x,y,fun,a,b,eInf,eSup):
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
    
    ax.scatter(x1_x, x1_y, c='red', label='+1')
    ax.scatter(x2_x, x2_y, c='blue', label='-1')
    
    x_aux=np.linspace(eInf,eSup,200)
    y_aux=np.linspace(eInf,eSup,200)
    X, Y=np.meshgrid(x_aux,y_aux)
    C=[0]
    ax.contour(X,Y,fun(X,Y,a,b),C,linewidths=3,colors='green')
    
    ax.legend()
    
    plt.show() 

#generando los puntos como en el ejercicio 1
def f1(x,y,a,b):
    return np.sign(y-a*x-b)

x=simula_unif(100,2,[-50,50])
a,b=simula_recta([-50,50])
y=[]

for i in range(len(x)) :
    etiq=f1(x[i][0],x[i][1],a,b)
    y.append(etiq)

x_carac=[]
for i in range(len(x)):
    x_carac.append([1.0,x[i][0],x[i][1]])

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

    
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
        
    return w,it 

#Perceptron inicializado a [0,0,0]
    
print("PERCEPTRON ")

w,it= ajusta_PLA(x_carac, y, np.Inf, np.array([0.0,0.0,0.0]) )
def f2(x,y,a,b):
    return a[2]*y+a[1]*x+a[0]

print("\nGráfico para perceptrón sin ruido con el vector (0,0,0)")
PaintResults(x_carac,y,f2,w,0,-50,50)

print("\nel número de iteraciones es: ", it)
print("Los pesos son: ", w)
iterations1 = []
for i in range(0,10):
    w=np.array([0.0,0.0,0.0])
    w,it= ajusta_PLA(x_carac, y, np.Inf, w)
    iterations1.append(it)

print('\nValor medio de iteraciones necesario para converger usando (0,0,0): {}'.format(np.mean(np.asarray(iterations1))))

# Random initializations
iterations1 = []
for i in range(0,10):
    w=np.array([np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)])
    w,it= ajusta_PLA(x_carac, y, np.Inf, w)
    iterations1.append(it)
    
print('\nValor medio de iteraciones necesario para converger usando pesos aleatorios: {}'.format(np.mean(np.asarray(iterations1))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b
print("PERCEPTRON CON RUIDO")
x_ruido,y_ruido=RuidoEtiquetas(x,y)
x_carac=[]
for i in range(len(x)):
    x_carac.append([1,x_ruido[i][0],x_ruido[i][1]])
    
w,it= ajusta_PLA(x_carac, y_ruido, 1000, np.array([0.0,0.0,0.0]) )
def f2(x,y,a,b):
    return a[2]*y+a[1]*x+a[0]

print("\nGráfico para perceptrón sin con ruido con el vector (0,0,0)")
PaintResults(x_carac,y_ruido,f2,w,0,-50,50)

print("\nel número de iteraciones es: ", it)
print("Los pesos son: ", w)

# Random initializations
iterations2 = []
for i in range(0,10):
    w=np.array([np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)])
    w,it= ajusta_PLA(x_carac, y_ruido, 1000, w)
    iterations2.append(it)

print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations2))))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
np.random.seed(2)
# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

# Funcion para calcular el error
def Err(x,y,w):
    error=0
    for i in range(len(x)):
        error += m.log(1+np.exp(-y[i]*w.dot(x[i])))
    error=error/len(x)
    return error

def dEin(x,y,w):
    aux=[]
    for i in range(len(x)):
        aux.append(-1*(y*x[i]) / (1+np.exp(y*w.dot(x))))
    return aux

#algoritmo de regresión logistica
def sgdRL(x, y, vini, lr):
    
    w=vini
    z=np.arange(len(x))
    it=1
    
    while True:
        w_ant=np.copy(w)
        
        for i in range(len(x)):
            aux=dEin(x[z[i]],y[z[i]],w)
            for j in range(len(x[i])):
                w[j] = w[j] -lr * aux[j]
                
        if np.linalg.norm(w_ant - w) < 0.01:
            break
        
        it+=1
        np.random.shuffle(z)
        
    return w,it

print("REGRESIÓN LOGÍSTICA")

#creamos los datos
    
x=simula_unif(100,2,[0,2])
a,b=simula_recta([0,2])
y=[]

def f3(x,y,a,b):
    return np.sign(y-a*x-b)

for i in range(len(x)) :
    etiq=f3(x[i][0],x[i][1],a,b)
    y.append(etiq)

x_carac=[]
for i in range(len(x)):
    x_carac.append([1.0,x[i][0],x[i][1]])
    
w,it= sgdRL(x_carac, y, np.zeros(3,np.float64), 0.01 )
def f4(x,y,a,b):
    return a[2]*y+a[1]*x+a[0]

print("\nGráfico para REGRESIÓN LOGÍSTICA con el vector (0,0,0)")
PaintResults(x_carac,y,f4,w,0,0,2)

print("\nel número de iteraciones es: ", it)
print("Los pesos son: ", w)
print("\n \t Ein: ", Err(x_carac,y,w) )
input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


print("\nEl error generando 1000 poblaciones de 100 puntos uniformes: ")
err=0
for i in range(1000):    
    x_test=simula_unif(100,2,[0,2])
    y_test=[]
    x_carac_test=[]
    for j in range(len(x_test)) :
        etiq=f3(x_test[j][0],x_test[j][1],a,b)
        y_test.append(etiq)
        x_carac_test.append([1.0,x_test[j][0],x_test[j][1]])
        
        
    err+=Err(x_carac_test,y_test,w)

err= err/1000   
print("\n \t Eout medio: ", err)

print("\nEl error generando 1000 puntos uniformes: ")

x_test=simula_unif(2000,2,[0,2])
y_test=[]
x_carac_test=[]
for j in range(len(x_test)) :
    etiq=f3(x_test[j][0],x_test[j][1],a,b)
    y_test.append(etiq)
    x_carac_test.append([1.0,x_test[j][0],x_test[j][1]])
print("\n \t Eout: ", Err(x_carac_test,y_test,w) )



