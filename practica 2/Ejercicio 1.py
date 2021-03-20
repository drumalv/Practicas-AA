# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as m


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out


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

#ejercicio 1 
    
def paintEj1(z):
    z=np.transpose(z)
    fig, ax = plt.subplots()
    ax.scatter(z[0], z[1])
    plt.show() 
    z=np.transpose(z)

print("\nEjercicio 1.a gráfica usando simula_unif")

z=simula_unif(50,2,[-50,50])

paintEj1(z)

input("\n--- Pulsar tecla para continuar ---\n")

print("\nEjercicio 1.b gráfica usando simula_gaus")

z=simula_gaus(50,2,[5,7])

paintEj1(z)

input("\n--- Pulsar tecla para continuar ---\n")

#ejercicio 2

def PaintResults(x,y,fun,a,b):
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
    
    ax.scatter(x1_x, x1_y, c='red', label='+1')
    ax.scatter(x2_x, x2_y, c='blue', label='-1')
    
    x_aux=np.linspace(-50,50,200)
    y_aux=np.linspace(-50,50,200)
    X, Y=np.meshgrid(x_aux,y_aux)
    C=[0]
    ax.contour(X,Y,fun(X,Y,a,b),C,linewidths=3,colors='green')
    
    ax.legend()
    
    plt.show() 
    
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
            
def PorcentajeError(x,y,fun,a,b):
    num_fallos_pos=0
    num_fallos_neg=0
    num_pos=0
    num_neg=0
    for i in range(len(x)):
        if y[i]==1:
            num_pos+=1
            if fun(x[i][0],x[i][1],a,b)!=y[i]:
                num_fallos_pos+=1
        if y[i]==-1:
            num_neg+=1
            if fun(x[i][0],x[i][1],a,b)!=y[i]:
                num_fallos_neg+=1
                
    if num_pos==0:
        num_pos=1
    if num_neg==0:
        num_neg=1
        
    return (num_fallos_pos/num_pos)*100,(num_fallos_neg/num_neg)*100

 
print("\nEjercicio 2.a gráfica etiquetada sin ruido")

def f1(x,y,a,b):
    return np.sign(y-a*x-b)

x=simula_unif(100,2,[-50,50])
a,b=simula_recta([-50,50])
y=[]

for i in range(len(x)) :
    etiq=f1(x[i][0],x[i][1],a,b)
    y.append(etiq)
PaintResults(x,y,f1,a,b)
print("error: ",PorcentajeError(x,y,f1,a,b))
input("\n--- Pulsar tecla para continuar ---\n")

print("\nEjercicio 2.b gráfica etiquetada con ruido")

x,y=RuidoEtiquetas(x,y)
PaintResults(x,y,f1,a,b)
print("error: ",PorcentajeError(x,y,f1,a,b))
input("\n--- Pulsar tecla para continuar ---\n")

#Ejercicio 

print("\nEjercicio 2.c.1 gráfica ")

def f2(x,y,a,b):
    return np.sign(  (x-10)**2 + (y-20)**2 -400  )

PaintResults(x,y,f2,a,b)
print("error: ",PorcentajeError(x,y,f2,a,b))
input("\n--- Pulsar tecla para continuar ---\n")

print("\nEjercicio 2.c.2 gráfica ")

def f3(x,y,a,b):
    return np.sign(  0.5*(x+10)**2 + (y-20)**2 -400  )

PaintResults(x,y,f3,a,b)
print("error: ",PorcentajeError(x,y,f3,a,b))
input("\n--- Pulsar tecla para continuar ---\n")

print("\nEjercicio 2.c.3 gráfica")

def f4(x,y,a,b):
    return np.sign(  0.5*(x-10)**2 - (y+20)**2 -400  )

PaintResults(x,y,f4,a,b)
print("error: ",PorcentajeError(x,y,f4,a,b))
input("\n--- Pulsar tecla para continuar ---\n")

print("\nEjercicio 2.c.4 gráfica")

def f5(x,y,a,b):
    return np.sign(  y-20*x**2 -5*x +3  )


PaintResults(x,y,f5,a,b)
print("error: ",PorcentajeError(x,y,f5,a,b))
input("\n--- Pulsar tecla para continuar ---\n")
