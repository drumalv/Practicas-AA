# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import math as m
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


#------------------------------Ejercicio 1 -------------------------------------#



"""APARTADO 1"""
#Este es el algoritmo de gradiente descendiente visto en teoria, que paramos cuando llegamos 
#a un número máximo de iteraciones o cuando la funcion es menor que epsilon.
def gd(w, lr, grad_fun, fun, epsilon, max_iters = 50):		
    it=0
    while(it<max_iters and fun(w)>=epsilon):
        w_ant=np.copy(w)
        for j in range(len(w) ):
            w[j] = w[j] - lr*grad_fun(w_ant)[j]
        it+=1
    return w,it

"""APARTADO 2"""

# Funcion E(u,v)
def E(w): 
	return ( w[0]*m.exp(w[1])-2*w[1]*m.exp(-w[0]) )**2
			 
# Derivada parcial de E respecto de u
def Eu(w):
    u=w[0]
    v=w[1]
    return 2*( u*m.exp(v)-2*v*m.exp(-u) )*( m.exp(v) + 2*v*m.exp(-u) )

# Derivada parcial de E respecto de v
def Ev(w):
	return 2*( w[0]*m.exp(w[1])-2*w[1]*m.exp(-w[0]) )*( w[0]*m.exp(w[1])-2*m.exp(-w[0]) )
	
# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')
print ('\nApartado 2.b\n')

w,num_ite=gd([1.0,1.0], 0.1, gradE, E, 1e-14,50)
print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar tecla para continuar ---\n")

print ('\nApartado 2.c\n')
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")
"""
#------------------------------Apartado 3 -------------------------------------#
"""
# Funcion f(x,y)== (x − 2)^2 + 2(y + 2)^2 + 2*sin(2πx)sin(2πy)
def f(w): 
	return (w[0]-2)**2 + 2*(w[1]+2)**2 + 2*m.sin(2*m.pi*w[0])*m.sin(2*m.pi*w[1])			 
# Derivada parcial de E respecto de u
def fx(w):
    return 2*(w[0]-2) + 4*m.pi*m.cos(2*m.pi*w[0])*m.sin(2*m.pi*w[1])

# Derivada parcial de E respecto de v
def fy(w):
	return 4*(w[1]+2) + 4*m.pi*m.cos(2*m.pi*w[1])*m.sin(2*m.pi*w[0])
	
# Gradiente de E
def gradf(w):
	return np.array([fx(w), fy(w)])
	
# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1
def gd_grafica(w, lr, grad_fun, fun, max_iters = 50):
    graf=[]
    it=0
    graf.append(f(w))
    while(it<max_iters):
        w_ant=np.copy(w)
        for j in range(len(w) ):
            w[j] = w[j] - lr*grad_fun(w_ant)[j]
        it+=1
        graf.append(f(w))
	
    plt.plot(range(0,max_iters+1), graf, 'bo')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()	

print ('Resultados apartado 3\n')
print ('\nGrafica con learning rate igual a 0.01')
gd_grafica([1.0,-1.0], 0.01, gradf, f, max_iters = 50)
print ('\nGrafica con learning rate igual a 0.1')
gd_grafica([1.0,-1.0], 0.1, gradf, f, max_iters = 50)

input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

def gd(w, lr, grad_fun, fun, max_iters = 50):	
    it=0
    while(it<max_iters):
        w_ant=np.copy(w)
        for j in range(len(w) ):
            w[j] = w[j] - lr*grad_fun(w_ant)[j]
        it+=1
    return w

print ('\n\nPunto de inicio: (2.1, -2.1)\n')
w=gd([2.1,-2.1], 0.01, gradf, f, max_iters = 50)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('\n\nPunto de inicio: (3.0, -3.0)\n')
w=gd([3.0,-3.0], 0.01, gradf, f, max_iters = 50)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('\n\nPunto de inicio: (1.5, 1.5)\n')
w=gd([1.5,1.5], 0.01, gradf, f, max_iters = 50)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('\n\nPunto de inicio: (1.0, -1.0)\n')
w=gd([1.0,-1.0], 0.01, gradf, f, max_iters = 50)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")
