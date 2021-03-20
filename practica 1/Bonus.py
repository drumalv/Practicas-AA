# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import math as m
import numpy as np
import matplotlib.pyplot as plt

# Funcion f(x,y)== (x − 2)^2 + 2(y + 2)^2 + 2*sin(2πx)sin(2πy)
def f(w): 
	return (w[0]-2)**2 + 2*(w[1]+2)**2 + 2*m.sin(2*m.pi*w[0])*m.sin(2*m.pi*w[1])			 
# Derivada parcial de E respecto de x
def fx(w):
    return 2*(w[0]-2) + 4*m.pi*m.cos(2*m.pi*w[0])*m.sin(2*m.pi*w[1])

# Derivada parcial doble de E respecto de x
def f2x(w):
    return 2 - 8*((m.pi)**2)*m.sin(2*m.pi*w[0])*m.sin(2*m.pi*w[1])

# Derivada parcial de E respecto de y
def fy(w):
	return 4*(w[1]+2) + 4*m.pi*m.cos(2*m.pi*w[1])*m.sin(2*m.pi*w[0])

# Derivada parcial doble de E respecto de y
def f2y(w):
	return 4 - 8*((m.pi)**2)*m.sin(2*m.pi*w[0])*m.sin(2*m.pi*w[1])

# Derivada parcial de E respecto de x, luego respecto de y
def fxy(w):
    return 8*((m.pi)**2)*m.cos(2*m.pi*w[0])*m.cos(2*m.pi*w[1])

# Derivada parcial de E respecto de y, luego respecto de x
def fyx(w):
    return 8*((m.pi)**2)*m.cos(2*m.pi*w[0])*m.cos(2*m.pi*w[1])

# Gradiente de E
def gradf(w):
	return np.array([fx(w), fy(w)])
#matriz hessiana
def H(w):
    matriz = [[None] * 2] * 2
    matriz[0][0]=f2x(w)
    matriz[0][1]=fxy(w)
    matriz[1][0]=fyx(w)
    matriz[1][1]=f2y(w)
    return matriz
#algoritmo newton
def newton(w, grad_fun, fun, H , max_iters = 50):		
    graf=[]
    it=0
    graf.append(f(w))
    while(it<max_iters):
        deltaW=(-1)*np.linalg.pinv(H(w)).dot(gradf(w))
        w=w+deltaW
        it+=1
        graf.append(f(w))
	
    plt.plot(range(0,max_iters+1), graf, 'bo')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()
    
    
newton([1.0,-1.0],gradf, f, H)





















