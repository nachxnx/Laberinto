# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:38:38 2022

@author: INM
"""

import gym
entorno= gym.make("Taxi-v3")
entorno.reset()
entorno.render()

print('Cantidad de estados: {}'.format(entorno.observation_space.n))
print('Cantidad de acciones: {}'.format(entorno.action_space.n))
import numpy as np
import random
from IPython.display  import clear_output
from matplotlib import pyplot as plt
import gym
import time

import pandas as pd
from tqdm import tqdm

#Q learning

# alpha=0.01
# gamma=0.9
# epsilon=0.01
alpha=0.1
gamma=0.9
epsilon=0.001
q_table=np.zeros([entorno.observation_space.n,entorno.action_space.n])
estado_anterior=None
mostrar_entrenamiento=False
r_mean=[]
steps_por_episodios=[]

#nro_de_episodios=100000
nro_de_episodios=1000

listarecompnesa=[]
listaepocas=[]

for episodio in tqdm(range(0,nro_de_episodios)):
  #Resetear el entorno
  estado = entorno.reset()
  estado_anterior= estado

  #Initialize variables
  recompensa=0
  r_sum=0
  r_sums=[]
  steps=0
  terminado=False
  
  while not terminado:
    #tomar el camino aprendido o explorar nuevas acciones basadas en  eps
    if random.uniform(0,1)<epsilon:
      accion=entorno.action_space.sample()
    else:
      accion=np.argmax(q_table[estado])
    #tomar la accion
    siguiente_estado, recompensa, terminado, info = entorno.step(accion)
    
    #Recalcular
    q_valor=q_table[estado,accion]
    max_valor=np.max(q_table[siguiente_estado])
    nuevo_q_valor=(1-alpha)*q_valor + alpha*(recompensa + gamma*max_valor)

    #Guardar estado anterior
    estado_anterior=estado
    
    r_sum +=recompensa
    
    steps += 1
    #Actualizar Q-table
    q_table[estado,accion]=nuevo_q_valor
    estado= siguiente_estado

  listarecompnesa.append(r_sum)






plt.show()

print(len(listarecompnesa))




# total_epochs=0
# total_penalties=0
# num_of_episodes=10

# for episode in range(num_of_episodes):
#   state=entorno.reset()
#   epochs=0
#   penalties=0
#   reward=0

#   terminated = False

#   while not terminated:
#     action=np.argmax(q_table[state])
#     state, reward, terminated, info=entorno.step(action)

#     time.sleep(0.5)
#     clear_output(wait=True)
#     entorno.render()
      

#     if reward == -10:
#       penalties += 1
      
#     epochs += 1
    
#   total_penalties += penalties
#   total_epochs +=epochs
  
  
# #plt.plot(X,Y)
# #plt.show()
# print(q_table)
# print(np.shape(q_table))

# plt.plot(listarecompnesa)
# plt.show()

# plt.plot(listarecompnesa[99000:-1])
# plt.show()
