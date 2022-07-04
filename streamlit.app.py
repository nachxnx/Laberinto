# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 16:48:37 2022

@author: INM
"""
import pandas as pd
import streamlit as st
import numpy as np



st.title("***Resolucion de un Laberinto por medio de Reinforcement Learning***")


txt = st.text_area('Introduccion', '''
     Para la aplicacion del Reinforcement Learning optamos por el 'taxi-V3' que se encuentra en 
     el repositorio de  OpenAI-gym.
     Hay varias formas de implementar estos procesos de aprendizaje.
     En nuestro caso optamos por Q-learning ya que tenemos un entorno discreto y por ende los 
     resultados seran los mas optimos, ademas el Q-learning permite resolver problemas de decision
     secuencial en los cuales la utilidad de una accion depende de una secuencia de decisiones y 
     donde ademas existe incertidumbre en cuanto a las dinamicas del ambiente,laberinto, en que 
     esta situado el agente, robot.
     Se define como el optimo Q* como el retorno que se puede obtener que obedece la ecuacion
     de Bellman.
     ''',height=330)
     
st.header("Mapa obtenido de OpenAI-gym del 'taxi-V3'")
code = '''MAP = [ "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",]'''
st.code(code, language='python')

cool1, cool2,cool3 = st.columns(3)
with cool1:
    """
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi"""
with cool2:
    """
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    """
with cool3:
    """
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger    
    """

with st.container():
    st.header("class TaxiEnv")
    code = '''
    def __init__(self):
    self.desc = np.asarray(MAP, dtype="c")
    self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
    num_states = 500
    num_rows = 5
    num_columns = 5
    max_row = num_rows - 1
    max_col = num_columns - 1
    initial_state_distrib = np.zeros(num_states)
    num_actions = 6
    P = {
        state: {action: [] for action in range(num_actions)}
        for state in range(num_states)
    }
    for row in range(num_rows):
        for col in range(num_columns):
            for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                for dest_idx in range(len(locs)):
                    state = self.encode(row, col, pass_idx, dest_idx)
                    if pass_idx < 4 and pass_idx != dest_idx:
                        initial_state_distrib[state] += 1
                    for action in range(num_actions):
                        # defaults
                        new_row, new_col, new_pass_idx = row, col, pass_idx
                        reward = (-1) # default reward when there is no pickup/dropoff
                        done = False
                        taxi_loc = (row, col)
                        if action == 0:
                            new_row = min(row + 1, max_row)
                        elif action == 1:
                            new_row = max(row - 1, 0)
                        if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                            new_col = min(col + 1, max_col)
                        elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                            new_col = max(col - 1, 0)
                        elif action == 4:  # pickup
                            if pass_idx < 4 and taxi_loc == locs[pass_idx]:
                                new_pass_idx = 4
                            else:  # passenger not at location
                                reward = -10
                        elif action == 5:  # dropoff
                            if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                new_pass_idx = dest_idx
                                done = True
                                reward = 20
                            elif (taxi_loc in locs) and pass_idx == 4:
                                new_pass_idx = locs.index(taxi_loc)
                            else:  # dropoff at wrong location
                                reward = -10
                        new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)
                        P[state][action].append((1.0, new_state, reward, done))
    initial_state_distrib /= initial_state_distrib.sum()
    discrete.DiscreteEnv.__init__(
        self, num_states, num_actions, P, initial_state_distrib
    )'''
    st.code(code, language='python')

col1, col2 = st.columns(2)

with col1:
    st.header("encode")
    code = '''
    
    def encode(self, taxi_row,
    taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i
    '''
    st.code(code, language='python')
with col2:
    st.header("decode")
    code = '''
    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)
    '''
    st.code(code, language='python')

    
st.title("***Nuestra implementacion***") 


code = '''MAP = [
        "+---------+",
        "| | : : : |",
        "|*|*|||||*|",
        "| | : |R| |",
        "|*|||*|*|*|",
        "| : : | | |",
        "|*|*|||*|*|",
        "| | : : | |",
        "|*|*|||*|*|",
        "| | :R| |G|",
        "+---------+",]'''
st.code(code, language='python')

cool1, cool2,cool3 = st.columns(3)
with cool1:
   """ ### Acciones
    - 0: mover abajo
    - 1: mover arriba
    - 2: mover derecha
    - 3: mover izquierda"""
    
with cool2:
    """
    ### Observaciones
    Hay 25 estados discretos 
    """
with cool3:
    """
    ### Recompensas
    - -1 por paso a menos que se active otra recompensa.
    - +20 llegar a la meta.
    - -10 caer en la trampa. 
    """                   
                   
with st.container():
    st.header("class TaxiEnv")
    code = ''' def __init__(self):
    self.desc = np.asarray(MAP, dtype="c")
    self.locs= locs=[(1,3),(4,2),(4,4)]
    num_states = 25
    num_rows = 5
    num_columns = 5
    max_row = num_rows - 1
    max_col = num_columns - 1
    self.initial_state_distrib = np.zeros(num_states)
    num_actions = 4 #arriba, abajo, izquierda, derecha
    self.P = {state: {action: [] for action in range(num_actions)}for state in range(num_states)}
    for row in range(num_rows):
        for col in range(num_columns):
            state = self.encode(row, col)
            self.initial_state_distrib[state] += 1
            for action in range(num_actions):
                robot_loc = (row, col)
                new_row, new_col = row, col
                done = False
                reward = (-1)
                if action == 0 and self.desc[2*row+2 , 2*col+1]== b"*": #desc es el mapa"
                    new_row = min(row+1, max_row)
                elif action == 1 and self.desc[2*row , 2*col+1]== b"*":
                    new_row = max(row-1, 0)
                    
                if action == 2 and self.desc[1 + row*2, 2 * col + 2] == b":":
                    new_col = min(col + 1, max_col)
                elif action == 3 and self.desc[1 + row*2, 2 * col] == b":":
                    new_col = max(col - 1, 0)
                    
                if robot_loc == locs[2]:
                   reward = 20
                   done = True
                elif robot_loc == locs[0] or robot_loc == locs[1]:
                   reward = -10
                   done = True
                new_state = self.encode(new_row, new_col)
                self.P[state][action].append((1.0, new_state, reward, done))
    self.initial_state_distrib /= self.initial_state_distrib.sum()
    self.action_space = spaces.Discrete(num_actions)
    self.observation_space = spaces.Discrete(num_states) 
    '''      

col1, col2 = st.columns(2)

with col1:
    st.header("encode")
    code = '''
    
    def encode(self, robot_row, robot_col):
        i = robot_row
        i *= 5
        i += robot_col
        return i
    '''
    st.code(code, language='python')
with col2:
    st.header("decode")
    code = '''
    def decode(self, i):
        out = []
        out.append(i // 5)
        out.append(i % 5)
        assert 0 <= i < 25
        return (out)
    '''
    st.code(code, language='python')
    
    
with st.container():
    code='''
        alpha=0.5
gamma=0.9
epsilon=0.001
q_table=np.zeros([entorno.observation_space.n,entorno.action_space.n])
estado_anterior=None
mostrar_entrenamiento=False
r_mean=[]
steps_por_episodios=[]
nro_de_episodios=100
listarecompnesa=[]
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
    '''
    st.code(code, language='python')

st.title("      ")
import numpy as np
import random
from IPython.display  import clear_output
from matplotlib import pyplot as plt
import gym
import time

import pandas as pd
from tqdm import tqdm    
import gym

             
alpha = st.selectbox(
     'Seleccione el valor de alpha',
     (0.01, 0.05, 0.1,0.5,0.9))
st.write('alpha: ', alpha)
gamma = st.selectbox(
     'Seleccione el valor de gamma',
     (0.01, 0.1, 0.5,0.9))
st.write('gamma: ', gamma)
epsilon = st.selectbox(
     'Seleccione el valor de epsilon',
     (0.001, 0.01, 0.05 ,0.1 ,0.5,0.9))
st.write('epsilon: ', epsilon)
nro_de_episodios = st.selectbox(
     'Seleccione el valor de epocas',
     (10, 1000, 10000,100000))
st.write('Epocas:', nro_de_episodios)

def main():
    st.header("Pulse para Entrenar")
    if st.button('Entrenar'):
        entorno= gym.make("Taxi-v3")
        entorno.reset()
        q_table=np.zeros([entorno.observation_space.n,entorno.action_space.n])
        listarecompnesa=[]
        acciones=[]
        for episodio in tqdm(range(0,nro_de_episodios)): 
          #Resetear el entorno
          estado = entorno.reset()
          estado_anterior= estado

          #Initialize variables
          recompensa=0
          r_sum=0
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
          acciones.append(accion)
        
        st.header("Recompensa obtenida")
        st.dataframe(listarecompnesa)
        st.header("Q_table obtenida")
        st.dataframe(q_table)
        st.header("Variacion de Recompensa:")
        col1, col2 ,col3, col4 = st.columns(4)
        with col1:
            st.write('alpha: ', alpha)
        with col2:
            st.write('gamma: ', gamma)
        with col3:
            st.write('epsilon: ', epsilon)
        with col4:
            st.write('Epocas:', nro_de_episodios)
        st.line_chart(listarecompnesa)
    else:
         st.write(' ')

if __name__== '__main__':
    main()
  
# st.header("Recompensa")
# Lectura=pd.read_csv("recompensa.csv",index_col=(0))
# st.dataframe(Lectura)
# st.header("Variacion de Recompensa para 1000 epocas")
# char_data=pd.DataFrame(np.random.randn(20,3),columns=["length","width","size"])

# st.line_chart(Lectura)


