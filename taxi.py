# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:58:09 2022

@author: INM
"""

import sys
from contextlib import closing
from io import StringIO
from typing import Optional

import numpy as np
from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

MAP = [
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
        "+---------+",
]


class TaxiEnv(Env):
    """

    El Laberinto
    
    ### Descripcion
    Hay una meta designada en el mundo de la cuadricula G(reen).  Cuando comienza el episodio, 
    el robot se pone en marcha, en un cuadro al azar. El robot se mueve por el trayecto más rápido hasta la meta, 
    pero además hay trampas por el camino. Una vez que llega, el episodio termina
    Map:
        
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
        "+---------+",
        
        # / es una pared, : puede continuar horizontalmente, *puede continuar verticalmente 
    ### Acciones
    - 0: mover abajo
    - 1: mover arriba
    - 2: mover derecha
    - 3: mover izquierda
    
    
    
    ### Observaciones
    Hay 25 estados discretos 
 

     ### Recompensas
     - -1 por paso a menos que se active otra recompensa.
     - +20 llegar a la meta.
     - -10 caer en la trampa.
    


    ### Rendering
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    state space is represented by:
    (robot_row, robot_col, passenger_location, destination)

    ### Arguments

    ```
    gym.make('Taxi-v3')
    ```

    ### Version History
    * v3: Map Correction + Cleaner Domain Description
    * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
    * v1: Remove (3,2) from locs, add passidx<4 check
    * v0: Initial versions release
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype="c")

        #elf.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs= locs=[(1,3),(4,2),(4,4)]
        self.locs_colors = [(255, 0, 0),  (255, 255, 0),(0, 255, 0), (0, 0, 255)]

        num_states = 25
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        num_actions = 4 #arriba, abajo, izquierda, derecha
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(num_rows):
            for col in range(num_columns):
                state = self.encode(row, col)
                a=(row,col)
                if a!=locs[2]:
                    self.initial_state_distrib[state] += 1
                for action in range(num_actions):
                    new_row, new_col = row, col
                    robot_loc = (row, col)
                    # if robot_loc!=locs[2]:
                    #     self.initial_state_distrib[state] += 1
                    done = False
                    reward = (-1)
                    if action == 0 and self.desc[2*row+2 , 2*col+1]== b"*": #desc es el mapa"
                        new_row = min(row+1, max_row)
                    elif action == 1 and self.desc[2*row , 2*col+1]== b"*":
                        new_row = max(row-1, 0)
                    if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                        new_col = min(col + 1, max_col)
                    elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                        new_col = max(col - 1, 0)
                        #print(self.desc[1 + row, 2 * col])
                    if robot_loc == locs[2]:
                       reward = 20
                       done = True
                    elif robot_loc == locs[0] or robot_loc == locs[1]:
                       reward = -10
                       done = True
                    new_state = self.encode(
                        new_row, new_col
                    )
                    self.P[state][action].append((1.0, new_state, reward, done))
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)   
                
                

    
#   def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
    def encode(self, robot_row, robot_col):
        
        # (5) 5, 5, 4
        i = robot_row
        i *= 5
        i += robot_col
        #i += pass_loc
        #i *= 4
        #i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i // 5)
        out.append(i % 5)
        assert 0 <= i < 25
        return (out)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

    def reset(self,*,seed: Optional[int] = None,return_info: bool = False,
        options: Optional[dict] = None,):
        
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        #taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)
        robot_row, robot_col = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if (robot_row,robot_col)!= self.locs[2]:
            out[2 * robot_row +1][2 * robot_col + 1] = utils.colorize(
            out[2 * robot_row +1][2 * robot_col + 1], "yellow", highlight=True)
        
        #pi, pj = self.locs[pass_idx]
        #    out[1 + pi][2 * pj + 1] = utils.colorize(
        #        out[1 + pi][2 * pj + 1], "blue", bold=True)
        else:  
            # out[1 + robot_row][2 * robot_col + 1] = utils.colorize(
            #     ul(out[1 + robot_row][2 * robot_col + 1]), "green", highlight=True)
            out[2 * robot_row +1][2 * robot_col + 1] = utils.colorize(
                ul(out[2 * robot_row +1][2 * robot_col + 1]), "green", highlight=True)
        #di, dj = self.locs[dest_idx]
        #out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['Abajo', 'Arriba', 'Derecha', 'Izquiera'][self.lastaction]})\n\n")
        else:
            outfile.write("\n\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()