#%%
import numpy as np
import itertools
from itertools import product,combinations_with_replacement
from project.env.environnement import Environnement
from project.env.actions import Action


#matrice de viellissement 

A = np.array([
    [0.9848, 0.015, 0, 0, 0.0002],
    [0, 0.989, 0.01, 0, 0.001],
    [0, 0, 0.98, 0.005, 0.015],
    [0, 0, 0, 0.94, 0.06],
    [0, 0, 0, 0, 1]
    ])

#nombre d'éoliennes
p = 3

def get_all_states(A,p):
    liste = list(combinations_with_replacement([i for i in range (len(A))],p))
    states = []
    for x in liste : 
        states.append(list(reversed(x)))
    return states

#toutes les permutations permet d'avoir tous les états dans les lequels on arrive deuis un état intiale ma_liste

def toutes_les_permutations(ma_liste):
    permutations = list(itertools.permutations(ma_liste,len(ma_liste)))
    res =[]
    for x in list(set(permutations)):
        res.append(list(x))
    return res

print(get_all_states(A,p))   
# print(len(get_all_states(A,p)))

def get_matrice_stochastique(A,p):
    states =  get_all_states(A,p)
    m = len(states)
    Q = np.zeros((m,m))
    r = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            res = 0
            class_j = toutes_les_permutations(states[j])

            for state_j in class_j:
                    proba = 1
                    for k in range (p):
                        proba = proba*A[states[i][k]][state_j[k]]
                    res+=proba
            r[i][j] = p-states[j].count(4)
            Q[i][j] = res             
    return Q,r 


def est_stochastique(B):
    for i in range(len(B)):
    #     if sum(B[i])!= 1:
    #         return False 
    # return True
        print(sum(B[i]))

#Vérifie que tous les termes de la matrice sont dans [0,1]
def terme_proba(B):
    for i in range(len(B)):
        for j in range(len(B)):
            if B[i][j] > 1:
                print(i,j)
                return False
    return True
    

def from_wears_en_place(wears,env:Environnement):
    for i,item in enumerate(env.state.items):
        item.wear = wears[i]

def get_wears(env):
    wears = []
    for i,item in enumerate(env.items):
        wears.append(item.wear)
    return wears

env = Environnement.init("3dadvanced")
env.reset()
Actions = env.getPossibleActions()

print(env.getPossibleActions())

#env.step(A)

def matrices(Actions,A):
    T = []
    R = []
    States = get_all_states(A,p)
    for a in Actions:
        if a == Action.ActionDoNothing():
            T.append(get_matrice_stochastique(A,p)[0])
            R.append(get_matrice_stochastique(A,p)[1])
        else : 
            Q = [[0 for i in range(len(States))] for i in range(len(States))]
            r = [[0 for i in range(len(States))] for i in range(len(States))]
            for i,state in enumerate(States):
                from_wears_en_place(state,env)
                _,reward,_ = env.step(a)
                new_state = get_wears(env)
                j = States.index(new_state)
                r[i][j] = reward
                Q[i][j] = 1
            T.append(Q)
            R.append(r)
    return np.array(T),np.array(R)

#print(matrices(Actions,A)[1])


    
P,R =matrices(Actions,A)
#%%

def matrice_delay(Actions,A,n):
    p,r = matrices(Actions,A)
    po = p[0]
    ro = r[0]
    P = np.zeros(p.shape)
    R = np.zeros(r.shape)
    P[0] = np.linalg.matrix_power(po,n)
    

    
    for i in range(0,p.shape[0]):
        for j in range(n):
            R[i] += np.matmul(np.linalg.matrix_power(po,j),ro)
        R[i] += np.matmul(np.linalg.matrix_power(po,n),r[i])
        P[i] = np.matmul(np.linalg.matrix_power(po,n),p[i])
        
        

    return P,R

P,R =  matrice_delay(Actions,A,14)

print(P)
print("test pass")

# %%

import mdptoolbox

fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.9999,10000) 
fh.run()
print("value")

print(fh.V)
print(fh.policy.shape)
best_policy = fh.policy[:,0]

print(best_policy)

# %%

from copy import deepcopy
state = env.reset()
States = get_all_states(A,p)
done = False
print(States)
while not done:
    # Agent takes action
    new_state = get_wears(env)
    int_state = States.index(new_state)
    action_int = best_policy[int_state]
    action = Actions[action_int]

    # Action has effect on environment
    next_state, reward, done = env.step(action)

    # Render environment for user to see
    env.render()

            # Update state
    state = deepcopy(next_state)
# %%


