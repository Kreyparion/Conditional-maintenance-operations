from project.env.items import Item
from random import random
import numpy as np
from random import choices

def wearing_prob(i,j,p,q,s):      #probability to go from state i to j
    if i==j:
        if i!=s-1:
            return 1-p-q
        else:
            return 1
    elif j==i+1:
        return p
    elif j==s-1 and i<s-2:
        return q
    else:
        return 0

def markov_matrice(s,p,q):          #Stochastique matrix associated to our Markov chain
    A=np.zeros((s,s))
    for i in range(s):
        for j in range(s):
            A[i][j]=wearing_prob(i,j,p,q,s)
    return A
    

def wearing_func(self,p,q):
    s=self.threshold
    assert self.wear <=s
    A=markov_matrice(s,p,q)
    w=self.wear
    L=A[w]
    list_probas=[]
    for i in range(len(L)):
        list_probas.append(L[i])
    choix=choices(range(len(w)),weights=list_probas)
    self.wear=choix[0]
    return self.wear


