import itertools
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
np.set_printoptions(precision=15)
def state_index(bool_array):
    p=1
    res=0
    for k in bool_array:
        if k: res+=p
        p=p*2
    return res
def expected_value_matrix(dice_max):
    probs=[0]*(2*dice_max)
    M=np.zeros((2**(2*dice_max),2**(2*dice_max)),dtype=np.float64)
    n=0
    for i,j in itertools.product(range(1,dice_max+1),repeat=2):
        ks=set([i-1,j-1,i+j-1])
        for k in ks:
            probs[k]+=1/(dice_max**2)
    for state in itertools.product((False,True),repeat=2*dice_max):   #implementation of optimal strategy, turn face down lowest prob, face up highest prob
        for i, j in itertools.product(range(1,1+dice_max), repeat=2):
            poss=[i-1,j-1,i+j-1]
            on=[x for x in poss if not state[x]]
            if on:                                  #we  turn a card face down that has lwoest prob
                index=min(on,key=probs.__getitem__)
            else:                                   #we turn a card with highest prob face up
                index=max(poss,key=probs.__getitem__)
            final_state = list(state)
            final_state[index] = not final_state[index]
            M[ state_index(state),state_index(final_state)] += 1/(dice_max**2)
        n+=1
        print(n)
    return sp.sparse.csc_matrix(M)

n=4
M=expected_value_matrix(n)[:-1,:-1]
E=sp.sparse.csc_matrix(np.identity(2**(2*n)-1))-M
x=sp.sparse.linalg.inv(E)
print(x.todense().sum(axis=1)[0])
