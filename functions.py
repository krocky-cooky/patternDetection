import numpy as np
import sys,os

def dist(a,b):
    ret = np.sum((a-b)**2,axis = 1)

    return ret