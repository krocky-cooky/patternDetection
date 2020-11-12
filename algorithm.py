import numpy as np
import math
import sys,os
import matplotlib.pyplot as plt
from functions import *


class KNeighborsClassifier:
    def __init__(self,n_neighbors):
        self.n_neighbors = n_neighbors
        



    def fit(self,x,t):
        self.x = x
        self.t = t

    def predict(self,x):
        ans_list = list()
        for i,data in enumerate(x):
            tmp_t = self.t
            tmp_x = self.x
            neighbors = dict()
            sub = list()
            dists = dist(data,self.x)
            for j in range(self.n_neighbors):
                argmin = np.argmin(dists)
                sub.append(tmp_t[argmin])
                if tmp_t[argmin] in neighbors.keys():
                    neighbors[tmp_t[argmin]] -= 1 
                else:
                    neighbors[tmp_t[argmin]] = -1
                dists = np.concatenate([dists[:argmin],dists[argmin+1:]])
                
                tmp_t = np.concatenate([tmp_t[:argmin],tmp_t[argmin+1:]])
                tmp_x = np.concatenate([tmp_x[:argmin],tmp_x[argmin+1:]])
            
            sorted_neighbors = sorted(neighbors.items(),key=lambda x:x[1])
            
            samp = sorted_neighbors[0][0]
            majority_list = [samp]
            num = sorted_neighbors[0][1]
            for i in range(1,len(sorted_neighbors)):
                if num != sorted_neighbors[i][1]:
                    break
                else : majority_list.append(ret)
            
            for i in range(len(sub)):
                if sub[i] in majority_list:
                    ret = sub[i]
                    break
            
            ans_list.append(ret)
        
        return np.array(ans_list)
    
    
    def accuracy(self,x,t):
        y = self.predict(x)
        acc = np.sum(y == t)/float(y.shape[0])

        return acc


            


                


