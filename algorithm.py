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
        for data in x:
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


class KMeans:
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
        self.x = None
        self.classify = None
        self.represent = None
        self.done = False
        self.color_list = [
            'b',
            'r',
            'g',
            'y',
            'k',
            'c',
            'm',
            'w'
        ]

    def fit(self,x):
        self.x = x
        index = np.array([i for i in range(self.x.shape[0])])
        choice = np.random.choice(index,self.n_clusters,replace = False)
        represent = self.x[choice]
        classify = list()
        pre = None
        while pre != classify:
            next_represent = [list() for i in range(self.n_clusters)]
            pre = classify
            classify = list()
            for data in self.x:
                dists = dist(data,represent)
                argmin = np.argmin(dists)
                classify.append(argmin)
                next_represent[argmin].append(data)
            
            for i,data_list in enumerate(next_represent):
                next_represent[i] = np.sum(next_represent[i],axis = 0)/float(len(next_represent[i]))
            
            represent = np.array(next_represent)
        self.represent = represent
        self.classify = np.array(classify)
        self.done = True 
        
        print('<< successfully classified >>')
        return self.classify

    def visualize(self,argx = 0,argy = 1):
        if not self.done:
            print('please fit a data')
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('argx : ' + str(argx))
        ax.set_ylabel('argy : ' + str(argy))
        for i in range(self.n_clusters):
            mask = self.classify == i
            scatter_x = self.x[mask][:,argx]
            scatter_y = self.x[mask][:,argy]
            ax.scatter(scatter_x,scatter_y,s=50,c=self.color_list[i])

        plt.show()

        





        


        

            


                


