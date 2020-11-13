import numpy as np
from algorithm import KNeighborsClassifier,KMeans
from sklearn.datasets import load_iris,load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_iris()
#data = load_digits()

x_train,x_test,t_train,t_test = train_test_split(data.data,data.target,random_state = 0)

#way = input()

if __name__ == '__main__' :
    x = list()
    y = list()
    for i in range(1,50):
        print('phase : {}'.format(i))
        accuracy = KNeighborsClassifier.leave_one_out_accuracy(data.data,data.target,n_neighbors = i)
        y.append(accuracy)
        x.append(i)
        print('done')

    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(111)
    ax.plot(x,y,marker = "o")
    ax.set_xlabel('n_neighbors')
    ax.set_ylabel('leave one out accuracy')
    plt.show()
