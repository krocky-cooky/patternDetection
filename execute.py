import numpy as np
from algorithm import KNeighborsClassifier,KMeans
from sklearn.datasets import load_iris,load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_iris()
#data = load_digits()

x_train,x_test,t_train,t_test = train_test_split(data.data,data.target,random_state = 0)

way = input()

if __name__ == '__main__' and way == '1':
    x = list()
    y = list()
    for i in range(1,50):
        print('phase : {}'.format(i))
        clf = KNeighborsClassifier(n_neighbors = i)
        clf.fit(x_train,t_train)
        y.append(clf.accuracy(x_test,t_test))
        x.append(i)
        print('done')

    plt.figure()
    plt.plot(x,y)
    plt.show()
elif __name__ == '__main__' and way == '2':
    clf = KMeans(n_clusters = 4)
    clf.fit(data.data)
    clf.visualize()