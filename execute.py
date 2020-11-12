import numpy as np
from algorithm import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

x_train,x_test,t_train,t_test = train_test_split(iris.data,iris.target,random_state = 0)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train,t_train)
if __name__ == '__main__':
    print('accuracy : {}'.format(classifier.accuracy(x_test,t_test)))