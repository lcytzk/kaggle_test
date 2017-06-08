from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np

default = [0, 2, 'male', 30, 3, 0, 32, 'Q']
names = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
fm = {'male': 1, 'female': 100}
emb = {'S': 1, 'C': 100, 'Q': 1000}
c2c = [int, int, lambda x: fm[x], float, int, int, float, lambda x: emb[x]]
#c2c = [int, int, str, float, int, int, float, str]

def loadTest():
    f = open('test.csv', 'r')
    f.readline()
    X = []
    for line in f:
        fields = line.strip().split(',')
        for i in range(1, len(fields)):
            if fields[i] == '':
                fields[i] = default[i]
            X.append(c2c[i](fields[i]))
    X = np.array(X)
    X = X.reshape(-1, len(default) - 1)
    return X

def loadTrain():
    f = open('train.csv', 'r')
    f.readline()
    X = []
    y = []
    for line in f:
        fields = line.strip().split(',')
        y.append(int(fields[0]))
        for i in range(1, len(fields)):
            if fields[i] == '':
                fields[i] = default[i]
            X.append(c2c[i](fields[i]))
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1, len(default) - 1)
    y.ravel()
    return X, y

X, y = loadTrain()
nn = 150
al = X.shape[0]
X_train, y_train = X[range(al - nn)], y[range(al-nn)]
X_test, y_test = X[range(al - nn, al)], y[range(al-nn, al)]
X_n = loadTest()

#clf = tree.DecisionTreeClassifier(max_depth=3)
clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(X, y)

#for depth in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
depth = 4
if True:
    clf.set_params(max_depth=depth)
    clf = clf.fit(X, y)
    #clf = clf.fit(X_train, y_train)

    #with open("iris.dot", 'w') as f:
    #    f = tree.export_graphviz(clf, out_file=f)
    #import os
    #os.unlink('iris.dot')
    #import pydotplus
    #dot_data = tree.export_graphviz(clf, out_file=None)
    #graph = pydotplus.graph_from_dot_data(dot_data)
    #graph.write_pdf("iris-%d.pdf" % depth)

    #from IPython.display import Image
    #dot_data = tree.export_graphviz(clf, out_file=None,
    #    feature_names=names,
    #    class_names=['0', '1'],
    #    filled=True, rounded=True,
    #    special_characters=True)
    #graph = pydotplus.graph_from_dot_data(dot_data)
    #Image(graph.create_png())

    #y_pred = clf.predict(X_test)
    y_pred = clf.predict(X_n)
    #print depth, np.mean(y_pred == y_test)
    for y in y_pred:
        print y
