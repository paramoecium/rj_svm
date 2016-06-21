#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from math import log
import matplotlib
matplotlib.use('Agg') # for linux server without display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
#from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from rsvm import SVM, RSVM, CSSVM

n_samples = 500
X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(n_samples=n_samples, noise=0.3, random_state=0),
            make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

tuned_parameters = [{'kernel': ['rbf'],
                     'C': [0.01, 0.1, 1, 10, 100]
                    }]   

figure = plt.figure()
gs = gridspec.GridSpec(len(datasets), 10)
# iterate over datasets
for i, ds in enumerate(datasets):
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    m = np.linspace(2, len(X_train)*4/5, num=10).astype(int) # mind the 5-fold crossvalidation
    names = ["RSVM(rbf)", "CSSVM(rbf)"]
    colors = ['r','g','b']
    acc = []
    for m_i in m:
        
        classifiers = [GridSearchCV(RSVM(m=m_i), tuned_parameters, cv=5, scoring=None, n_jobs=20),
                       GridSearchCV(CSSVM(m=m_i), tuned_parameters, cv=5, scoring=None, n_jobs=20)]
        acc_m = []
        for name, clf in zip(names, classifiers):   
            print name, m_i
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            for params, mean_score, scores in clf.grid_scores_:
                #print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))
                #print()
                acc_m.append(mean_score)
        acc.append(acc_m)
    acc = np.array(acc)

    ax = plt.subplot(gs[i, :3])
    dot_size = 5000.0/len(X)
    cm_bright = ListedColormap(['#0000FF', '#FF0000'])
    ax.scatter(X_test[:, 0], X_test[:, 1], s=dot_size, c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xticks(())
    ax.set_yticks(())

    ax = plt.subplot(gs[i, 4:])
    for j, name in enumerate(names):
        ax.plot(m, acc[:,j], linewidth=2, label=name, color=colors[j])
    ax.set_xlabel('m')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0, 1)

    plt.legend(loc=4, prop={'size': 12})
    #ax.suptitle('ds')
    i += 1

#plt.show()
plt.savefig('comparison_m.png',dpi=500)
