from cvxpy import *
import numpy as np
import math
from util import Timer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

class SVM(object):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, 
                 decision_function_shape=None, random_state=None):
        self.C = C
        self.gamma = gamma
        assert kernel in ['linear', 'rbf'], 'Error: specified kernel type not found!'
        self.kernel = kernel

    def get_params(self, deep=False):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"C": self.C, "gamma": self.gamma, "kernel": self.kernel}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        assert len(X) == len(y), 'Error: inconsistent length of features and label!'
        labels = set(y)
        assert len(labels) == 2, 'Error: more than two class in the labels!'
        self._pos_label, self._neg_label = labels
        if self.gamma == 'auto':
            self.gamma = 1/float(len(X[0]))
        y = np.vstack([int(yi==self._pos_label)*2-1 for yi in y]) # convert labels to (+1,-1)
        Q = np.zeros((len(X), len(X)))
        if self.kernel == 'linear':
            self._solve_primal_weight(X, y)
            # TODO linear SVM can be solved in dual form, too.
        elif self.kernel == 'rbf':
            self._X = X # TODO until I find the way to select support vectors
            Q = rbf_kernel(X, gamma=self.gamma)
            Q = np.multiply(Q, y)            
            Q = np.multiply(Q, y.T)
            self._solve_primal_alpha(Q, y)

    def predict(self, X):
        y = self.decision_function(X)
        y = np.sign(y).astype(int)
        mapping = {1: self._pos_label, -1: self._neg_label}
        return [mapping[yi] for yi in y]

    def decision_function(self, X):
        if self.kernel == 'linear':
            f = np.dot(X, self._w) + self._b
        elif self.kernel == 'rbf':
            ## rbf_kernel returns array of shape (n_samples_X, n_samples_Y)
            assert self._ya.shape == (len(self._X), 1)
            f = np.sum(np.multiply(rbf_kernel(self._X, X), self._ya), axis=0) + self._b
        f = np.squeeze(np.array(f))
        return f

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    # solve primal svm in domain of weight
    def _solve_primal_weight(self, X, y):
        w = Variable(len(X[0]))
        b = Variable()
        xi = Variable(len(y))
        C = self.C
        objective = Minimize((0.5/C)*(sum_squares(w) + square(b)) + 
                             sum_squares(xi))
        constrain = [diag(y)*X*w + b*y + xi >= 1, xi >= 0]
        prob = Problem(objective, constrain)
        with Timer('solving'):
            prob.solve()
        #print "status:", prob.status
        #print "optimal value", prob.value
        #print "optimal var", w.value.ravel(), b.value
        self._w = w.value
        self._b = b.value
        optimal_xi = xi.value
        self._SV = []
        for i in xrange(len(optimal_xi)):
            if optimal_xi[i] > 10**-5: # TODO tolerance
                self._SV.append(X[i])

    # solve primal svm in domain of alpha
    def _solve_primal_alpha(self, Q, y):
        l = len(y)
        alpha = Variable(l)
        b = Variable()
        xi = Variable(l)
        C = self.C
        objective = Minimize((0.5/C)*((quad_form(alpha, Q)) + square(b)) + 
                             sum_squares(xi))
        constrain = [Q*alpha + b*y + xi >= 1, alpha >= 0, xi >= 0]
        prob = Problem(objective, constrain)
        with Timer('solving'):
            prob.solve()
        #print "status:", prob.status
        #print "optimal value", prob.value
        #"optimal var", alpha.value.ravel(), b.value
        self._alpha = alpha.value
        self._b = b.value
        self._ya = np.multiply(self._alpha, np.vstack(y))

class RSVM(SVM):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, 
                 decision_function_shape=None, random_state=None):
        super(RSVM, self).__init__(C, 'rbf', degree, gamma, coef0,
                 shrinking, probability, tol, cache_size,
                 class_weight, verbose, max_iter, 
                 decision_function_shape, random_state)
        if kernel == 'linear': print 'Warning: only rbf kernel available now!' # TODO because CVXPY can't solve with indefinite Q

    def _solve_primal_alpha(self, Q, y):
        l = len(Q)
        m = int(l/4) # TODO
        self._R_index = np.random.choice(l, m, replace=False)
        Q_R = Q[:, self._R_index]
        Q_RR = Q_R[self._R_index, :]

        alpha = Variable(m)
        b = Variable()
        xi = Variable(l)
        C = self.C
        objective = Minimize((0.5/C)*((quad_form(alpha, Q_RR)) + square(b)) + 
                             sum_squares(xi))
        constrain = [Q_R*alpha + b*y + xi >= 1, alpha >= 0, xi >= 0]
        prob = Problem(objective, constrain)
        with Timer('solving'):
            prob.solve()
        #print "status:", prob.status
        #print "optimal value", prob.value
        #"optimal var", alpha.value.ravel(), b.value
        self._alpha = alpha.value
        self._b = b.value
        y_R = np.array(y)[self._R_index]
        self._ya = np.multiply(self._alpha, np.vstack(y_R))
        self._X = self._X[self._R_index]
        assert len(self._ya) == len(self._X)

class CSSVM(SVM):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, 
                 decision_function_shape=None, random_state=None):
        super(CSSVM, self).__init__(C, 'rbf', degree, gamma, coef0,
                 shrinking, probability, tol, cache_size,
                 class_weight, verbose, max_iter, 
                 decision_function_shape, random_state)
        if kernel == 'linear': print 'Warning: only rbf kernel available now!' # TODO because of indefinite Q

    def _solve_primal_alpha(self, Q, y):
        l = len(Q)
        m = int(l/4) # TODO
        self._phi = np.random.uniform(-1, 1, (l, m))
        Q_R = Q.dot(self._phi)
        Q_RR = self._phi.T.dot(Q_R)

        alpha = Variable(m)
        b = Variable()
        xi = Variable(l)
        C = self.C
        objective = Minimize((0.5/C)*(quad_form(alpha, Q_RR) + square(b)) + sum_squares(xi))
        #objective = Minimize((0.5/C)*(sum_squares(alpha) + square(b)) + sum_squares(xi))
        constrain = [Q_R*alpha + b*y + xi >= 1, alpha >= 0, xi >= 0]
        prob = Problem(objective, constrain)
        with Timer('solving'):
            prob.solve()
        #print "status:", prob.status
        #print "optimal value", prob.value
        #"optimal var", alpha.value.ravel(), b.value
        self._alpha = alpha.value
        self._b = b.value
        self._y = y

    def decision_function(self, X):
        assert self.kernel == 'rbf'
        y = np.zeros(len(X))
        y_dot_K = np.multiply(rbf_kernel(self._X, X), self._y)
        w_dot_x = np.dot(self._alpha.T, np.dot(self._phi.T, y_dot_K))
        f = w_dot_x + self._b
        f = np.squeeze(np.array(f))
        return f

def main():
    pass

if __name__ == '__main__':
    from sklearn.datasets import make_moons, make_classification
    from sklearn.cross_validation import train_test_split
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)
    #X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1) # linearly separable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    print('dual SVM, linear kernel')
    clf = SVM(C=1, kernel='linear', gamma='auto')
    with Timer('training'):
        clf.fit(X_train, y_train)
    with Timer('testing'):
        print 'accuracy:', clf.score(X_test, y_test)
    print

    print('dual SVM, rbf kernel')
    clf = SVM(C=1, kernel='rbf', gamma='auto')
    with Timer('training'):
        clf.fit(X_train, y_train)
    with Timer('testing'):
        print 'accuracy:', clf.score(X_test, y_test)
    print

    print('RSVM, rbf kernel')
    clf = RSVM(C=1, kernel='rbf', gamma='auto')
    with Timer('training'):
        clf.fit(X_train, y_train)
    with Timer('testing'):
        print 'accuracy:', clf.score(X_test, y_test)
    print

    print('CSSVM, rbf kernel')
    clf = CSSVM(C=1, kernel='rbf', gamma='auto')
    with Timer('training'):
        clf.fit(X_train, y_train)
    with Timer('testing'):
        print 'accuracy:', clf.score(X_test, y_test)
    print

    from sklearn.svm import SVC
    clf = SVC(gamma='auto', C=1)
    print('LibSVM, rbf kernel')
    with Timer('training'):
        clf.fit(X_train, y_train)
    with Timer('testing'):
        print 'accuracy:', clf.score(X_test, y_test)
    print
