# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
"""Implementation of Stochastic Gradient Descent (SGD) with sparse data."""

import numpy as np
from scipy import sparse

from ...base import ClassifierMixin
from ..base import LinearModel
from .sgd_fast_sparse import plain_sgd, Hinge, Log, ModifiedHuber


class SGD(LinearModel, ClassifierMixin):
    """Linear Model trained by minimizing a regularized training
    error using SGD.

    This implementation works on scipy.sparse X and dense coef_.

    Parameters
    ----------
    loss : str, ('hinge'|'log'|'modifiedhuber')
        The loss function to be used. Defaults to 'hinge'. 
    penalty : str, ('l2'|'l1'|'elasticnet')
        The penalty (aka regularization term) to be used. Defaults to 'l2'.
    alpha : float
        Constant that multiplies the regularization term. Defaults to 0.0001
    rho : float
        The Elastic Net mixing parameter, with 0 < rho <= 1.
        Defaults to 0.85.
    coef_ : ndarray of shape n_features
        The initial coeffients to warm-start the optimization.
    intercept_ : float
        The initial intercept to warm-start the optimization.
    fit_intercept: bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.
    n_iter: int
        The number of passes over the training data (aka epochs).
        Defaults to 5. 
    shuffle: bool
        Whether or not the training data should be shuffled after each epoch.
        Defaults to False.
    verbose: int
        Verbose output. If > 0, learning statistics are printed to stdout.
        Defaults to 0.

    Attributes
    ----------
    FIXME adhere to SVM signature [n_classes, n_features].
    `coef_` : array, shape = [n_features]
        Weights asigned to the features.

    `intercept_` : float
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> Y = np.array([1, 1, 2, 2])
    >>> from scikits.learn.sgd.sparse import SGD
    >>> clf = SGD()
    >>> clf.fit(X, Y)
    SGD(loss='hinge', shuffle=False, fit_intercept=True, n_iter=5, penalty='l2',
      coef_=array([ 9.80373,  9.80373]), rho=1.0, alpha=0.0001,
      intercept_=-0.1)
    >>> print clf.predict([[-0.8, -1]])
    [ 1.]

    See also
    --------
    LinearSVC

    """

    def _get_loss_function(self):
        loss_functions = {"hinge" : Hinge(),
                          "log" : Log(),
                          "modifiedhuber" : ModifiedHuber(),
                          }
        try:
            self.loss_function = loss_functions[self.loss]
        except KeyError:
            raise ValueError("The loss %s is not supported. " % self.loss)

    def _set_coef(self, coef_):
        self.coef_ = coef_
        if coef_ is None:
            self.sparse_coef_ = None
        else:
            # sparse representation of the fitted coef for the predict method
            self.sparse_coef_ = sparse.csr_matrix(coef_)

    def fit(self, X, Y, **params):
        """Fit current model with SGD

        X is expected to be a sparse matrix. For maximum efficiency, use a
        sparse matrix in CSR format (scipy.sparse.csr_matrix)
        """
        self._set_params(**params)
        X = sparse.csr_matrix(X)
        Y = np.asanyarray(Y, dtype=np.float64)
        # largest class id is positive class
        classes = np.unique(Y)
        self.classes = classes
        if classes.shape[0] > 2:
            self._fit_multiclass(X, Y)
        elif classes.shape[0] == 2:
            self._fit_binary(X, Y)
        else:
            raise ValueError("The number of class labels must be " \
                             "greater than one. ")
        # return self for chaining fit and predict calls
        return self

    def _fit_binary(self, X, Y):
        """Fit a binary classifier.
        """
        # encode original class labels as 1 (classes[1]) or -1 (classes[0]).
        Y_new = np.ones(Y.shape, dtype=np.float64) * -1.0
        Y_new[Y == self.classes[1]] = 1.0
        Y = Y_new

        n_samples, n_features = X.shape[0], X.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros(n_features, dtype=np.float64, order="C")
        if self.intercept_ is None:
            self.intercept_ = 0.0

        X_data = np.array(X.data, dtype=np.float64, order="C")
        X_indices = X.indices
        X_indptr = X.indptr
        coef_, intercept_ = plain_sgd(self.coef_,
                                      self.intercept_,
                                      self.loss_function,
                                      self.penalty_type,
                                      self.alpha, self.rho,
                                      X_data,
                                      X_indices, X_indptr, Y,
                                      self.n_iter,
                                      int(self.fit_intercept),
                                      int(self.verbose),
                                      int(self.shuffle))

        # update self.coef_ and self.sparse_coef_ consistently
        self._set_coef(coef_)
        self.intercept_ = intercept_

    def _fit_multiclass(self, X, Y):
        """Fit a multi-class classifier with a combination
        of binary classifiers, each predicts one class versus
        all others (OVA).
        """
        print("Trace: fit multi")
        n_classes = self.classes.shape[0]
        n_samples, n_features = X.shape[0], X.shape[1]
        if self.coef_ is None:
            coef_ = np.zeros((n_classes, n_features),
                             dtype=np.float64, order="C")
        else:
            if self.coef_.shape != (n_classes, n_features):
                raise ValueError("Provided coef_ does not match dataset. ")
            coef_ = self.coef_
        
        if self.intercept_ is None \
               or isinstance(self.intercept_, float):
            intercept_ = np.zeros(n_classes, dtype=np.float64,
                                  order="C")
        else:
            if self.intercept_.shape != (n_classes, ):
                raise ValueError("Provided intercept_ does not match dataset. ")
            intercept_ = self.intercept_

        X_data = np.array(X.data, dtype=np.float64, order="C")
        X_indices = X.indices
        X_indptr = X.indptr
        
        for i, c in enumerate(self.classes):
            Y_i = np.ones(Y.shape, dtype=np.float64) * -1.0
            Y_i[Y == c] = 1.0
            coef, intercept = plain_sgd(coef_[i],
                                        intercept_[i],
                                        self.loss_function,
                                        self.penalty_type,
                                        self.alpha, self.rho,
                                        X_data,
                                        X_indices, X_indptr, Y_i,
                                        self.n_iter,
                                        int(self.fit_intercept),
                                        int(self.verbose),
                                        int(self.shuffle))
            coef_[i] = coef
            intercept_[i] = intercept
            
        self._set_coef(coef_)
        self.intercept_ = intercept_

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : scipy.sparse matrix of shape [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels (either -1 or 1).
        """
        # Binary classification
        scores = self.predict_margin(X)
        print "Score", scores
        if self.classes.shape[0] == 2:
            indices = np.array(scores > 0, dtype=np.int)
        else:
            indices = scores.argmax(axis=1)
        print "indices", indices
        return self.classes[np.ravel(indices)]

    def predict_margin(self, X):
        """Predict signed 'distance' to the hyperplane (aka confidence score).

        Parameters
        ----------
        X : scipy.sparse matrix of shape [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples] with signed 'distances' to the hyperplane.
        """
        # np.dot only works correctly if both arguments are sparse matrices
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        print "sparse_coef_", self.sparse_coef_.shape
        print "X", X.shape
        print "np.dot(X, self.sparse_coef_.T).todense()",
        print np.dot(X, self.sparse_coef_.T).todense().shape
        scores = np.dot(X, self.sparse_coef_.T).todense() + self.intercept_
        if self.classes.shape[0] == 2:
            return np.ravel(scores)
        else:
            return scores

    def predict_proba(self, X):
        """Predict class membership probability.

        Parameters
        ----------
        X : scipy.sparse matrix of shape [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
            Contains the membership probabilities of the positive class.
        """
        # TODO change if multi class
        if isinstance(self.loss_function, Log) and \
               self.classes.shape[0] == 2:
            return 1.0 / (1.0 + np.exp(-self.predict_margin(X)))
        else:
            raise NotImplementedError('%s loss does not provide "\
            "this functionality' % self.loss)
