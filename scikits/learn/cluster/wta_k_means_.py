""" K-means clustering
"""

# Authors: Gael Varoquaux <gael.xaroquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
# License: BSD

import warnings

import numpy as np

from ..base import BaseEstimator
from ..metrics.pairwise import euclidean_distances


###############################################################################
# Initialisation heuristic

#def k_init(X, k, rng=None):
#    """Init k seeds with assignment from first random k samples

#    Parameters
#    -----------
#    X: array, shape (n_samples, n_features)
#        The data

#    k: integer
#        The number of seeds to choose

#    Notes
#    ------
#	Selects initial cluster centers for w-t-a sequential k-means clustering in a
#	naive way. 
#    """

#    n_samples = X.shape[0]
#    if rng is None:
#        rng = np.random

#    """if n_samples >= n_samples_max:
#        X = X[rng.randint(n_samples, size=n_samples_max)]
#        n_samples = n_samples_max"""

#    #distances = euclidean_distances(X, X, squared=True)

#	centers = []
#    # choose the k initial centers randomly (can probably do this better...)
#	for i in range(k)	
#	    idx = rng.randint(n_samples)
#    	centers.append(X[idx])

#    return np.array(centers)


###############################################################################
# K-means estimation by incremental winner-take-all

def wta_k_means(X, k, init='random', max_iter=1500, verbose=1,
                    tol=1e-4, rng=None, copy_x=True):
    """ K-means clustering algorithm.

    Parameters
    ----------
    X: ndarray
        A M by N array of M observations in N dimensions or a length
        M array of M one-dimensional observations.

    k: int or ndarray
        The number of clusters to form as well as the number of
        centroids to generate. If minit initialization string is
        'matrix', or if a ndarray is given instead, it is
        interpreted as initial cluster to use instead.

    max_iter: int, optional, default 0
        Maximum number of iterations of the k-means algorithm to run.
		0 implies no maximum.

    init: {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

    tol: float, optional
        The relative increment in the results before declaring convergence.

    verbose: boolean, optional
        Verbosity mode

    rng: numpy.RandomState, optional
        The generator used to initialize the centers. Defaults to numpy.random.

    copy_x: boolean, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    Returns
    -------
    centroid: ndarray
        A k by N array of centroids found at the last iteration of
        k-means.

    label: ndarray
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia: float
        The final value of the inertia criterion

    """
    if rng is None:
        rng = np.random
    n_samples = X.shape[0]

    vdata = np.mean(np.var(X, 0))
    #best_inertia = np.infty
    labels = -np.ones(n_samples).astype(np.int)
    if hasattr(init, '__array__'):
        init = np.asarray(init)
    'subtract of mean of x for more accurate distance computations'
    Xmean = X.mean(axis=0)
    if copy_x:
        X = X.copy()
    X -= Xmean
    # init
    if init == 'random':
        seeds = np.argsort(rng.rand(n_samples))[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        centers = np.asanyarray(init).copy()
    elif callable(init):
        centers = init(X, k, rng=rng)
    else:
        raise ValueError("the init parameter for the k-means should "
            "be 'random' or an ndarray, "
            "'%s' (type '%s') was passed.")

    if verbose:
        print 'WTA k-means Initialization complete'
    
    # average squared distance to all points
    inertia = np.mean(euclidean_distances(centers, X, squared=True))
    # counts for assignments  
    assignments = np.zeros(k).astype(np.int)
    # starvation trace for augmenting winner selection
    starvation_trace = np.ones((k,1)).astype(np.float32)
    # M2 metric used in online center variance computation
    M2 = np.zeros((k,X.shape[1])).astype(np.float32)
    variance = np.zeros((k,X.shape[1])).astype(np.float32)
    # set up constant step size (suboptimal for sure)
    stepsize = 0.15*np.min(Xmean)
    # iterations
    i = 0
    while (max_iter==0 or i<=max_iter):
        centers_old = centers.copy()
        # pull k samples to update before stepping iteration
        samples = rng.randint(n_samples,size=k)
        for sampleIdx in samples:
            # use WTA assignment to update center
            centers,assignments,predelta,dist,winner = adjustment_step(X[sampleIdx],centers,stepsize,assignments,starvation_trace,i)
            # record keeping on where samples are assigned
            labels[sampleIdx] = winner
            # update variance of centroid from adjustment
            M2[winner],variance[winner] = update_online_variance(assignments[winner],predelta,centers[winner],X[sampleIdx],M2[winner])
            inertia = (1-(stepsize/n_samples))*inertia + stepsize*(1/n_samples)*dist
            # update starvation trace by reducing all traces, then upping the selected winner's trace 
            starvation_trace = (1-0.05)*starvation_trace
            starvation_trace[winner] = 0.05*(starvation_trace[winner]+1) 

        if verbose and not (i % 5):
            print 'Iteration %i, centroid updates %i, inertia %s' % (i, (i+1)*k, inertia)
        #if np.sum((centers_old - centers) ** 2) < tol * vdata:
        #   if verbose:
        #       print 'Converged to similar centers at iteration', iter
        #   break
        i += 1

    if not copy_x:
        X += Xmean
   
    varianceF = np.zeros((k,X.shape[1])).astype(np.float32) 
    counts = np.zeros(k).astype(np.int)
    # recalculate each assignment and do final variance computation
    for sample in range(n_samples):
        # compute distance to each center to select winner
        distances = euclidean_distances(centers, [X[sample]], squared=True)
        winner = np.argmin(distances)
        varianceF[winner] = varianceF[winner] + (X[sample]-centers[winner])*(X[sample]-centers[winner])
        counts[winner] = counts[winner]+1
    # normalize all the variances by their assignments
    for centroid in range(k):
        if counts[centroid] is not 0:
            varianceF[centroid] = varianceF[centroid]/counts[centroid];

    #import pdb; pdb.set_trace()
    return centers + Xmean, labels, inertia

def adjustment_step(x, centers, stepsize, assignments, starvation_trace, current_iter, x_squared_norm=None):
    """ WTA Adjustment step for k-means

    Computes "winner" for datapoint and adjusts winning mean.

    Parameters
    ----------
    x: datapoint of shape (p)
      	p = number of features
    centers: array of shape (k, p)
        The cluster centers

    Returns
    -------
    centers, array of shape (k, p)
        The resulting centers
    mindist: float
        The absolute distance of the sample before this assignment
    dist: float
        The Euclidean distance of the sample after this assignment
    """

    n_samples = x.shape[0]
    k = centers.shape[0]

    # compute distance to each center to select winner
    distances = euclidean_distances(centers, [x], x_squared_norm, squared=True)
    if current_iter <= 500:
        distances_adjusted = distances*starvation_trace
        winner = np.argmin(distances_adjusted)
    else:
        winner = np.argmin(distances)
   
    mindist = distances[winner]

    # increment count for this assignment 
    assignments[winner] += 1

    # calculate absolute distance before change (to later use in online variance estimation)
    delta = (x-centers[winner])

    # update winner by pulling it closer to sample
    #centers[winner] = centers[winner] + stepsize*(x - centers[winner])
    centers[winner] = centers[winner] + (1/assignments[winner])*(delta)

    # evaluate new distance and return for some measure of how things are going
    dist = np.sum((x - centers[winner]) ** 2)

    return centers,assignments,delta,dist,winner

def update_online_variance(num_samples, predelta, mean, x, M2):
    """ From wikipedia: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    M2 = M2 + predelta*(x-mean) 
    population_variance = M2/num_samples
    #if not n == 1
    #    sample_variance = M2/(n-1)
    #else sample_variance = population_variance
    return M2,population_variance


#def _m_step(x, z, k):
#    """ M step of the K-means EM algorithm
 
#    Computation of cluster centers/means

#    Parameters
#    ----------
#    x array of shape (n,p)
#      n = number of samples, p = number of features
#    z, array of shape (x.shape[0])
#        Current assignment
#    k, int
#        Number of desired clusters

#    Returns
#    -------
#    centers, array of shape (k, p)
#        The resulting centers
#    """
#    dim = x.shape[1]
#    centers = np.repeat(np.reshape(x.mean(0), (1, dim)), k, 0)
#    for q in range(k):
#        if np.sum(z == q) == 0:
#            pass
#        else:
#            centers[q] = np.mean(x[z == q], axis=0)
#    return centers


#def _e_step(x, centers, precompute_distances=True, x_squared_norms=None):
#    """E step of the K-means EM algorithm

#    Computation of the input-to-cluster assignment

#    Parameters
#    ----------
#    x: array of shape (n, p)
#      n = number of samples, p = number of features

#    centers: array of shape (k, p)
#        The cluster centers

#    Returns
#    -------
#    z: array of shape(n)
#        The resulting assignment

#    inertia: float
#        The value of the inertia criterion with the assignment
#    """

#    n_samples = x.shape[0]
#    k = centers.shape[0]

#    if precompute_distances:
#        distances = euclidean_distances(centers, x, x_squared_norms,
#                                        squared=True)
#    z = -np.ones(n_samples).astype(np.int)
#    mindist = np.infty * np.ones(n_samples)
#    for q in range(k):
#        if precompute_distances:
#            dist = distances[q]
#        else:
#            dist = np.sum((x - centers[q]) ** 2, axis=1)
#        z[dist < mindist] = q
#        mindist = np.minimum(dist, mindist)
#    inertia = mindist.sum()
#    return z, inertia


class WTAKMeans(BaseEstimator):
    """ Winner-take-all K-Means clustering

    Parameters
    ----------

    data : ndarray
        A M by N array of M observations in N dimensions or a length
        M array of M one-dimensional observations.

    k : int or ndarray
        The number of clusters to form as well as the number of
        centroids to generate. If init initialization string is
        'matrix', or if a ndarray is given instead, it is
        interpreted as initial cluster to use instead.

    max_iter : int
        Maximum number of iterations of the k-means algorithm for a
        single run. 0 implies no max and is default.

    init : {'k-means++', 'random', 'points', 'matrix'}
        Method for initialization, defaults to 'random':

        'points': choose k observations (rows) at random from data for
        the initial centroids.

        'matrix': interpret the k parameter as a k by M (or length k
        array for one-dimensional data) array of initial centroids.

    tol: float, optional default: 1e-4
        Relative tolerance w.r.t. inertia to declare convergence

    Methods
    -------

    fit(X):
        Compute K-Means clustering

    Attributes
    ----------

    cluster_centers_: array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_:
        Labels of each point

    inertia_: float
        The value of the inertia criterion associated with the chosen
        partition.

    Notes
    ------

    The k-means problem is solved using the Lloyd algorithm.

    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the K-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.
    """

    def __init__(self, k=8, init='random', max_iter=1500, tol=1e-4,
            verbose=1, rng=None, copy_x=True):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.rng = rng
        self.copy_x = copy_x

    def fit(self, X, **params):
        """Compute k-means"""
        X = np.asanyarray(X)
        self._set_params(**params)
        self.cluster_centers_, self.labels_, self.inertia_ = wta_k_means(
            X, k=self.k, init=self.init,
            max_iter=self.max_iter, verbose=self.verbose,
            tol=self.tol, rng=self.rng, copy_x=self.copy_x)
        return self
