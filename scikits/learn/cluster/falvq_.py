""" Fuzzy Algorithm for Learning Vector Quantization
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
# Online competitive learning step

def falvq_learning(X, k, init='random', max_iter=1500, verbose=1,
                    tol=1e-4, rng=None, copy_x=True, step_initial=0.005):
    """ Fuzzy algorithms for learning vector quantization

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

    # Steps for FALVQ Algorithm
    # 1. Select c (given here), 
    #     step_initial as learning rate parameter (also passed as a parameter),
    #     timestep = 0,
    #     and initial codebook (set of centers V0={v1,0, v2,0, ..., vc,0})
    # 2. Calculate learning rate = initial_rate * (1-timeStep/max_iter)
    # 3. Set timeStep = timeStep+1 (this is done somewhat out of order here, without issues)
    # 4. For each input vector
    #     find index of "winning" center
    #     calculate membership value for all non-winning centers
    #     calculate interference from nonwinning prototypes (only sum is used in update)
    #     calculate interference from winning prototype
    #     update winning centers
    #     update all non-winning centers
    # 5. If timestep < N, goto 2.

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
        print 'FALVQ Initialization complete'
    
    # average squared distance to all points
    inertia = np.mean(euclidean_distances(centers, X, squared=True))
    # counts for assignments  
    assignments = np.zeros(k).astype(np.int)
    # starvation trace for augmenting winner selection
    starvation_trace = np.ones((k,1)).astype(np.float32)
    # M2 metric used in online center variance computation
    M2 = np.zeros((k,X.shape[1])).astype(np.float32)
    variance = np.zeros((k,X.shape[1])).astype(np.float32)
    # iterations
    i = 0
    while (max_iter==0 or i<=max_iter):
        # adjust learning rate (note i=0 is step_initial)
        stepsize = step_initial*(1-i/max_iter)

        #centers_old = centers.copy()
        # pull k samples to update before stepping iteration
        samples = rng.randint(n_samples,size=k)
        for sampleIdx in samples:
            # use WTA assignment to update center
            centers,assignments,predelta,winner = adjustment_step(X[sampleIdx],centers,stepsize,assignments,i)
            # record keeping on where samples are assigned
            labels[sampleIdx] = winner
            # update variance of centroid from adjustment
            M2[winner],variance[winner] = update_online_variance(assignments[winner],predelta,centers[winner],X[sampleIdx],M2[winner])
            #inertia = (1-(stepsize/n_samples))*inertia + stepsize*(1/n_samples)*dist

        if verbose and not (i % 5):
            print 'Iteration %i, centroid updates %i' % (i, (i+1)*k)
        #if np.sum((centers_old - centers) ** 2) < tol * vdata:
        #   if verbose:
        #       print 'Converged to similar centers at iteration', iter
        #   break
        i += 1

    if not copy_x:
        X += Xmean
  
    # -- final variance calculation, updating done -- 
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

def membership_function(X, centers, x_squared_norm=None):
    """ Membership function for FALVQ

    Calculate membership values for all examples in X.

    Parameters
    ----------
    X: ndarray
        A M by N array of M observations in N dimensions or a length
        M array of M one-dimensional observations.

    centers: ndarray
        Cluster center matrix of size k by N where k is number of
        centers. 

    Returns
    ----------
    membership values, array of shape (M,k)
    sum_w, interference from nonwinners, sum of values, array of shape (M,1)
    n, interference from winner, array of shape (M,k)
    """
    if(len(X.shape) is not 1):
        M = X.shape[0]
    else:
        M = 1
    k = centers.shape[0]
    #import pdb; pdb.set_trace()

    memberships = np.zeros((M,k)).astype(centers.dtype)
    sum_w = np.zeros(M).astype(np.float32)
    n = np.zeros((M,k)).astype(np.float32)
    winners = np.zeros(M).astype(np.int)

    for sample in range(M):  
        # compute distance to each center to select winner
        if M == 1:
            distances = euclidean_distances(centers, [X], x_squared_norm, squared=True)
        else:
            distances = euclidean_distances(centers, [X(sample)], x_squared_norm, squared=True)

        winner = np.argmin(distances)
        #mindist = distances[winner]
        #import pdb; pdb.set_trace()

        input_args = (distances[winner]/distances).T
        memberships[sample,:] = FALVQ1_membership(input_args)
        w = FALVQ1_interference_from_nonwinning(input_args)
        n[sample,:] = FALVQ1_interference_from_winning(input_args)

        # mark winning centroid 
        memberships[sample,winner] = 1
        n[sample,winner] = 0
        w[0,winner] = 0
        sum_w[sample] = np.sum(w)
 

        #memberships[sample, winner] = 1 
        #input_arg = distances[winner]/distances;

        #for centroid in range(k):
        #    if centroid is not winner:
        #        input_arg = distances[winner]/distances[centroid]
        #        memberships[sample, centroid] = FALVQ1_membership(input_arg)
        #        # find sum of interference values for winner update
        #        sum_w[sample] = sum_w[sample] + FALVQ1_interference_from_nonwinning(input_arg)
        #        n[sample,centroid] = FALVQ1_interference_from_winning(input_arg)

        winners[sample] = winner
    
    if M == 1:
        n.shape = (k);

    return memberships, sum_w, n, winners

def FALVQ1_membership(x, alpha=1.0):
    return x/(1+alpha*x)

def FALVQ1_interference_from_nonwinning(x, alpha=1.0):
    return 1/((1+alpha*x)*(1+alpha*x))

def FALVQ1_interference_from_winning(x, alpha=1.0):
    return alpha*x*x/((1+alpha*x)*(1+alpha*x))

def adjustment_step(x, centers, stepsize, assignments, current_iter, x_squared_norm=None):
    """ Fuzzy Adjustment step for FALVQ

    Computes "winner" for datapoint and adjusts winning mean along with non-winners with fuzzy rules.

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

    # find winner and calculate membership values
    u,sum_w,n,tmp = membership_function(x,centers)
    winner = tmp[0]
 
    interference = n
    interference[winner] = 1+sum_w
    interference.shape = (k,1)

    # increment count for this assignment 
    assignments[winner] += 1

    # calculate absolute distance before change (to later use in online variance estimation)
    delta = (x-centers[winner])

    # update all centers simultaneously with appropriate weighting
    centers = centers + stepsize*(x-centers)*interference

    # update winner by pulling it closer to sample
    #centers[winner] = centers[winner] + stepsize*(x - centers[winner])*(1+sum_w)
    #centers[winner] = centers[winner] + stepsize*(x - centers[winner])
    #centers[winner] = centers[winner] + (1/assignments[winner])*(delta)

    # update all other centers
    #for c in range(k):
    #    if c is not winner:
    #        centers[c] = centers[c] + stepsize*(x - centers[c])*n[c]
    
    # evaluate new distance and return for some measure of how things are going
    #dist = np.sum((x - centers[winner]) ** 2)

    return centers,assignments,delta,winner

def update_online_variance(num_samples, predelta, mean, x, M2):
    """ From wikipedia: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    M2 = M2 + predelta*(x-mean) 
    population_variance = M2/num_samples
    #if not n == 1
    #    sample_variance = M2/(n-1)
    #else sample_variance = population_variance
    return M2,population_variance


#    dim = x.shape[1]
#    centers = np.repeat(np.reshape(x.mean(0), (1, dim)), k, 0)
#    for q in range(k):
#        if np.sum(z == q) == 0:
#            pass
#        else:
#            centers[q] = np.mean(x[z == q], axis=0)


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


class FALVQ(BaseEstimator):
    """ Fuzzy Algorithms for Learning Vector Quantization

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
        Compute FALVQ clustering

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
        self.cluster_centers_, self.labels_, self.inertia_ = falvq_learning(
            X, k=self.k, init=self.init,
            max_iter=self.max_iter, verbose=self.verbose,
            tol=self.tol, rng=self.rng, copy_x=self.copy_x)
        return self
    
    def membership_from_distances(self,distances):
        """ Compute membership values from distance matrix
            
        Parameters
        -----
            distances: M by k array of distance norms
        """
        memberships = np.zeros(distances.shape, dtype=distances.dtype)
        winners = np.argmin(distances,axis=1)

        for example in range(distances.shape[0]):
            #memberships[example,winner[example]] = 1
            input_args = distances[example,winners[example]]/distances[example]
            memberships[example,:] = FALVQ1_membership(input_args)
            #for c in range(distances.shape[1]):
            #    if c is not winner[example]:
            #        memberships[example,c] = FALVQ1_membership(distances[example,winner[example]]/distances[example,c])

        # mark winning centroids for each sample 
        memberships[range(distances.shape[0]),winners] = 1

        return memberships

