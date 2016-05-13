import numpy as np
import scipy as sci
from collections import Counter

def binary_search(lst, val):
    '''
    returns the highest index in sorted lst containing a value less than val
    '''
    i = len(lst)/2
    low = 0
    high = len(lst)
    while abs(low-high) > 1:
#         print i, low, high, lst[i]
        if val > lst[i]:
            if i == len(lst)-1:
                return i
            low = i
            i += len(lst[i:high])/2
        if val < lst[i]:
            high = i
            i -= len(lst[low:i])/2
        if val == lst[i]:
            return i
    return low

def sample_one_dist(n, pdf):
    '''
    take n samples from a one dim pdf
    '''
    total = 0
    cdf = [total]
    for prob in pdf:
        total += prob
        cdf.append(total)
    pulls = np.random.rand(n)
    samples = []
#     print cdf
    for i in pulls:
#         print i
        samples.append(binary_search(cdf, i))
    return samples

def create_distribution(n, d):
    '''
    Creates a set of random samples, with n samples and quantization d, that is,
    the distribution takes on 2^d different values

    the return value is a tuple, containing the generated samples and the pdf of the
    function that generated them
    '''
    pdf = np.random.rand(2**d)
    pdf = pdf/np.sum(pdf)
    samples = sample_one_dist(n, pdf)


    return samples, pdf


def create_dependent(px, k):
	'''
	Creates a dependent distribution based on the input distribution px, where
	the output distribution has k values

	returns the dependent (P(y|x)) pdf
	the dependent pdf has the row index as output value, and the column index as x value
	'''
	Pyx = np.random.rand(k, px.shape[0])
	Pyx = Pyx/np.sum(Pyx, axis = 0)
	return Pyx


def sum_dist(px, py, pi):
	'''
	returns the distribution x + y
	'''
	xvals = [i for i in range(px.shape[0])]
	yvals = [i for i in range(py.shape[0])]

	sumdist = Counter()
	for valx in xvals:
		for valy in yvals:
			sumdist[valx*pi[0]+valy*pi[1]] += px[valx]*py[valy]
	return np.array([val for val in sumdist.keys()])



def create_subdependent(x, y, dy):
	'''
	Creates a new distribution dependent on x with some varying dependence on
	y quantified by dy, with the same shape as y

	returns a tuple of the sample sand the joint pdf (in both x and y, dim = 3), 
	which generated the samples
	'''
	pass

def expand(pt, px):
    '''
    creates a matrix of dimension (len(px), len(pt))
    '''
    try:
        return np.array([pt for i in range(len(px))])
    except TypeError:
        return np.array([pt for i in range(px)])

def joint_dist(x, yx):
    '''
    Creates a joint distribution from a distribution p(x) and p(y|x) 
    the rows are the x values, the columns are the y values
    '''
    return (np.array(expand(x, yx.shape[0])) * yx).T

def product_dist(x, y):
    '''
    creates the product distribution of x,y, where x is the row index and
    y is the column index
    '''
    return np.array([x for i in range(len(y))]).T * np.array([y for i in range(len(x))])

def multi_sum_dist(px, py, pi):
    '''
    returns the distribution x + y for x,y of the same shape
    '''
    sumdist = Counter()
    for i in range(px.shape[0]):
        for j in range(px.shape[0]):
            for k in range(px.shape[1]):
                sumdist[i*pi[0]+j*pi[1], k] += px[i,k]*py[j,k]
#     print sumdist
    keys = sumdist.keys()
    keysx = list(set([keys[i][0] for i in range(len(keys))]))
    keysy = list(set([keys[i][1] for i in range(len(keys))]))
    keysx.sort()
    keysy.sort()
#     print keysx
#     print keysy
    values = []
    for x in keysx:
        yvals = []
        for y in keysy:
            if x == int(x) and y == int(y):
                yvals.append(sumdist[(x,y)])
        if yvals:
            values.append(yvals)
#     print values
    return np.asarray(values)
#     return np.array([sumdist[val] for val in keys if val == int(val) ])

def sum_dist(px, py, pi):
    '''
    returns the distribution x + y
    '''
    xvals = [i for i in range(px.shape[0])]
    yvals = [i for i in range(py.shape[0])]

    sumdist = Counter()
    for valx in xvals:
        for valy in yvals:
            
            sumdist[valx*pi[0]+valy*pi[1]] += px[valx]*py[valy]
#     print sumdist
    keys = sumdist.keys()
    keys.sort()
    return np.array([sumdist[val] for val in keys if val == int(val)])

def kl_divergence(px1, px2):
    '''
    computes the kl_divergence between two distributions assumed vector form
    '''
    return np.sum(px1 * np.log2(px1/px2))

def expanded_kl_divergence(pyx1, pyt2):
    '''
    computes the kl_divergence between two joint distributions,
    summing out the mutual variable, and returning a matrix of shape
    t by x
    '''
    joint_dist = np.ones((pyt2.shape[1], pyx1.shape[1]))
    for xi in range(pyx1.shape[0]):
        for ti in range(pyt2.shape[0]):
            joint_dist[ti, xi] = kl_divergence(pyx1[xi], pyt2[ti])
    return joint_dist

def js_divergence(px1, px2, pi):
    '''
    computes the js divergence between two distributions
    '''
    pbar = multi_sum_dist(px1, px2, pi)
#     print pbar
    return pi[0] * kl_divergence(px1, pbar) + pi[1] * kl_divergence(px2, pbar)

def entropy(px):
    '''
    Computes the entropy of a distributon with probability vector px
    '''
    return np.sum([-p*np.log2(p) for p in px])

def relative_entropy(px, pyx):
    '''
    Computes the relative entropy between two distributions, based on the dependence
    matrix
    '''
    return np.sum([psx*np.sum(
        [pyx[y,x] *np.log2(pyx[y,x]
            ) for y in range(pyx.shape[0])])
             for psx in px])
def kmeans(input, num_clusters, num_iters=50):
    '''
    performs k-means clustering on a set of data
    '''
    # choose clusters points as centers
    center_locs = []
    centers = []
    clusters = []

    prev_loss = -1
    for i in range(num_clusters):
        new_center = np.random.randint(len(images))
        while (new_center in center_locs):
            new_center = np.random.randint(len(images))
        center_locs.append(new_center)
        centers.append(images[new_center])
        clusters.append([])

    total_loss = 0
    curr_iters = 0
    while (prev_loss != total_loss and curr_iters < num_iters):
        prev_loss = total_loss
        total_loss = 0
        # assign points to clusters
        for image in images:
            best_center = 0
            best_dist = np.linalg.norm(image - centers[0])
            for i in range(len(centers)):
                dist = np.linalg.norm(image - centers[i])
                if dist < best_dist:
                    best_center = i
                    best_dist = dist
            # find loss
            clusters[best_center].append(image)
            total_loss += best_dist

        # update centers
        for i in range(len(clusters)):
            updated_center = np.mean(clusters[i], axis=0)
            centers[i] = updated_center
            
        curr_iters+=1
        
    # if loss changed, repeat as before
    
    return centers, clusters

def empirical_dist(x):
    '''
    Construct a distribution for samples x
    '''
    pxh = Counter()

    for val in x:
        pxh[val] += 1
    total = 0.0
    px = []
    for i in range(np.max(px.keys())):
        px.append(pxh[i])
        total += pxh[i]
    return np.array(px)/total

def empirical_dependent_dist(x, y):
    '''
    given x with corresponding labels y, determine the
    distribution of P(y|x)
    '''
    Pyxc = Counter()
    yvals = []
    xvals = []
    for xv in x:
        for yv in y:
            Pyxc[(y,x)] += 1
            yvals.append(y)
            xvals.append(x)
    yvals = set(yvals)
    xvals = set(xvals)
    colvals = []
    for xv in xvals:
        coltotal = 0
        for yv in yvals:
            coltotal += Pyxc[(yv, xv)]
        colvals.append(coltotal)
    colvals = np.array(colvals)
    ixv = list(xvals).sort()
    iyv = list(yvals).sort()
    Pyx = np.ones((np.max(iyv), np.max(ixv)))
    for i in range(np.max(iyv)):
        for j in range(np.max(ixv)): 
            Pyx[i,j]  = Pyxc[(i,j)]
    return Pyx/colvals

def hamming_distortion(x,y):
    '''
    for input vectors x, y of values, returns the hamming distortion
    of the values, as a matrix of x by y values
    '''
    values = np.ones((x.shape[0], y.shape[0]))
    for xi in range(x.shape[0]):
        for yi in range(y.shape[0]):
            if x[xi] == y[yi]:
                values[xi, yi] = 0
    return values

def blahut_arimoto(px, k, beta, distortion, epsilon = 1e-10): 
    '''
    finds the unique point of the rate distortion curve, for the
    distribution px, with distortion tradeoff beta and distortion function
    distortion, which takes in the vector of x values and vector of t values (
        0,...,x and 0,...,k respectively) and outputs the matrix of differences)
    '''
    # Ptx = np.array([0 for i in range(px.shape[0])])
    pt = np.random.rand(k)
    pt = pt/np.sum(pt)
#     print pt
    lastR = 99999999
    R = 9999999
    tvals = np.array([i for i in range(k)])
    xvals = np.array([i for i in range(len(px))])
    itr = 0
    while np.abs(lastR - R) > epsilon and itr < 10000:
        itr += 1 
        lastR = R
        pxboost = np.array([px for i in range(k)])
        ptboost = expand(pt, px)
        Ptx = ptboost.T * np.exp(-beta*distortion(tvals, xvals))
        ptnorm = np.sum(Ptx, axis = 0)
        Ptx = Ptx/ptnorm
        pt = np.sum(Ptx * pxboost, axis = 1)
        R = kl_divergence(joint_dist(px, Ptx), product_dist(px, pt))
    print itr
    return R, Ptx, pt, joint_dist(px, Ptx), product_dist(px, pt)

def iterative_IB(pxy, beta, k, epsilon  =1e-3):
    '''
    converges to a local optimum of the p(t|x) distribution 
    Note capital P denotes a probability distribution given a value
    and lower case is joint probability
    '''
    # Ptx = np.array([0 for i in range(px.shape[0])])
    Ptx = np.random.rand(k, pxy.shape[0])
    Ptx = Ptx/np.sum(Ptx, axis = 0)
    px = np.sum(pxy, axis = 1)/np.sum(pxy)
    pxboost = expand(px, k)
    Pyx = pxy.T/px
    pt = np.sum(Ptx * pxboost, axis = 1)
    Pyt = np.dot(Ptx,pxy)/pt
    itr = 0
    while itr<1000:
        itr += 1
        lastPtx = Ptx
        Ptx = np.array([pt for i in range(len(px))]).T * np.exp(-beta*expanded_kl_divergence(Pyx, Pyt))
        ptnorm = np.sum(Ptx, axis = 0)
        Ptx = Ptx/ptnorm
        pxboost = expand(px, k)
        pt = np.sum(Ptx * pxboost, axis = 1)
#         print pt
#         print Ptx
        Pyt = np.dot(Ptx,pxy).T/pt
        if np.abs(js_divergence(Ptx, lastPtx,(.5, .5))) <= epsilon:
            break
        
    print itr
    return Ptx, pt, Pyt

def determine_value(xsamples, Ptx, y):
	'''
	determines the effectiveness of a quantization at prediction
	of y based on the distribution and applied to the givne samples
	'''
	pass 
	
def IB_cluster():
	pass

def PCA():
	pass

def gaussian_IB():
	pass

def compute_covariance():
	pass

def nn_autoencoder():
	pass

