import numpy as np 
import sys 
sys.path.insert(0, '../data')
import data_utils
import nimfa



V = np.random.rand(30, 20)

init_W = np.random.rand(30, 4)
init_H = np.random.rand(4, 20)

# Fixed initialization of latent matrices
nmf = nimfa.Nmf(V, seed="fixed", W=init_W, H=init_H, rank=4)
nmf_fit = nmf()

# print("Euclidean distance: %5.3f" % nmf_fit.distance(metric="euclidean"))
# print('Initialization type: %s' % nmf_fit.seeding)
# print('Iterations: %d' % nmf_fit.n_iter)

W = nmf_fit.basis()
print('Basis matrix:\n%s' % W)

H = nmf.coef()
print('Mixture matrix:\n%s' % H)