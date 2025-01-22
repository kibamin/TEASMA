import os
import numpy as np
from dnn_tip.surprise import LSA, DSA

def get_sc(sa, k=1000):
    """Surprise Coverage

    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        sa (list): List of lsa or dsa.

    Returns:
        cov (int): Surprise coverage.
    """
    lower, upper = np.amin(sa), np.amax(sa)
    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k)


train_ATs = np.load("SA_results/lenet5_mnist/mnist_train_dense_1_ats.npy")
test_ATs = np.load("SA_results/lenet5_mnist/mnist_test_dense_1_ats.npy")

train_pred = np.load('SA_results/lenet5_mnist/mnist_train_pred.npy')
test_pred = np.load('SA_results/lenet5_mnist/mnist_test_pred.npy')

lsa = False
dsa = True


if lsa:
    sa = LSA(train_ATs)
    lsa_output = sa(test_ATs)
    
if dsa:
    sa = DSA(train_ATs, train_pred) 
    dsa_output = sa(test_ATs, test_pred) 


dsa_output_from_original = np.load("SA_results/lenet5_mnist/DSA/test_dsa_dense_1.npy")

print(dsa_output_from_original.shape)


from scipy.spatial.distance import cosine
cos_sim = 1 - cosine(dsa_output_from_original, dsa_output)  # cosine returns the cosine distance, so subtract from 1
print("Cosine Similarity:", cos_sim)


from scipy.spatial.distance import euclidean

# Compute Euclidean distance
euclid_dist = euclidean(dsa_output_from_original, dsa_output)

print("Euclidean Distance:", euclid_dist)


sc_output_weiss = get_sc(dsa_output)
sc_output_original = get_sc(dsa_output_from_original)

print(sc_output_weiss)
print(sc_output_original)