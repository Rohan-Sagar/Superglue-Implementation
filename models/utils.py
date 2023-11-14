from scipy.spatial import KDTree
import numpy as np
import torch

# Padding for same size tensors
def pad_tensor(tensor, target_size):
    padding_size = target_size - tensor.size(0)
    if padding_size > 0:
        padding = torch.zeros(padding_size, tensor.size(1))
        padded_tensor = torch.cat([tensor, padding], dim=0)
        return padded_tensor
    return tensor


def create_edges(kpts1, kpts2, max_neighbors=10):
    print("Hello!!")
    num_kpts1 = len(kpts1)
    num_kpts2 = len(kpts2)
    self_edges = []
    cross_edges = []

    max_self_index = 0
    max_cross_index = 0

    for i, kpt1 in enumerate(kpts1):
        neighbors = find_nearest_neighbors(kpt1, kpts1, max_neighbors)
        for n in neighbors:
            if n != i and (i, n) not in self_edges:
                self_edges.append((i, n))

    for i, kpt2 in enumerate(kpts2):
        neighbors = find_nearest_neighbors(kpt2, kpts2, max_neighbors)
        for n in neighbors:
            adjusted_index = num_kpts1 + n
            if adjusted_index != num_kpts1 + i and (num_kpts1 + i, adjusted_index) not in self_edges:
                self_edges.append((num_kpts1 + i, adjusted_index))

    for i, kpt1 in enumerate(kpts1):
        cross_neighbors = find_nearest_neighbors(kpt1, kpts2, max_neighbors)
        for n in cross_neighbors:
            cross_edges.append((i, num_kpts1 + n))
            max_cross_index = max(max_cross_index, num_kpts1 + n)

    for i, kpt2 in enumerate(kpts2):
        cross_neighbors = find_nearest_neighbors(kpt2, kpts1, max_neighbors)
        for n in cross_neighbors:
            cross_edges.append((num_kpts1 + i, n))
            max_cross_index = max(max_cross_index, n)

    print("Num self edges", len(self_edges))
    print("Num cross edges", len(cross_edges))
    print("Max self edge index:", max_self_index)
    print("Max cross edge index:", max_cross_index)

    return self_edges + cross_edges

def find_nearest_neighbors(ref, kpts, max_neighbors):
    kpts_array = np.array([k.pt for k in kpts])
    tree = KDTree(kpts_array)
    k = min(len(kpts), max_neighbors + 1)
    dist, idx = tree.query(ref.pt, k = k)
    return idx[1:] if len(kpts) > 1 else [] 
