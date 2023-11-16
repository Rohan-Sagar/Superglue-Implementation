import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.decomposition import PCA
import numpy as np
from copy import deepcopy

class KeypointsMLP(nn.Module):
    "This class transforms the coordinates to a 128-dimension (output_dim size) vector"
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=128):
        super(KeypointsMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.layers(x)

class KeypointEncoder(nn.Module):
    def __init__(self, kpts, des):
        super(KeypointEncoder, self).__init__()
        self.kpts = kpts
        self.des = des
        self.mlp = KeypointsMLP()

    def forward(self):
        kpts_pos = np.array([kp.pt for kp in self.kpts], dtype=np.float32)
        kpts_pos_tensor = torch.tensor(kpts_pos)
        kpt_embeddings = self.mlp(kpts_pos_tensor)

        # embeddings_np = kpt_embeddings.detach().numpy()
        #
        # pca = PCA(n_components=2)
        # reduced_embeddings = pca.fit_transform(embeddings_np)
        #
        # plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
        # plt.xlabel('Component 1')
        # plt.ylabel('Component 2')
        # plt.title('PCA of Keypoint Embeddings')
        # plt.show()
        
        des_to_tensor = torch.from_numpy(self.des)
        output = des_to_tensor + kpt_embeddings
        return output

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.preprocess = nn.Linear(d_model, d_model)
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
    
    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [self.preprocess(x) for x in (query, key, value)]
        query, key, value = [x.view(batch_dim, -1, self.num_heads, self.dim).transpose(1, 2)
                             for x in (query, key, value)]
        x, _ = attention(query, key, value)
        print("MHA X shape(1):", x.shape)
        x = x.transpose(1, 2).contiguous().view(batch_dim, self.dim * self.num_heads, -1)
        print("MHA X shape(2):", x.shape)
        return self.merge(x)


class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(GNNLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads, in_channels)
        self.mlp = KeypointsMLP(in_channels * 2, out_channels, out_channels)

    def forward(self, x, edges):
        print("X shape:", x.shape)               # [num_nodes, num_features]
        print("Edges shape:", edges.shape)
        edges = edges.transpose(0, 1)
        print("Edges shape:", edges.shape)       # Row 1: Src Nodes, Row 2: Dest Nodes , Format: [2, num_edges]
        row, col = edges
        src_features = x[row]
        dest_features = x[col]

        message = self.attention(src_features, dest_features, dest_features)
        print("Message shape(1)", message.shape)
        message = message.squeeze(-1)           # Removing last dim since its size 1
        print("Message shape(2)", message.shape)
        aggregated_message = torch.zeros_like(x)
        for node_idx in range(x.size(0)):
            aggregated_message[node_idx] = message[col == node_idx].sum(dim=0)

        node_features = torch.cat([x, aggregated_message], dim=1)
        return self.mlp(node_features)


class AttentionalGNN(nn.Module):
    def __init__(self, num_features, layer_count, num_heads=4):
        super(AttentionalGNN, self).__init__()
        self.layers = nn.ModuleList([
            GNNLayer(num_features, num_features, num_heads)
            for _ in range(layer_count)])

    def forward(self, desc1, desc2, edges):
        concat_desc = torch.cat((desc1, desc2), dim=0)

        for layer in self.layers:
            concat_desc = layer(concat_desc, edges)

        split_idx = desc1.shape[0]
        desc1_new, desc2_new = concat_desc[:split_idx], concat_desc[split_idx:]
        combined = torch.cat((desc1_new.unsqueeze(1), desc2_new.unsqueeze(1)), dim=1)
        "Returning a tensor for computing similiary"
        return combined 


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    if scores.dim() == 2:
        scores = scores.unsqueeze(0)
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def compute_similarity(gnn_output):
    gnn_output = torch.nn.functional.normalize(gnn_output, dim=2)
    descriptors1, descriptors2 = torch.split(gnn_output, gnn_output.size(1) // 2, dim=1)
    similarity_scores = torch.matmul(descriptors1.transpose(1, 2), descriptors2).squeeze(0)
    return similarity_scores
