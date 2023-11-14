import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.decomposition import PCA
import numpy as np

class KeypointsMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KeypointsMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

input_dim = 2       # (x,y)
hidden_dim = 128
output_dim = 128    # 128-dimensional vector

mlp = KeypointsMLP(input_dim, hidden_dim, output_dim)

class KeypointEncoder(nn.Module):
    def __init__(self, kpts, des):
        super(KeypointEncoder, self).__init__()
        self.kpts = kpts
        self.des = des

    def forward(self, kpts, des):
        # print(type(kpts))
        # converting kpts to tensor
        kpts_pos = np.array([kp.pt for kp in kpts], dtype=np.float32)
        kpts_pos_tensor = torch.tensor(kpts_pos)
        # print(kpts_pos_tensor.shape)
        kpt_embeddings = mlp(kpts_pos_tensor)

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
        
        des_to_tensor = torch.from_numpy(des)
        # print(des_to_tensor.shape)
        output = des_to_tensor + kpt_embeddings
        # print(output)
        return output


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.query = nn.Linear(in_features, out_features * num_heads)
        self.key = nn.Linear(in_features, out_features * num_heads)
        self.value = nn.Linear(in_features, out_features * num_heads)
        self.fc_out = nn.Linear(out_features * num_heads, out_features)

    def forward(self, query, key, value):
        B, N, C = query.shape
        queries = self.query(query).view(B, N, self.num_heads, -1)
        keys = self.key(key).view(B, N, self.num_heads, -1)
        values = self.value(value).view(B, N, self.num_heads, -1)
        attention = torch.einsum("bnhc,bmhc->bnhm", [queries, keys]) / (C ** (1/2))
        attention = F.softmax(attention, dim=-1)
        out = torch.einsum("bnhm,bmhc->bnhc", [attention, values]).reshape(B, N, -1)
        return self.fc_out(out)


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads):
        super(GNNLayer, self).__init__(aggr='add')  # With aggregation
        self.attention = MultiHeadAttentionLayer(in_channels, out_channels, num_heads)
        self.lin = nn.Linear(in_channels, out_channels)
        self.num_heads = num_heads

    def forward(self, x, edge_index):
        "Propogation from all keypoiunts"
        print("Shape of x before reshape:", x.shape)
        B, N, C = 1, 200, 128 
        print("B:", B)
        print("N:", N)
        print("C:", C)
        x = x.view(B, N, C)
        print("Shape of x after reshape:", x.shape)
        print("Shape of edge_index:", edge_index.shape)
        print("Sample edge_index values:", edge_index[:, :10])
        output = self.propagate(edge_index, x=x, size=(N, N))
        print("Shape of output in forward:", output.shape)
        return output

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        print("x_j shape:", x_j.shape, "row max:", row.max(), "col max:", col.max())
        print("Sample row values:", row[:10], "Sample col values:", col[:10]) 
        src_features = x_j[:, row, :]
        dest_features = x_j[:, col, :]
        q = self.lin(src_features)
        k = self.lin(dest_features)
        v = self.lin(dest_features)
        return self.attention(q, k, v)

class AttentionalGNN(nn.Module):
    def __init__(self, num_features, num_heads=4):
        super(AttentionalGNN, self).__init__()
        self.conv1 = GNNLayer(num_features, 128, num_heads)
        self.conv2 = GNNLayer(128, 128, num_heads)


    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        print("Shape of x after conv1:", x.shape)
        x = self.conv2(x, edge_index)
        print("Shape of x after conv2:", x.shape)
        return F.log_softmax(x, dim=1)

