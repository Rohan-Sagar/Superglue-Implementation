import torch
import torch.nn as nn
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
