import torch
import torch.nn as nn
from omegaconf import ListConfig, OmegaConf

from einops import rearrange

class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super(DenseNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        # Convert out_dim to list if it's not already
        if not isinstance(out_dim, list):
            self.out_dim = list(out_dim)
        
        flatten_dim = torch.prod(torch.tensor(self.out_dim))
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, flatten_dim)
        )
    
    def forward(self, x):
        assert x.shape[1] == self.embedding_dim, f"Input must have shape [B, {self.embedding_dim}]"
        output = self.network(x)
        if len(self.out_dim) == 2:
            output = rearrange(output, 'b (t a) -> b t a', t=self.out_dim[0], a=self.out_dim[1])
        return output
    
class DenseNetwork_lelan(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super(DenseNetwork_lelan, self).__init__()
        
        # self.max_linvel = 0.5
        # self.max_angvel = 1.0
        self.embedding_dim = embedding_dim 
        self.out_dim = out_dim
        # Convert out_dim to list if it's not already
        if not isinstance(out_dim, list):
            self.out_dim = list(out_dim)
        
        flatten_dim = torch.prod(torch.tensor(self.out_dim))
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, flatten_dim),       
            # nn.Sigmoid() # Remove this since we manage the normalization separately
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        if len(self.out_dim) == 2:
            output = rearrange(output, 'b (t a) -> b t a', t=self.out_dim[0], a=self.out_dim[1])

        # linear_vel = self.max_linvel*output[:, 0:self.out_dim[0]]  #max +0.5 m/s min 0.0 m/s
        # angular_vel = self.max_angvel*2.0*(output[:, self.out_dim[0]:self.out_dim[1]*self.out_dim[0]] - 0.5)  #max +1.0 rad/s min -1.0 rad/s
        # import pdb; pdb.set_trace()  # Debugging line to inspect the output
        return output