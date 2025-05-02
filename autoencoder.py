import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GATConv
from torch_geometric.nn import global_add_pool
from transformers import AutoModel, AutoTokenizer

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, d_cond): # n_cond, d_cond are added
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        
        ###### ADDED ##########
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )
        bn_layers = [nn.BatchNorm1d(hidden_dim) for i in range(n_layers-1)]
        self.bn = nn.ModuleList(bn_layers)
        #####################
        
        mlp_layers = [nn.Linear(latent_dim+d_cond, hidden_dim)] + [nn.Linear(hidden_dim+d_cond, hidden_dim) for i in range(n_layers-2)]  ## MODIFIED
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond):
        
        cond = self.cond_mlp(cond)### ADDED  
        
        for i in range(self.n_layers-1):
            x = torch.cat((x, cond), dim=1) ### ADDED
            x = self.relu(self.mlp[i](x))
            x = self.bn[i](x) ###ADDED
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))  # (batch_size,num_edges, 2)
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]  # (batch_size,num_edges)

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)  # (2, num_indices = num_edges)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj




class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
        
        out = global_add_pool(x, data.batch)   # x: (num_nodes,hidden_dim)    out: (n_graphs,hidden_dim)
        out = self.bn(out)
        out = self.fc(out)
        return out





class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, heads=4, dropout=0.2):
        """
        Args:
            input_dim (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of hidden layers.
            latent_dim (int): Dimensionality of the output latent space.
            n_layers (int): Number of GAT layers.
            heads (int): Number of attention heads in each GAT layer.
            dropout (float): Dropout rate.
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # GAT Layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))  # First layer with multiple heads
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))  # Hidden layers
        
        # Batch normalization for hidden layers
        self.bn = nn.BatchNorm1d(hidden_dim * heads)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * heads, latent_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        # Apply GAT layers
        for conv in self.convs:
            x = conv(x, edge_index)  # GAT layer
            x = F.elu(x)             # Activation function
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout for regularization

        # Global pooling over all nodes in each graph
        out = global_add_pool(x, data.batch)  # Shape: (batch_size, hidden_dim * heads)
        
        # Batch normalization and output layer
        out = self.bn(out)
        out = self.fc(out)  # Shape: (batch_size, latent_dim)
        
        return out
    
    
    
    

# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, d_cond, encoder_type, bert_model_name="bert-base-uncased"):  #  d_cond are added
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        if encoder_type == 'GIN':
            self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        elif encoder_type == 'GAT':
            self.encoder = GAT(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        else:
            raise Exception(f"The type of encoder named {encoder_type} is not found")
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, d_cond)   #  d_cond are added 


    def forward(self, data, cond):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, cond)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar, cond):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g, cond)
       return adj

    def decode_mu(self, mu, cond):
       adj = self.decoder(mu, cond)
       return adj

    def loss_function(self, data, cond, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, cond)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld
