import torch
import torch.nn.functional as F

def dot_scorer(action_emb, item_emb, item_dim):
    '''
    score = item_emb * weight
    
    @input:
    - action_emb: (B, i_dim)
    - item_emb: (B, L, i_dim) or (1, L, i_dim)
    @output:
    - score: (B, L)
    '''
    output = torch.sum(action_emb.view(-1,1,item_dim) * item_emb, dim = -1)
    return output

def linear_scorer(action_emb, item_emb, item_dim ):
    '''
    score = item_emb * weight + bias
    
    @input:
    - action_emb: (B, (i_dim+1))
    - item_emb: (B, L, i_dim) or (1, L, i_dim)
    @output:
    - score: (B, L)
    '''
    fc_weight = action_emb[:, :item_dim].view(-1, 1, item_dim) # * 2 / math.sqrt(self.item_dim)
    fc_bias = action_emb[:,-1].view(-1, 1)

    output = torch.sum(fc_weight * item_emb, dim = -1) + fc_bias
    return output


def two_layer_mlp_scorer(action_emb, item_emb, item_dim, hidden_dim ):
    '''
    score = MLP(item_emb)
    - h1 = leaky_relu(item_emb * fc1_weight + fc1_bias)
    - score = h1 * fc2_weight + fc2_bias
    
    @input:
    - action_emb: (B, (i_dim+1)*f_dim+f_dim+1)
    - item_emb: (B, L, i_dim) or (1, L, i_dim)
    
    @output:
    - score: (B, L)
    '''
    fc1_weight = action_emb[:, :item_dim*hidden_dim].view(-1, item_dim, hidden_dim)
    fc1_bias = action_emb[:,item_dim*hidden_dim:(item_dim+1)*hidden_dim].view(-1, 1, hidden_dim)
    fc2_weight = action_emb[:, (item_dim+1)*hidden_dim:(item_dim+2)*hidden_dim].view(-1, 1, hidden_dim)
    fc2_bias = action_emb[:,-1].view(-1, 1)

    h1 = torch.matmul(item_emb, fc1_weight) + fc1_bias
    h1 = F.leaky_relu(h1)
    output = torch.sum(h1 * fc2_weight, dim = -1) + fc2_bias
    return output

def wide_and_deep_scorer(action_emb, item_emb, item_dim, hidden_dim, norm_layer, dropout_rate ):
    '''
    score = Wide&Deep(item_emb)
    - Deep component:
        - h1 = leaky_relu(item_emb * fc1_weight + fc1_bias)
        - score1 = h1 * fc2_weight + fc2_bias
    - Wide component:
        - score2 = item_emb * weight + bias
    - score = 0.5 * score1 + 0.5 * score2
    
    total_n_parameters 
    = (i_dim * h_dim + h_dim) + (h_dim + 1) + (i_dim + 1)
    = (i_dim + 2) * (h_dim + 1)
    
    @input:
    - action_emb: (B, (i_dim+2)*(h_dim+1))
    - item_emb: (B, L, i_dim) or (1, L, i_dim)
    
    @output:
    - score: (B, L)
    '''
    deep_component_dim = (item_dim+2)*hidden_dim + 1
    fc1_weight = action_emb[:, :item_dim*hidden_dim].view(-1, item_dim, hidden_dim)
    fc1_bias = action_emb[:,item_dim*hidden_dim:(item_dim+1)*hidden_dim].view(-1, 1, hidden_dim)
    fc2_weight = action_emb[:, (item_dim+1)*hidden_dim:(item_dim+2)*hidden_dim].view(-1, 1, hidden_dim)
    fc2_bias = action_emb[:,(item_dim+2)*hidden_dim].view(-1, 1)
    wide_weight = action_emb[:,deep_component_dim:deep_component_dim+item_dim].view(-1,1,item_dim)
    wide_bias = action_emb[:,-1].view(-1, 1)

    h1 = torch.matmul(item_emb, fc1_weight) + fc1_bias
    h1 = norm_layer(F.dropout(h1, dropout_rate))
    deep_score = torch.sum(h1 * fc2_weight, dim = -1) + fc2_bias
    wide_score = torch.sum(wide_weight * item_emb, dim = -1) + wide_bias
    output = deep_score * 0.5 + wide_score * 0.5
    return output

