import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.general import BaseModel
from model.components import DNN
from model.policy.OneStagePolicy import OneStagePolicy

class PointwiseRanker(OneStagePolicy):
    '''
    GFlowNet with Detailed Balance for listwise recommendation
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - ptranker_state2z_hidden_dims
        - ptranker_pos_offset
        - ptranker_neg_offset
        - from OneStagePolicy:
            - from BackboneUserEncoder:
                - user_latent_dim
                - item_latent_dim
                - transformer_enc_dim
                - transformer_n_head
                - transformer_d_forward
                - transformer_n_layer
                - state_hidden_dims
                - dropout_rate
                - from BaseModel:
                    - model_path
                    - loss
                    - l2_coef
        '''
        parser = OneStagePolicy.parse_model_args(parser) 
        parser.add_argument('--ptranker_state2z_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of state_slate encoding layers')
        parser.add_argument('--ptranker_pos_offset', type=float, default=0.8, 
                            help='smooth offset of positive prob')
        parser.add_argument('--ptranker_neg_offset', type=float, default=0.1, 
                            help='smooth offset of negative prob')
        
        return parser
        
    def __init__(self, *input_args):
        args, env = input_args
        self.ptranker_state2z_hidden_dims = args.ptranker_state2z_hidden_dims
        self.ptranker_pos_offset = args.ptranker_pos_offset
        self.ptranker_neg_offset = args.ptranker_neg_offset
        super().__init__(*input_args)
        self.display_name = "PointwiseRanker"
        
    def to(self, device):
        new_self = super(PointwiseRanker, self).to(device)
        return new_self

    def _define_params(self, args, reader_stats):
        super()._define_params(args, reader_stats)
        self.state2z = DNN(self.state_dim, args.ptranker_state2z_hidden_dims, self.enc_dim, 
                           dropout_rate = args.state_dropout_rate, do_batch_norm = True)
        self.state2zNorm = nn.LayerNorm(self.enc_dim)

    def forward(self, feed_dict: dict, return_prob = True) -> dict:
        return self.get_forward(feed_dict)
    
    def generate_action(self, state_dict, feed_dict):
        user_state = state_dict['state']
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        action_slate = feed_dict['action'] # (B, K)
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        epsilon = feed_dict['epsilon']
        batch_wise = feed_dict['batch_wise']
        '''
        @input:
        - user_state: (B, state_dim) 
        - feed_dict: same as OneStagePolicy.get_forward@feed_dict
        @output:
        - out_dict: {'logP': (B, K), 
                     'logF': (B,),
                     'action': (B, K), 
                     'reg': scalar}
        '''
        B = user_state.shape[0]
        do_uniform = np.random.random() < epsilon
            
        candidate_item_enc, reg = self.user_encoder.get_item_encoding(candidates['item_id'], 
                                                       {k[3:]: v for k,v in candidates.items() if k != 'item_id'}, 
                                                                     B if batch_wise else 1)
        
        Z = self.state2z(user_state)
        Z = self.state2zNorm(Z)
        score = torch.sum(Z.view(B,1,self.enc_dim) * candidate_item_enc, dim = -1) #/ self.enc_dim
        prob = torch.softmax(score, dim = 1)
        
        if is_train or torch.is_tensor(action_slate):
            indices = action_slate
        else:
            if do_explore:
                if do_uniform:
                    indices = Categorical(torch.ones_like(prob)).sample((self.slate_size,)).transpose(0,1)
                else:
                    indices = Categorical(prob).sample((self.slate_size,)).transpose(0,1)
            else: 
                _, indices = torch.topk(prob, k = self.slate_size, dim = 1)
            indices = indices.view(-1,self.slate_size).detach()
        selected_P = torch.gather(prob,1,indices)
        slate_action = indices
                
        reg = self.get_regularization(self.state2z)

        out_dict = {'prob': selected_P, 
                    'all_prob': prob,
                    'action': slate_action, 
                    'indices': indices,
                    'reg': reg}
        
        return out_dict
    
    def get_loss(self, feed_dict, out_dict):
        '''
        Reward-based pointwise ranking loss
        * - Ylog(P) - (1-Y)log(1-P)
        * Y = sum(w[i] * r[i]) # the weighted sum of user responses
        
        @input:
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'action': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K*n_feedback),
            'immediate_response_weight: (n_feedback, ),
            'reward': (B,)}
        @output
        - loss
        '''
        B = out_dict['prob'].shape[0]
        log_P = torch.log(out_dict['prob'] + self.ptranker_pos_offset)
        log_neg_P = torch.log(1 - out_dict['prob'] + self.ptranker_neg_offset)
        weighted_response = out_dict['immediate_response'].view(B,self.slate_size,-1) \
                                * out_dict['immediate_response_weight'].view(1,1,-1)
        Y = torch.mean(weighted_response, dim = 2)
        R_loss = - torch.mean(Y * log_P + (1-Y) * log_neg_P)
        loss = R_loss + self.l2_coef * out_dict['reg']
        
        return {'loss': loss, 'R_loss': R_loss, 'P': torch.mean(out_dict['prob'])}

    def get_loss_observation(self):
        return ['loss', 'R_loss', 'P']
        
        
        