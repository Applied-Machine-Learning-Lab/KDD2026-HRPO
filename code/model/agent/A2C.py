import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseRLAgent import BaseRLAgent
    
class A2C(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - episode_batch_size
        - batch_size
        - actor_lr
        - critic_lr
        - actor_decay
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - n_iter
            - train_every_n_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - with_eval
            - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        parser.add_argument('--advantage_bias', type=float, default=0, 
                            help='mitigation factor')
        parser.add_argument('--entropy_coef', type=float, default=0.1, 
                            help='mitigation factor')
        return parser
    
    
    def __init__(self, *input_args):
        '''
        self.gamma
        self.n_iter
        self.check_episode
        self.with_eval
        self.save_path
        self.facade
        self.exploration_scheduler
        '''
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay
        
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        
        
        
        
    def action_before_train(self):
        super().action_before_train()
        self.training_history = {'actor_loss': [], 'critic_loss': [], 'entropy_loss':[], 'advantage':[],
                            'Q': [], 'next_Q': []}
        
            

    def step_train(self):
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        reward = user_feedback['reward'].view(-1)
        
        critic_loss, actor_loss, entropy_loss, advantage = self.get_a2c_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['entropy_loss'][-1], 
                              self.training_history['advantage'][-1])}
    
    def get_a2c_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        
        current_policy_output = self.apply_policy(observation, self.actor)
        current_target_critic_output = self.apply_critic(observation, current_policy_output, self.critic_target)
        V_S = current_target_critic_output['v']
        
        next_policy_output = self.apply_policy(next_observation, self.actor_target)
        target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
        V_S_prime = target_critic_output['v'].detach()
        
        Q_S = reward + self.gamma * (done_mask * V_S_prime)
        advantage = torch.clamp((Q_S - V_S).detach(), -1, 1) # (B,)

        value_loss = F.mse_loss(V_S, Q_S).mean()
        

        if do_critic_update and self.critic_lr > 0:
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        current_policy_output = self.apply_policy(observation, self.actor)
        A = policy_output['action']
        logp = -torch.log(current_policy_output['probs'] + 1e-6) # (B,K)
        actor_loss = torch.mean(torch.sum(logp * (advantage.view(-1,1) + self.advantage_bias), dim = 1))
        entropy_loss = torch.sum(current_policy_output['all_probs'] \
                                  * torch.log(current_policy_output['all_probs']), dim = 1).mean()
        

        if do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            (actor_loss + self.entropy_coef * entropy_loss).backward()
            self.actor_optimizer.step()
            
        return value_loss, actor_loss, entropy_loss, torch.mean(advantage)

    def apply_policy(self, observation, policy_model, epsilon = 0, 
                     do_explore = False, do_softmax = True):
        '''
        @input:
        - observation: input of policy model
        - policy_model
        - epsilon: greedy epsilon, effective only when do_explore == True
        - do_explore: exploration flag, True if adding noise to action
        - do_softmax: output softmax score
        '''
        feed_dict = observation
        is_train = True
        input_dict = {'observation': observation, 
                'candidates': self.env.get_candidate_info(observation), 
                'epsilon': epsilon, 
                'do_explore': do_explore, 
                'is_train': is_train, 
                'batch_wise': False}
        out_dict = policy_model(input_dict)
            
            

        return out_dict
        
    def apply_critic(self, observation, policy_output, critic_model):
        feed_dict = {'state': policy_output['state'],
                'action': policy_output['hyper_action']}
        critic_output = critic_model(feed_dict)
        return critic_output 

    def save(self):
        torch.save(self.critic.state_dict(), self.save_path + "_critic")
        torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
