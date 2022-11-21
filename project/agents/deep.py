import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torchtyping import TensorType
from project.agents.agent import Agent
from project.env.environnement import Environnement
from project.env.states import State
from project.env.actions import Action
import random



# Multi Layers Perceptron
# en fonction de l'observation, il nous donne la distribution de proba de prendre une action
class mlp(nn.Module):
    
    def __init__(self,sizes, activation=nn.ReLU, output_activation=nn.Sigmoid):
        super(mlp, self).__init__() 
        self.sizes = sizes   # tailles des couches de neurones successives du NN
        self.activation=activation   #fonction d'activation du NN
        self.output_activation = output_activation
    
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes)-1):   #len(sizes) = nombre de layers du NN
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.seq = nn.Sequential(*layers) # * indique qu'on prend tous les paramètres de layers 
        
    def forward(self,input):
        output = self.seq(input)
        return output



class DeepLearningAgent(Agent):
    
    def __init__(self, env : Environnement, hidden_sizes=[32], lr=1e-2, batch_size=10,render=False, **kwargs):
    
        self.env = env
        self.states = env.getEveryState()
        self.possible_actions = self.env.getPossibleActions(self.states[0])
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.render = render
        self.n_batch = 0
        self.batch_size = batch_size

        self.obs_dim = len(self.states)   # dimension du vecteur d'entrée = nombre d'états possibles
        # vecteur d'entrée = vecteur avec un 1 quand l'état est celui dans lequel on est  
        
        self.n_acts = len(self.possible_actions)           # longueur du vecteur de sortie = nb d'actions possibles
        # sortie = distribution de probabilités pour l'ensemble des actions possibles

        # Core of policy network
        # make function to compute action distribution
        self.logits_net = mlp(sizes=[self.obs_dim] + self.hidden_sizes + [self.n_acts])
        
        # make optimizer
        self.optimizer = Adam(self.logits_net.parameters(), lr=self.lr)
        # on va entrainer le réseau logits_net pour prendre les décisions des actions à prendre
         
        
        # make some empty lists/tensors for logging.
        self.batch_obs = torch.zeros(size=(batch_size,self.obs_dim))       # for observations
        self.batch_acts = torch.zeros(size=(batch_size,self.n_acts))         # for actions
        self.batch_weights = []      # for R(tau) weighting in policy gradient

        # un batch est composé d'épisode
        # une epoch est composée de plusieurs batch donc de plusieurs épisodes
    
        self.batch_rewards = []            # liste des rewards sur un batch

    # collect experience by acting in the environment with current policy
       
    
    def get_policy(self,state:TensorType)-> torch.Tensor:
        logits = self.logits_net(state)
        return logits
    # on récupère une distribution de probabilité sur les actions à prendre

    
    def compute_loss(self,state:TensorType,action:TensorType, weights:TensorType) -> TensorType:
        logp = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            logp[i] = self.get_policy(state[i]) @ action[i]
        return -(weights * logp).mean()

    # on regarde tous les états dans lesquels on a été et toutes les actions qu'on a prise
    # Pour chaque action prise sur un batch, on lui attribue un poids stocké dans weigths 
    # les poids sont en fait tous égaux pour les actions prises pendant un même batch et ils sont égaux à la reward obtenue à la fin du batch
    # la loss est la moyenne de la reward qu'on a recupérée pour avoir fait chaque action pondérée par la proba de prendre chaque action
    
    def state_to_vector(self,state:dict) -> torch.Tensor:
        vector = torch.zeros(self.obs_dim)
        state_idx = self.states.index(state)
        vector[state_idx] = 1
        return vector
    
    def action_to_vector(self,action:dict) -> torch.Tensor:
        vector = torch.zeros(self.n_acts)
        action_idx = self.possible_actions.index(action)
        vector[action_idx] = 1
        return vector

    
    def act(self, state: State) -> Action:
        state_vector = self.state_to_vector(state)
        if random.random() < 0.1:
            action_idx = random.randint(0,self.n_acts-1)
            print("random action")
        else:
            action_idx = torch.argmax(self.get_policy(state_vector))
        return self.possible_actions[action_idx]

    def observe(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        """Observe the transition and stock in memory"""
        # save state, action, reward
        
        self.batch_obs[self.n_batch] = self.state_to_vector(state)    # on stock l'état dans lequel on était
        self.batch_acts[self.n_batch] = self.action_to_vector(action)    # on stock l'action qu'on vient de faire
        self.batch_rewards.append(reward)      # on stock la reward qu'on vient de récupérer
        self.done = done
        self.n_batch += 1
        

    def learn(self):
        """Learn using the memory"""

        
        self.batch_weights = torch.as_tensor(self.batch_rewards, dtype=torch.float32)
        total_reward = sum(self.batch_rewards)  
        total = torch.ones_like(self.batch_weights)
        self.batch_weights = total*total_reward
        # batch_weigths est un vecteur de taille batch_size et de valeurs la reward totale d'un batch
        
        
        # take a single policy gradient update step
        self.optimizer.zero_grad()
        
    
        batch_loss = self.compute_loss(state=self.batch_obs,    # loss sur un batch
                                action=self.batch_acts,
                                weights=self.batch_weights
                                )    
        batch_loss.backward()
        self.optimizer.step()
        
        print('loss: %.3f \t episode reward: %.3f \t episode len: %.3f'%
            (batch_loss, np.mean(self.batch_rewards), self.batch_size))
    
        # reset episode-specific variables
        self.done = False
        
        
        
    def reset(self):
        self.logits_net = mlp(sizes=[self.obs_dim] + self.hidden_sizes + [self.n_acts])
        self.optimizer = Adam(self.logits_net.parameters(), lr=self.lr)
        
    def batch_reset(self):
        self.batch_obs = torch.zeros(size=(self.batch_size,self.obs_dim))
        self.batch_acts = torch.zeros(size=(self.batch_size,self.n_acts))   
        self.batch_weights = []     
        self.batch_rewards = []           
        self.n_batch = 0 