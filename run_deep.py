# ENV
from project.env.environnement import Environnement

# AGENT
from project.agents.agent import Agent
from project.implemented_agents import agents_map
from project.agents.SARSA import EpsilonGreedyAgent
from project.agents.SARSA_expected import EpsilonGreedyAgent_expected
from project.agents.Qlearning import EpsilonGreedyAgent_Qlearning
from project.agents.random import RandomAgent
from project.agents.deep import DeepLearningAgent, mlp



# PYTHON
from project.agents.agent import Agent
from project.env.environnement import Environnement
import pickle
from copy import deepcopy


batch_size = 20
N_epoch = int(100000/batch_size)

HS_counter = [0,0,0]
                
# for training policy
def train_one_epoch(agent : Agent, env : Environnement,state,render=True):
    
    batch = 0
    
    done = False
    
    agent.batch_reset()
    
    while not done:
        # Agent takes action
        action = agent.act(state)   # on prend l'action avec l'état dans lequel on est

        # Action has effect on environment
        
        current_state = deepcopy(state)
        
        next_state, reward, done = env.step(action)
        
        """
        if next_state.getItems()[0] == state.getItems()[0] and state.getItems()[0].wear == 5:
            HS_counter[0] += 1
        if next_state.getItems()[1] == state.getItems()[1] and state.getItems()[1].wear == 5:
            HS_counter[1] += 1
        if next_state.getItems()[2] == state.getItems()[2] and state.getItems()[2].wear== 5:
            HS_counter[2] += 1
        if next_state.getItems()[0].wear != 5:
            HS_counter[0] = 0
        if next_state.getItems()[1].wear != 5:
            HS_counter[1] = 0
        if next_state.getItems()[2].wear != 5:
            HS_counter[2] = 0
            
        print("HS_counter:",HS_counter)
            
        for x in HS_counter:
            reward -= x
        """
        

        # Agent observe the transition and possibly learns
        agent.observe(current_state, action, reward, next_state, done)
        
        batch += 1
        
        if batch == batch_size:      # agent learns
            batch = 0
            done = True
            
            
        # Render environment for user to see
        if render == True:
            env.render()

        # Update state
        
        print("state:", current_state)
        print("reward:", reward)
        print("action", action)
        print("next state:", next_state)
    
        state = next_state
        

    agent.learn()
    return state
            
     
# Create the environnement
env = Environnement.init("3dadvanced")

# Create the agent
agent = DeepLearningAgent(env,hidden_sizes=[64,32],lr=1e-2,batch_size=batch_size,render=False)
    

def train(N_epoch,render=True):
    agent.reset()
    state = env.reset()   # first obs/state comes from starting distribution 
    for i in range(N_epoch):
        print("Epoch:" +str(i))
        state = train_one_epoch(agent,env,state,render)


train(N_epoch,render=True)

# SAVE AGENT
def save_agent(agent):
    object = agent.logits_net
    filehandler = open("network.pkl", 'wb') 
    pickle.dump(object, filehandler)
  
# LOAD AGENT WITHOUT EPSILON  
def load_agent(filehandler):
    network = pickle.load(filehandler)
    agent_test = DeepLearningAgent(env,hidden_sizes=[64,32],lr=1e-2,batch_size=batch_size,render=False,random=False)
    agent_test.logits_net = network
    return agent_test
    
    
save_agent(agent)
filehandler = open('network.pkl', 'rb') 
agent_trained = load_agent(filehandler)

#TRAIN AGENT WITHOUT EPSILON
#state = env.reset()
#for i in range(100):
#    train_one_epoch(agent_trained,env,state)
