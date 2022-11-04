# ENV
from project.env.environnement import Environnement

# AGENT
from project.agents.agent import Agent
from project.implemented_agents import agents_map
from project.agents.SARSA import EpsilonGreedyAgent
from project.agents.SARSA_expected import EpsilonGreedyAgent_expected
from project.agents.Qlearning import EpsilonGreedyAgent_Qlearning
from project.agents.random import RandomAgent

# PYTHON
from argparse import ArgumentParser



N_EPISODES = 100

def train(agent : Agent, env : Environnement):

    for episode in range(N_EPISODES):
        print(f"Episode {episode} starts.")
        
        agent.reset()

        state = env.reset()
        done = False
        while not done:
            # Agent takes action
            action = agent.act(state)

            # Action has effect on environment
            print(action)
            next_state, reward, done = env.step(action)
            print(reward)

            # Agent observe the transition and possibly learns
            agent.observe(state, action, reward, next_state, done)
            agent.learn()

            # Render environment for user to see
            env.render()

            # Update state
            state = next_state


if __name__ == "__main__":
    
    # Get args

    parser = ArgumentParser(description="Run a reinforcement learning agent")
    parser.add_argument("--agent", type=str, required=True, help="Agent to run")
    args = parser.parse_args()
    agent_name = args.agent

    # Create the environnement
    env = Environnement.init("3d")

    # Create the agent
    agent = agents_map[agent_name](env)
    # Run the agent
    print(agent)
    train(agent, env)