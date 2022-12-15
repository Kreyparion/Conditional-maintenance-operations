# ENV
from project.env.environnement import Environnement

# AGENT
from project.agents.agent import Agent
from project.implemented_agents import agents_map
# PYTHON
from argparse import ArgumentParser
from copy import deepcopy

N_EPISODES = 100000

def train(agent : Agent, env : Environnement):

    for episode in range(N_EPISODES):
        print(f"Episode {episode} starts.")

        state = env.reset()
        done = False
        while not done:
            # Agent takes action
            action = agent.act(state)

            # Action has effect on environment
            next_state, reward, done = env.step(action)

            # Agent observe the transition and possibly learns
            agent.observe(state, action, reward, next_state, done)
            agent.learn()

            # Render environment for user to see
            env.render()

            # Update state
            state = deepcopy(next_state)


if __name__ == "__main__":
    
    # Get args

    parser = ArgumentParser(description="Run a reinforcement learning agent")
    parser.add_argument("--agent", type=str, required=True, help="Agent to run")
    args = parser.parse_args()
    agent_name = args.agent

    # Create the environnement
    env = Environnement.init("3dadvanced")
    print(env.getEveryState())
    # Create the agent
    agent = agents_map[agent_name](env)
    # Run the agent
    print(agent)
    train(agent, env)
