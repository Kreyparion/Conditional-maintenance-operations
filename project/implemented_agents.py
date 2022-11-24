from project.agents.random import RandomAgent
from project.agents.SARSA import EpsilonGreedyAgent
from project.agents.SARSA_expected import EpsilonGreedyAgent_expected
from project.agents.Qlearning import EpsilonGreedyAgent_Qlearning
from project.agents.fixed_agents import FixedAgent


agents_map = {
    "random": RandomAgent,
    "sarsa":EpsilonGreedyAgent,
    "sarsa_expected":EpsilonGreedyAgent_expected,
    "Qlearning":EpsilonGreedyAgent_Qlearning,
    "fixed":FixedAgent,
}