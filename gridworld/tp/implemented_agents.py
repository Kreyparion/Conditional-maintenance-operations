from re import M
from tp.agents.random import RandomAgent
from tp.agents.SARSA import EpsilonGreedyAgent
from tp.agents.SARSA_expected import EpsilonGreedyAgent_expected
from tp.agents.Qlearning import EpsilonGreedyAgent_Qlearning


agents_map = {
    "random": RandomAgent,
    "sarsa": EpsilonGreedyAgent,
    "sarsa_expected":EpsilonGreedyAgent_expected,
    "Qlearning":EpsilonGreedyAgent_Qlearning,
}