from re import M
from tp.agents.random import RandomAgent
from tp.agents.SARSA import TD_EGreedyAgent

agents_map = {
    "random": RandomAgent,
    "td": TD_EGreedyAgent,
}