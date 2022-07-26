from kore_fleets import balanced_agent
from helpers import Board, Player, Observation, Configuration

def agent(obs,config):
    return balanced_agent(obs,config)
