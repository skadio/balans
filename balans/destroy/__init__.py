from typing import NamedTuple

from .crossover import crossover, crossover2,crossover3
from .dins import dins_50, dins_75
from .local_branching import local_branching_25, local_branching_50, local_branching_75
from .mutation import mutation_25, mutation_50, mutation_75, mutation_100
from .zero_objective import zero_objective
from .proximity import proximity
from .rens import rens
from .rins import rins


class DestroyOperators(NamedTuple):
    Crossover = crossover
    Crossover2 = crossover2
    Crossover3 = crossover3
    Dins = dins_50
    Dins2 = dins_75
    Local_Branching = local_branching_25
    Local_Branching2 = local_branching_50
    Local_Branching3 = local_branching_75
    Mutation = mutation_25
    Mutation2 = mutation_50
    Mutation3 = mutation_75
    Mutation4 = mutation_100
    Proximity = proximity
    Rens = rens
    Rins = rins
    Zero_Objective = zero_objective
