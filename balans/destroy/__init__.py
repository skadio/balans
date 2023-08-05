from typing import NamedTuple

from .crossover import crossover, crossover2, crossover3
from .dins import dins_50, dins_75, dins_random_50, dins_random_75
from .local_branching import local_branching_50
from .mutation import mutation_25, mutation_50, mutation_75, mutation_binary_50
from .proximity import proximity
from .rens import rens
from .rins import rins, rins_random_50
from .zero_objective import zero_objective


class DestroyOperators(NamedTuple):
    Crossover = crossover
    Crossover2 = crossover2
    Crossover3 = crossover3
    Dins = dins_50
    Dins_Random = dins_random_50
    Local_Branching = local_branching_50
    Mutation = mutation_25
    Mutation2 = mutation_50
    Mutation3 = mutation_75
    Mutation_Binary = mutation_binary_50
    Proximity = proximity
    Rens = rens
    Rins = rins
    Rins_Random = rins_random_50
    Zero_Objective = zero_objective
