from typing import NamedTuple

from .crossover import crossover
from .dins import dins
from .local_branching import local_branching
from .mutation import mutation_25
from .no_objective import no_objective
from .proximity import proximity
from .rens import rens


class DestroyOperators(NamedTuple):
    Crossover = crossover
    Dins = dins
    Local_Branching = local_branching
    Mutation = mutation_25
    No_Objective = no_objective
    Proximity = proximity
    Rens = rens

