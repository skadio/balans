# MAB_ALNS


## Description
The project is about the implementation of a bandit selector outside of the MIP solver such as "SCIP".  

The library provides the algorithm and various acceptance criteria, operator selection schemes, and stopping criteria. To solve your own MIP problem, you should provide the following:

-A solution state for your problem that implements an objective() function.
-An initial solution.
-One or more destroy and repair operators tailored to your problem.

Here is a minimum we are implementing:

1. LNS Heuristics as operators and we enumerate these operators and assign them as "arms".
2. Efficient bandit algorithm is run on top these heuristics. 
3. Novel rewarding scheme to measure how each operator is performing for a given instance.
3. Training the model with MIPLIP instances.



#Bandit Rewards 

The selected operators are applied to the current solution, resulting in a new candidate solution. This candidate is evaluated by the algorithm, which leads to one of four outcomes:

-The candidate solution is a new global best.

-The candidate solution is better than the current solution, but not a new global best.

-The candidate solution is accepted.

-The candidate solution is rejected.


### DATA REQUIREMENTS
MIPLIB 2017


```

### Output: Objective Function Values 
