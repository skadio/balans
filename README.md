# MAB_ALNS - Mutation Operator


## Description
The project is about the implementation of operators and a bandit selector outside of the MIP solver such as "SCIP".  

The library provides the operators and the algorithm in which operator selection schemes will be implemented.. To solve your own MIP problem, you should provide the following:

-MIP instance
-An initial solution.


-Note: LP relaxation and problem class will be implemented automatically inside the library.

Here is a minimum we are implementing:
MUTATION OPERATOR
1. Reading a MIP instance from MIPLIP library.
2. Obtaining a MIP instance class and lp relaxed solution. 
3. Defining Mutation Operator to get the next solution.


Here is a minimum we will be implementing:

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
