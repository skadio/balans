import os

import numpy as np


def generate(number_of_items, number_of_knapsacks, filename, random,
             min_range=10, max_range=20, scheme='weakly correlated'):
    """
    Generate a Multiple Knapsack problem following a scheme among those found in section 2.1. of
        Fukunaga, Alex S. (2011). A branch-and-bound algorithm for hard multiple knapsack problems.
        Annals of Operations Research (184) 97-119.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    number_of_items : int
        The number of items.
    number_of_knapsacks : int
        The number of knapsacks.
    filename : str
        Path to the file to save.
    random : numpy.random.RandomState
        A random number generator.
    min_range : int, optional
        The lower range from which to sample the item weights. Default 10.
    max_range : int, optional
        The upper range from which to sample the item weights. Default 20.
    scheme : str, optional
        One of 'uncorrelated', 'weakly correlated', 'strongly corelated', 'subset-sum'. Default 'weakly correlated'.
    """
    weights = random.randint(min_range, max_range, number_of_items)

    if scheme == 'uncorrelated':
        profits = random.randint(min_range, max_range, number_of_items)

    elif scheme == 'weakly correlated':
        profits = np.apply_along_axis(
            lambda x: random.randint(x[0], x[1]),
            axis=0,
            arr=np.vstack([
                np.maximum(weights - (max_range - min_range), 1),
                weights + (max_range - min_range)]))

    elif scheme == 'strongly correlated':
        profits = weights + (max_range - min_range) / 10

    elif scheme == 'subset-sum':
        profits = weights

    else:
        raise NotImplementedError

    capacities = np.zeros(number_of_knapsacks, dtype=int)
    capacities[:-1] = random.randint(0.4 * weights.sum() // number_of_knapsacks,
                                     0.6 * weights.sum() // number_of_knapsacks,
                                     number_of_knapsacks - 1)
    capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        for knapsack in range(number_of_knapsacks):
            for item in range(number_of_items):
                file.write(f" +{profits[item]} x{item + number_of_items * knapsack + 1}")

        file.write("\n\nsubject to\n")
        for knapsack in range(number_of_knapsacks):
            variables = "".join([f" +{weights[item]} x{item + number_of_items * knapsack + 1}"
                                 for item in range(number_of_items)])
            file.write(f"C{knapsack + 1}:" + variables + f" <= {capacities[knapsack]}\n")

        for item in range(number_of_items):
            variables = "".join([f" +1 x{item + number_of_items * knapsack + 1}"
                                 for knapsack in range(number_of_knapsacks)])
            file.write(f"C{number_of_knapsacks + item + 1}:" + variables + " <= 1\n")

        file.write("\nbinary\n")
        for knapsack in range(number_of_knapsacks):
            for item in range(number_of_items):
                file.write(f" x{item + number_of_items * knapsack + 1}")


def main():
    # 1. Create a folder called 'sc'
    if not os.path.exists('../mk/'):
        os.mkdir('../mk/')

    for i in range(1, 101):
        filename = os.path.join("../mk/", f'mk_{i}.lp')
        generate(400, 40, filename, np.random.RandomState(i + 51))


if __name__ == '__main__':
    main()
