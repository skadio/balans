import math
import random
from typing import Tuple, Dict, Any, List

from highspy import Highs

from balans.base_mip import _BaseMIP
from balans.utils import Constants


class _HIGHS(_BaseMIP):

    def __init__(self, instance_path: str, seed: int):
        super().__init__(seed)

        # Create HiGHS model
        self.model = Highs()
        self.model.readModel(instance_path)
        self.model.setOption('random_seed', seed)

        # Set variables
        self.variables = self.model.getCols()

        # Set original objective and flag if we changed from max to min.
        # We always minimize
        self.org_objective_fn = self.model.getObjective()
        self.org_objective_sense = self.model.getObjectiveSense()  # 1 for minimize, -1 for maximize
        self.is_obj_sense_changed = False
        if self.org_objective_sense == -1:
            self.model.setObjective(-1 * self.org_objective_fn, 'minimize')
            self.is_obj_sense_changed = True

        # Set constraints, proximity z, and a flag for objective transformation
        # These are used for incremental solve and undo solve
        self.constraints = []
        self.proximity_z = None
        self.is_obj_transformed = False

    def get_obj_value(self, index_to_val) -> float:
        expr = self.org_objective_fn
        obj_val = 0
        for i in range(len(expr)):
            var = expr[i]
            coeff = self.model.getObjectiveCoeff(i)
            obj_val += coeff * index_to_val[var.index]

        return obj_val

    def extract_indexes(self) -> Tuple[List[Any], List[Any], List[Any]]:
        # Set indexes
        discrete_indexes = []
        binary_indexes = []
        integer_indexes = []

        for var in self.variables:
            # Variable types naming different
            if self.is_discrete(var.type):
                discrete_indexes.append(var.index)
                if self.is_binary(var.type):
                    binary_indexes.append(var.index)
                else:
                    integer_indexes.append(var.index)

        return discrete_indexes, binary_indexes, integer_indexes

    def extract_lp(self, discrete_indexes) -> Tuple[Dict[Any, float], float, List[Any]]:
        lp_index_to_val, lp_obj_val = self.solve_lp_and_undo()
        lp_floating_discrete_indexes = [i for i in discrete_indexes if not math.isclose(lp_index_to_val[i] % 1, 0.0)]
        return lp_index_to_val, lp_obj_val, lp_floating_discrete_indexes

    def fix_vars(self, index_to_val, skip_indexes=None) -> None:
        # If no solution given, do nothing
        if index_to_val is None:
            return

        # Walk the variables
        for var in self.variables:
            # If a value for this variable is not given, do nothing
            if var.index not in index_to_val or index_to_val[var.index] is None:
                continue
            # if skip list given, and the variable is in there, do nothing
            if skip_indexes and var.index in skip_indexes:
                continue

            # Variable has a value, and it's not in the skip set, FIX
            self.constraints.append(self.model.addConstraint(var.index, '==', index_to_val[var.index]))

    def dins(self, index_to_val, dins_set, lp_index_to_val) -> None:
        for var in self.variables:
            index = var.index
            if index in dins_set:
                # Add bounding constraint around initial lp solution
                current_lp_diff = abs(index_to_val[index] - lp_index_to_val[index])
                self.constraints.append(
                    self.model.addConstraint(var.index, '>=', lp_index_to_val[index] - current_lp_diff))
                self.constraints.append(
                    self.model.addConstraint(var.index, '<=', lp_index_to_val[index] + current_lp_diff))
            else:
                # fix the variable
                self.constraints.append(self.model.addConstraint(var.index, '==', index_to_val[index]))

    def local_branching(self, index_to_val, local_branching_size, binary_indexes) -> None:
        zero_binary_vars, one_binary_vars = self.split_binary_vars(self.variables, binary_indexes, index_to_val)
        zero_expr = sum(zero_binary_vars)
        one_expr = sum(1 - var for var in one_binary_vars)
        total_expr = zero_expr + one_expr
        self.constraints.append(self.model.addConstraint(total_expr, '<=', local_branching_size))

    def proximity(self, index_to_val, obj_val, proximity_delta, binary_indexes) -> None:
        # Set the flag so we can undo
        self.is_obj_transformed = True

        zero_binary_vars, one_binary_vars = self.split_binary_vars(self.variables, binary_indexes, index_to_val)
        zero_expr = sum(zero_binary_vars)
        one_expr = sum(1 - var for var in one_binary_vars)

        # add cutoff constraint depending on sense, so that next state is better quality
        self.proximity_z = self.model.addVar(lb=0, name='proximity_z')
        self.constraints.append(self.model.addConstraint(self.model.getObjective(), '<=',
                                                         obj_val * (1 - proximity_delta) + self.proximity_z))

        self.model.setObjective(zero_expr + one_expr + Constants.M * self.proximity_z, 'minimize')

    def rens(self, index_to_val, rens_float_set, lp_index_to_val) -> None:
        for var in self.variables:
            if var.index in rens_float_set:
                self.constraints.append(
                    self.model.addConstraint(var.index, '>=', math.floor(lp_index_to_val[var.index])))
                self.constraints.append(
                    self.model.addConstraint(var.index, '<=', math.ceil(lp_index_to_val[var.index])))
            else:
                self.constraints.append(self.model.addConstraint(var.index, '==', index_to_val[var.index]))

    def random_objective(self) -> None:
        # Set the flag so we can undo
        self.is_obj_transformed = True

        # Get original objective function
        expr = self.org_objective_fn

        # Extract variables and coefficients
        vars = []
        coeffs = []

        for i in range(len(expr)):
            vars.append(expr[i])
            coeffs.append(self.model.getObjectiveCoeff(i))

        num_vars = len(vars)
        num_keep = max(1, int(0.25 * num_vars))  # Ensure at least one variable is kept

        # Randomly select indices to keep
        indices = list(range(num_vars))
        random.shuffle(indices)
        keep_indices = set(indices[:num_keep])

        # Build new objective function with 25% coefficients
        new_obj = []
        for i in range(num_vars):
            if i in keep_indices:
                if self.is_obj_sense_changed:
                    new_obj.append((-coeffs[i], vars[i]))
                else:
                    new_obj.append((coeffs[i], vars[i]))
            else:
                # Coefficient is zero, do not add term
                pass

        # Set the new objective function
        self.model.setObjective(new_obj, 'minimize')

        # Set limits
        self.model.setOption('solution_limit', 1)
        self.model.setOption('heuristics', 0)

    def solve_and_undo(self, time_limit_in_sc=None, solution_limit=None) -> Tuple[Dict[Any, float], float]:
        # Set limits
        if time_limit_in_sc is not None:
            self.model.setOption('time_limit', time_limit_in_sc)
        if solution_limit is not None:
            self.model.setOption('solution_limit', solution_limit)

        # Optimize
        self.model.run()

        # Get solution
        index_to_val, obj_val = self.get_index_to_val_and_objective()

        # Free the model
        self.model.clear()

        # Undo limits
        if time_limit_in_sc is not None:
            self.model.setOption('time_limit', float('inf'))
        if solution_limit is not None:
            self.model.setOption('solution_limit', float('inf'))

        # Remove constraints, and reset
        for ct in self.constraints:
            self.model.removeConstraint(ct)
        self.constraints = []

        # Reset to original objective (random objective and proximity change objective)
        if self.is_obj_transformed:
            self.is_obj_transformed = False

            # Reset back to original minimize objective
            if self.is_obj_sense_changed:
                self.model.setObjective(-1 * self.org_objective_fn, 'minimize')
            else:
                self.model.setObjective(self.org_objective_fn, 'minimize')

            # if proximity_z variable is added, remove it
            if self.proximity_z:  # if proximity_delta > 0
                self.model.removeVar(self.proximity_z)
                self.proximity_z = None

        return index_to_val, obj_val

    def solve_random_and_undo(self, time_limit_in_sc=None) -> Tuple[Dict[Any, float], float]:
        # Set limits
        if time_limit_in_sc is not None:
            self.model.setOption('time_limit', time_limit_in_sc)

        # Set random objective
        self.random_objective()

        # Solve
        self.model.run()

        index_to_val, obj_val = self.get_index_to_val_and_objective()

        # Free the model
        self.model.clear()

        # Restore original objective
        self.is_obj_transformed = False

        # Reset back to original minimize objective
        if self.is_obj_sense_changed:
            self.model.setObjective(-1 * self.org_objective_fn, 'minimize')
        else:
            self.model.setObjective(self.org_objective_fn, 'minimize')

        self.model.setOption('solution_limit', float('inf'))
        if time_limit_in_sc is not None:
            self.model.setOption('time_limit', float('inf'))

        return index_to_val, obj_val

    def solve_lp_and_undo(self) -> Tuple[Dict[Any, float], float]:
        # Solve LP relaxation
        int_vars = []
        bin_vars = []
        variables = self.model.getCols()

        for var in variables:
            if self.is_binary(var.type):
                var.type = 'continuous'
                bin_vars.append(var)
            elif self.is_discrete(var.type):
                var.type = 'continuous'
                int_vars.append(var)

        # Solve
        self.model.runLpRelaxation()
        lp_index_to_val, lp_obj_val = self.get_index_to_val_and_objective()

        # Get back the original model
        self.model.clear()

        # Revert variable types
        for var in int_vars:
            var.type = 'integer'
        for var in bin_vars:
            var.type = 'binary'

        return lp_index_to_val, lp_obj_val

    def get_index_to_val_and_objective(self) -> Tuple[Dict[Any, float], float]:
        if self.model.getNumSolutions() == 0 or self.model.getModelStatus() == 'infeasible':
            return dict(), 9999999
        else:
            index_to_val = {var.index: var.value for var in self.variables}
            obj_value = self.model.getObjectiveValue()
            return index_to_val, obj_value

    @staticmethod
    def is_discrete(var_type) -> bool:
        return var_type in ('binary', 'integer')

    @staticmethod
    def is_binary(var_type) -> bool:
        return var_type == 'binary'

    @staticmethod
    def split_binary_vars(variables, binary_indexes, index_to_val) -> Tuple[List[Any], List[Any]]:
        zero_binary_vars = []
        one_binary_vars = []
        for var in variables:
            if var.index in binary_indexes:
                if math.isclose(index_to_val[var.index], 0.0):
                    zero_binary_vars.append(var)
                else:
                    one_binary_vars.append(var)
        return zero_binary_vars, one_binary_vars
