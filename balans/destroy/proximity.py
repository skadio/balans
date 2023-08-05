import copy
from balans.base_state import _State


def proximity(current: _State, rnd_state) -> _State:
    # Objective function modification
    # Change the objective coefficients of the original
    # problem based on the current solution value.
    # For binary variables,
    # if x_inc =0, update its objective coefficient to 1.
    # if x_inc =1, update its objective coefficient to -1.
    # Drop all other variables by making their coefficient 0.
    # Send the destroy set to base_instance.
    # Note : the required objective operations for proximity search happens in base_instance file.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  is_proximity=True)

