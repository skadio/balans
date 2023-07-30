import copy
from balans.base_state import _State


def rens(current: _State, rnd_state) -> _State:
    #  Take the LP relaxed solution of the original MIP.
    #  For discrete variables, choose the non-integral ones
    #  and store them in the rens_float_set to be bounded list.
    #  Send the destroy set (None for the rens case) and
    #  rens_float_set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    # Discrete variables, where the lp relaxation is not integral
    rens_float_set = [i for i in discrete_indexes if not lp_index_to_val[i].is_integer()]

    print("\t Float set:", rens_float_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  rens_float_set=rens_float_set)
