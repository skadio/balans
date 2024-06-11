import copy

from balans.base_state import _State


def _rens(current: _State, rnd_state, delta) -> _State:
    #  Take the LP relaxed solution of the original MIP.
    #  For discrete variables, choose the non-integral ones
    #  and store them in the rens_float_set to be bounded list.
    #  Send the destroy set (None for the rens case) and
    #  rens_float_set to base_instance.

    print("*** Operator: ", "RENS")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    # list of discrete indexes that are floating point in LP
    floating_discrete_indexes = current.instance.lp_floating_discrete_indexes

    # Randomization STEP
    size = int(delta * len(floating_discrete_indexes))
    floating_discrete_indexes = set(rnd_state.choice(floating_discrete_indexes, size))

    # Else potentially change it
    rens_float_set = set([i for i in discrete_indexes if i in floating_discrete_indexes])

    print("\t Float set:", rens_float_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  rens_float_set=rens_float_set)

def rens_50(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.50)