import copy

from balans.base_state import _State


def zero_objective(current: _State, rnd_state) -> _State:
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  is_zero_obj=True)
