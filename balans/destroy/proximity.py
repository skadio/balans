import copy

from balans.base_state import _State


def proximity(current: _State, rnd_state, delta=0.005) -> _State:
    # Objective function modification
    # Change the objective coefficients of the original
    # problem based on the current solution value.
    # For binary variables,
    # if x_inc =0, update its objective coefficient to 1.
    # if x_inc =1, update its objective coefficient to -1.
    # Drop all other variables by making their coefficient 0.
    # Send the destroy set to base_instance.
    # Note : the required objective operations for proximity search happens in base_instance file.

    print("*** Operator: ", "PROXIMITY SEARCH")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    next_state.proximity_delta = delta

    return next_state


def proximity_05(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.005)


def proximity_10(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.010)


def proximity_15(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.015)


def proximity_20(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.020)


def proximity_25(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.025)


def proximity_30(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.030)


def proximity_35(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.035)


def proximity_40(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.040)


def proximity_45(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.045)


def proximity_50(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.050)


def proximity_55(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.055)


def proximity_60(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.060)


def proximity_65(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.065)


def proximity_70(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.070)


def proximity_75(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.075)


def proximity_80(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.080)


def proximity_85(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.085)


def proximity_90(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.090)


def proximity_95(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.095)
