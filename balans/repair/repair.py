from balans.base_state import _State


def repair(current: _State, rnd_state) -> _State:
    print("\t Repair")

    # Solve the state with fixed variables to repair and update solution and objective
    current.solve_and_update()

    print("\t Repair objective:", current.obj_val)

    return current
