from balans.base_state import State


def repair(current: State, rnd_state) -> State:
    print("\t Repair")

    # Solve the state with fixed variables to repair and update solution and objective
    current.solve_and_update()

    print("\t Repair objective:", current.obj_val)

    return current
