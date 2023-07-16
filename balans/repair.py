from balans.base_state import State


def repair(current: State) -> State:

    # Solve the state with fixed variables to repair and update solution and objective
    current.solve_and_update()

    return current
