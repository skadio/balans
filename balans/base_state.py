from typing import Any, Dict

from balans.base_instance import _Instance
import pyscipopt as scip

from copy import deepcopy


class _State:
    """
    State of an instance with its solution and objective
    First, operators sets the solve settings of the state object
    Then, state calls instance.solve() with settings dictated by the operators
    """

    def __init__(self,
                 instance: _Instance,
                 index_to_val: Dict[Any, float],
                 obj_val: float,
                 destroy_set=None,
                 dins_set=None,
                 rens_float_set=None,
                 has_random_obj=False,
                 previous_index_to_val=None,
                 local_branching_size=0,
                 is_proximity=None):

        # State holds an instance and its solution and objective
        # Instance holds features and the solve() logic dictated by operators
        self.instance = instance
        self.index_to_val = index_to_val  # index defined by SCIP var.getIndex()
        self.obj_val = obj_val

        # State receives the solve settings from the operators
        # and passes to instance.solve() and updates its solution and objective
        self.destroy_set = destroy_set
        self.dins_set = dins_set
        self.rens_float_set = rens_float_set
        self.has_random_obj = has_random_obj
        self.previous_index_to_val = previous_index_to_val
        self.local_branching_size = local_branching_size
        self.is_proximity = is_proximity

    def __deepcopy__(self, memo):
        return deepcopy_with_sharing(self, shared_attribute_names=["instance"], memo=memo)

    def solution(self):
        return self.index_to_val

    def objective(self):
        return self.obj_val

    def reset_solve_settings(self):
        self.destroy_set = None
        self.dins_set = None
        self.rens_float_set = None
        self.has_random_obj = False
        self.local_branching_size = 0
        self.is_proximity = False

    def solve_and_update(self):
        # Solve the current state with the destroyed variables and update
        self.index_to_val, self.obj_val = self.instance.solve(index_to_val=self.index_to_val,
                                                              obj_val=self.obj_val,
                                                              destroy_set=self.destroy_set,
                                                              dins_set=self.dins_set,
                                                              rens_float_set=self.rens_float_set,
                                                              has_random_obj=self.has_random_obj,
                                                              local_branching_size=self.local_branching_size,
                                                              is_proximity=self.is_proximity)


def deepcopy_with_sharing(obj, shared_attribute_names, memo=None):
    '''
    Deepcopy an object, except for a given list of attributes, which should
    be shared between the original object and its copy.

    obj is some object
    shared_attribute_names: A list of strings identifying the attributes that
        should be shared between the original and its copy.
    memo is the dictionary passed into __deepcopy__.  Ignore this argument if
        not calling from within __deepcopy__.
    '''
    assert isinstance(shared_attribute_names, (list, tuple))
    shared_attributes = {k: getattr(obj, k) for k in shared_attribute_names}

    deepcopy_method = None

    if hasattr(obj, '__deepcopy__'):
        # Do hack to prevent infinite recursion in call to deepcopy
        deepcopy_method = obj.__deepcopy__
        obj.__deepcopy__ = None

    for attr in shared_attribute_names:
        del obj.__dict__[attr]

    clone = deepcopy(obj)

    for attr, val in shared_attributes.items():
        setattr(obj, attr, val)
        setattr(clone, attr, val)

    if hasattr(obj, '__deepcopy__'):
        # Undo hack
        obj.__deepcopy__ = deepcopy_method
        del clone.__deepcopy__

    return clone
