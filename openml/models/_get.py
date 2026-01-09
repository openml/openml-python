"""Model retrieval utility."""

from __future__ import annotations

from functools import lru_cache


def get(id: str):
    """Retrieve model object with unique identifier.

    Parameter
    ---------
    id : str
        unique identifier of object to retrieve

    Returns
    -------
    class
        retrieved object

    Raises
    ------
    ModuleNotFoundError
        if dependencies of object to retrieve are not satisfied
    """
    id_lookup = _id_lookup()
    obj = id_lookup.get(id)
    if obj is None:
        raise ValueError(f"Error in openml.get, object with package id {id} " "does not exist.")
    return obj(id).materialize()


# todo: need to generalize this later to more types
# currently intentionally retrieves only classifiers
# todo: replace this, optionally, by database backend
def _id_lookup(obj_type=None):
    return _id_lookup_cached(obj_type=obj_type).copy()


@lru_cache
def _id_lookup_cached(obj_type=None):
    all_objs = _all_objects(obj_type=obj_type)

    lookup_dict = {}
    for obj in all_objs:
        obj_index = obj.get_class_tag("pkg_id")
        if obj_index != "__multiple":
            lookup_dict[obj_index] = obj
        else:
            obj_all_ids = obj.contained_ids()
            lookup_dict.update({obj_id: obj for obj_id in obj_all_ids})

    return lookup_dict


@lru_cache
def _all_objects(obj_type=None):
    from skbase.lookup import all_objects

    from openml.models.apis._classifier import _ModelPkgClassifier

    return all_objects(object_types=_ModelPkgClassifier, package_name="openml", return_names=False)
