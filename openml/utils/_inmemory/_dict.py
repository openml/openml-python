"""Utilities module for serializing and deserializing dicts."""

from __future__ import annotations


def serialize_dict(d, mode="eval", name="d"):
    """Serialize a dict as an executable Python code snippet.

    To deserialize, simply execute the code snippet in a Python environment.

    Command for deserialization:

    * if ``mode == "eval"``, use ``deserialized = exec(code_snippet)``
    * if ``mode == "exec"``, use ``exec(code_snippet)`` and then access the dict

    Parameters
    ----------
    d : dict
        The dictionary to serialize.

    mode : str, "eval" or "exec", default="eval"
        The mode of serialization.

        * If ``"eval"``, the returned code snippet is an expression that evaluates to the dict.
        * If ``"exec"``, the returned code snippet is a series of statements that assign the dict
          to a variable named ``name``.

    name : str, default="d"
        The variable name to assign the dict to.
        Only used if mode is ``"exec"``.

    Returns
    -------
    code_snippet : str
        A string containing the Python code snippet that recreates the dict ``d``,
        assigned to the specified variable name ``name``.

    Example
    -------
    >>> my_dict = {'a': 'apple', 'b': 'banana'}
    >>> serialized_dict = serialize_dict(my_dict, name="my_dict")
    >>> deserialized_dict = eval(serialized_dict)
    >>> assert deserialized_dict == my_dict
    """

    def dq(s):
        # Escape backslashes and double quotes for valid Python strings
        return s.replace("\\", "\\\\").replace('"', '\\"')

    if mode == "eval":
        lines = ["{"]
    else:  # mode == "exec"
        lines = [f"{name} = {{"]
    for k, v in d.items():
        lines.append(f'    "{dq(k)}": "{dq(v)}",')
    lines.append("}")
    return "\n".join(lines)
