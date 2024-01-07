# License: BSD 3-Clause
from __future__ import annotations


class OpenMLDataFeature:
    """
    Data Feature (a.k.a. Attribute) object.

    Parameters
    ----------
    index : int
        The index of this feature
    name : str
        Name of the feature
    data_type : str
        can be nominal, numeric, string, date (corresponds to arff)
    nominal_values : list(str)
        list of the possible values, in case of nominal attribute
    number_missing_values : int
        Number of rows that have a missing value for this feature.
    """

    LEGAL_DATA_TYPES = ["nominal", "numeric", "string", "date"]

    def __init__(
        self,
        index: int,
        name: str,
        data_type: str,
        nominal_values: list[str],
        number_missing_values: int,
    ):
        if not isinstance(index, int):
            raise TypeError(f"Index must be `int` but is {type(index)}")
        if data_type not in self.LEGAL_DATA_TYPES:
            raise ValueError(
                f"data type should be in {self.LEGAL_DATA_TYPES!s}, found: {data_type}",
            )
        if data_type == "nominal":
            if nominal_values is None:
                raise TypeError(
                    "Dataset features require attribute `nominal_values` for nominal "
                    "feature type.",
                )
            elif not isinstance(nominal_values, list):
                raise TypeError(
                    "Argument `nominal_values` is of wrong datatype, should be list, "
                    f"but is {type(nominal_values)}",
                )
        else:
            if nominal_values is not None:
                raise TypeError("Argument `nominal_values` must be None for non-nominal feature.")
        if not isinstance(number_missing_values, int):
            msg = f"number_missing_values must be int but is {type(number_missing_values)}"
            raise TypeError(msg)

        self.index = index
        self.name = str(name)
        self.data_type = str(data_type)
        self.nominal_values = nominal_values
        self.number_missing_values = number_missing_values

    def __repr__(self):
        return "[%d - %s (%s)]" % (self.index, self.name, self.data_type)

    def __eq__(self, other):
        return isinstance(other, OpenMLDataFeature) and self.__dict__ == other.__dict__

    def _repr_pretty_(self, pp, cycle):
        pp.text(str(self))
