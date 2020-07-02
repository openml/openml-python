# License: BSD 3-Clause


class OpenMLDataFeature(object):
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
    """

    LEGAL_DATA_TYPES = ["nominal", "numeric", "string", "date"]

    def __init__(self, index, name, data_type, nominal_values, number_missing_values):
        if type(index) != int:
            raise ValueError("Index is of wrong datatype")
        if data_type not in self.LEGAL_DATA_TYPES:
            raise ValueError(
                "data type should be in %s, found: %s" % (str(self.LEGAL_DATA_TYPES), data_type)
            )
        if data_type == "nominal":
            if nominal_values is None:
                raise TypeError(
                    "Dataset features require attribute `nominal_values` for nominal "
                    "feature type."
                )
            elif not isinstance(nominal_values, list):
                raise TypeError(
                    "Argument `nominal_values` is of wrong datatype, should be list, "
                    "but is {}".format(type(nominal_values))
                )
        else:
            if nominal_values is not None:
                raise TypeError("Argument `nominal_values` must be None for non-nominal feature.")
        if type(number_missing_values) != int:
            raise ValueError("number_missing_values is of wrong datatype")

        self.index = index
        self.name = str(name)
        self.data_type = str(data_type)
        self.nominal_values = nominal_values
        self.number_missing_values = number_missing_values

    def __repr__(self):
        return "[%d - %s (%s)]" % (self.index, self.name, self.data_type)

    def _repr_pretty_(self, pp, cycle):
        pp.text(str(self))
