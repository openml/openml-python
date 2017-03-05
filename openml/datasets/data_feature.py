
class OpenMLDataFeature(object):
    """Data Feature (a.k.a. Attribute) object.

       Parameters
       ----------
       index : int
            The index of this feature
        name : string
            Name of the feature
        data_type : string
            can be nominal, numeric, string, date (corresponds to arff)
        nominal_values : list(str)
            list of the possible values, in case of nominal attribute
        number_missing_values : int
       """
    LEGAL_DATA_TYPES = ['nominal', 'numeric', 'string', 'date']

    def __init__(self, index, name, data_type, nominal_values, number_missing_values):
        assert type(index) is int, "Index is of wrong datatype"
        assert type(name) is str, "Name is of wrong datatype"
        assert type(data_type) is str, "Data_type is of wrong datatype"
        assert data_type in self.LEGAL_DATA_TYPES, "data type should be in %s" %str(self.LEGAL_DATA_TYPES)
        if nominal_values is not None:
            assert type(nominal_values) is list, "Nominal_values is of wrong datatype"
        assert type(number_missing_values) is int, "number_missing_values is of wrong datatype"

        self.index = index
        self.name = name
        self.data_type = data_type
        self.nominal_values = nominal_values
        self.number_missing_values = number_missing_values

    def __str__(self):
        return "[%d - %s (%s)]" %(self.index, self.name, self.data_type)