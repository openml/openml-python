class OpenMLDataFeature(object):
    """Data Feature (a.k.a. Attribute) object.

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
    LEGAL_DATA_TYPES = ['nominal', 'numeric', 'string', 'date']

    def __init__(self, index, name, data_type, nominal_values,
                 number_missing_values):
        if type(index) != int:
            raise ValueError('Index is of wrong datatype')
        if data_type not in self.LEGAL_DATA_TYPES:
            raise ValueError('data type should be in %s, found: %s' %
                             (str(self.LEGAL_DATA_TYPES), data_type))
        if nominal_values is not None and type(nominal_values) != list:
            raise ValueError('Nominal_values is of wrong datatype')
        if type(number_missing_values) != int:
            raise ValueError('number_missing_values is of wrong datatype')

        self.index = index
        self.name = str(name)
        self.data_type = str(data_type)
        self.nominal_values = nominal_values
        self.number_missing_values = number_missing_values

    def __str__(self):
        return "[%d - %s (%s)]" % (self.index, self.name, self.data_type)

    def _repr_pretty_(self, pp, cycle):
        pp.text(str(self))
