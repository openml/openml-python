import gzip
import io
import logging
import os
import pickle
from collections import OrderedDict

import arff
import numpy as np
import pandas as pd
import scipy.sparse
import xmltodict
from warnings import warn

import openml._api_calls
from .data_feature import OpenMLDataFeature
from ..exceptions import PyOpenMLError
from ..utils import _tag_entity


logger = logging.getLogger(__name__)


class OpenMLDataset(object):
    """Dataset object.

    Allows fetching and uploading datasets to OpenML.

    Parameters
    ----------
    name : str
        Name of the dataset.
    description : str
        Description of the dataset.
    format : str
        Format of the dataset which can be either 'arff' or 'sparse_arff'.
    dataset_id : int, optional
        Id autogenerated by the server.
    version : int, optional
        Version of this dataset. '1' for original version.
        Auto-incremented by server.
    creator : str, optional
        The person who created the dataset.
    contributor : str, optional
        People who contributed to the current version of the dataset.
    collection_date : str, optional
        The date the data was originally collected, given by the uploader.
    upload_date : str, optional
        The date-time when the dataset was uploaded, generated by server.
    language : str, optional
        Language in which the data is represented.
        Starts with 1 upper case letter, rest lower case, e.g. 'English'.
    licence : str, optional
        License of the data.
    url : str, optional
        Valid URL, points to actual data file.
        The file can be on the OpenML server or another dataset repository.
    default_target_attribute : str, optional
        The default target attribute, if it exists.
        Can have multiple values, comma separated.
    row_id_attribute : str, optional
        The attribute that represents the row-id column,
        if present in the dataset.
    ignore_attribute : str | list, optional
        Attributes that should be excluded in modelling,
        such as identifiers and indexes.
    version_label : str, optional
        Version label provided by user.
        Can be a date, hash, or some other type of id.
    citation : str, optional
        Reference(s) that should be cited when building on this data.
    tag : str, optional
        Tags, describing the algorithms.
    visibility : str, optional
        Who can see the dataset.
        Typical values: 'Everyone','All my friends','Only me'.
        Can also be any of the user's circles.
    original_data_url : str, optional
        For derived data, the url to the original dataset.
    paper_url : str, optional
        Link to a paper describing the dataset.
    update_comment : str, optional
        An explanation for when the dataset is uploaded.
    status : str, optional
        Whether the dataset is active.
    md5_checksum : str, optional
        MD5 checksum to check if the dataset is downloaded without corruption.
    data_file : str, optional
        Path to where the dataset is located.
    features : dict, optional
        A dictionary of dataset features,
        which maps a feature index to a OpenMLDataFeature.
    qualities : dict, optional
        A dictionary of dataset qualities,
        which maps a quality name to a quality value.
    dataset: string, optional
        Serialized arff dataset string.
    """
    def __init__(self, name, description, format=None,
                 data_format='arff', dataset_id=None, version=None,
                 creator=None, contributor=None, collection_date=None,
                 upload_date=None, language=None, licence=None,
                 url=None, default_target_attribute=None,
                 row_id_attribute=None, ignore_attribute=None,
                 version_label=None, citation=None, tag=None,
                 visibility=None, original_data_url=None,
                 paper_url=None, update_comment=None,
                 md5_checksum=None, data_file=None, features=None,
                 qualities=None, dataset=None):

        # TODO add function to check if the name is casual_string128
        # Attributes received by querying the RESTful API
        self.dataset_id = int(dataset_id) if dataset_id is not None else None
        self.name = name
        self.version = int(version) if version is not None else None
        self.description = description
        if format is None:
            self.format = data_format
        else:
            warn("The format parameter in the init will be deprecated "
                 "in the future."
                 "Please use data_format instead", DeprecationWarning)
            self.format = format
        self.creator = creator
        self.contributor = contributor
        self.collection_date = collection_date
        self.upload_date = upload_date
        self.language = language
        self.licence = licence
        self.url = url
        self.default_target_attribute = default_target_attribute
        self.row_id_attribute = row_id_attribute
        if isinstance(ignore_attribute, str):
            self.ignore_attributes = [ignore_attribute]
        elif isinstance(ignore_attribute, list) or ignore_attribute is None:
            self.ignore_attributes = ignore_attribute
        else:
            raise ValueError('Wrong data type for ignore_attribute. '
                             'Should be list.')
        self.version_label = version_label
        self.citation = citation
        self.tag = tag
        self.visibility = visibility
        self.original_data_url = original_data_url
        self.paper_url = paper_url
        self.update_comment = update_comment
        self.md5_checksum = md5_checksum
        self.data_file = data_file
        self.features = None
        self.qualities = None
        self._dataset = dataset

        if features is not None:
            self.features = {}
            # todo add nominal values (currently not in database)
            for idx, xmlfeature in enumerate(features['oml:feature']):
                nr_missing = xmlfeature.get('oml:number_of_missing_values', 0)
                feature = OpenMLDataFeature(int(xmlfeature['oml:index']),
                                            xmlfeature['oml:name'],
                                            xmlfeature['oml:data_type'],
                                            xmlfeature.get('oml:nominal_value'),
                                            int(nr_missing))
                if idx != feature.index:
                    raise ValueError('Data features not provided '
                                     'in right order')
                self.features[feature.index] = feature

        self.qualities = _check_qualities(qualities)

        if data_file is not None:
            self.data_pickle_file = self._data_arff_to_pickle(data_file)
        else:
            self.data_pickle_file = None

    def _data_arff_to_pickle(self, data_file):
        data_pickle_file = data_file.replace('.arff', '.pkl.py3')
        if os.path.exists(data_pickle_file):
            with open(data_pickle_file, "rb") as fh:
                data, categorical, attribute_names = pickle.load(fh)

            # Between v0.8 and v0.9 the format of pickled data changed from
            # np.ndarray to pd.DataFrame. This breaks some backwards compatibility,
            # e.g. for `run_model_on_task`. If a local file still exists with
            # np.ndarray data, we reprocess the data file to store a pickled
            # pd.DataFrame blob. See also #646.
            if isinstance(data, pd.DataFrame) or scipy.sparse.issparse(data):
                logger.debug("Data pickle file already exists.")
                return data_pickle_file

        try:
            data = self._get_arff(self.format)
        except OSError as e:
            logger.critical("Please check that the data file %s is "
                            "there and can be read.", data_file)
            raise e

        ARFF_DTYPES_TO_PD_DTYPE = {
            'INTEGER': 'integer',
            'REAL': 'floating',
            'NUMERIC': 'floating',
            'STRING': 'string'
        }
        attribute_dtype = {}
        attribute_names = []
        categories_names = {}
        categorical = []
        for name, type_ in data['attributes']:
            # if the feature is nominal and the a sparse matrix is
            # requested, the categories need to be numeric
            if (isinstance(type_, list)
                    and self.format.lower() == 'sparse_arff'):
                try:
                    np.array(type_, dtype=np.float32)
                except ValueError:
                    raise ValueError(
                        "Categorical data needs to be numeric when "
                        "using sparse ARFF."
                    )
            # string can only be supported with pandas DataFrame
            elif (type_ == 'STRING'
                  and self.format.lower() == 'sparse_arff'):
                raise ValueError(
                    "Dataset containing strings is not supported "
                    "with sparse ARFF."
                )

            # infer the dtype from the ARFF header
            if isinstance(type_, list):
                categorical.append(True)
                categories_names[name] = type_
                if len(type_) == 2:
                    type_norm = [cat.lower().capitalize()
                                 for cat in type_]
                    if set(['True', 'False']) == set(type_norm):
                        categories_names[name] = [
                            True if cat == 'True' else False
                            for cat in type_norm
                        ]
                        attribute_dtype[name] = 'boolean'
                    else:
                        attribute_dtype[name] = 'categorical'
                else:
                    attribute_dtype[name] = 'categorical'
            else:
                categorical.append(False)
                attribute_dtype[name] = ARFF_DTYPES_TO_PD_DTYPE[type_]
            attribute_names.append(name)

        if self.format.lower() == 'sparse_arff':
            X = data['data']
            X_shape = (max(X[1]) + 1, max(X[2]) + 1)
            X = scipy.sparse.coo_matrix(
                (X[0], (X[1], X[2])), shape=X_shape, dtype=np.float32)
            X = X.tocsr()

        elif self.format.lower() == 'arff':
            X = pd.DataFrame(data['data'], columns=attribute_names)

            col = []
            for column_name in X.columns:
                if attribute_dtype[column_name] in ('categorical',
                                                    'boolean'):
                    col.append(self._unpack_categories(
                        X[column_name], categories_names[column_name]))
                else:
                    col.append(X[column_name])
            X = pd.concat(col, axis=1)

        # Pickle the dataframe or the sparse matrix.
        with open(data_pickle_file, "wb") as fh:
            pickle.dump((X, categorical, attribute_names), fh, -1)
        logger.debug("Saved dataset {did}: {name} to file {path}"
                     .format(did=int(self.dataset_id or -1),
                             name=self.name,
                             path=data_pickle_file)
                     )
        return data_pickle_file

    def push_tag(self, tag):
        """Annotates this data set with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the dataset.
        """
        _tag_entity('data', self.dataset_id, tag)

    def remove_tag(self, tag):
        """Removes a tag from this dataset on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the dataset.
        """
        _tag_entity('data', self.dataset_id, tag, untag=True)

    def __eq__(self, other):

        if type(other) != OpenMLDataset:
            return False

        server_fields = {
            'dataset_id',
            'version',
            'upload_date',
            'url',
            'dataset',
            'data_file',
        }

        # check that the keys are identical
        self_keys = set(self.__dict__.keys()) - server_fields
        other_keys = set(other.__dict__.keys()) - server_fields
        if self_keys != other_keys:
            return False

        # check that values of the common keys are identical
        return all(self.__dict__[key] == other.__dict__[key]
                   for key in self_keys)

    def _get_arff(self, format):
        """Read ARFF file and return decoded arff.

        Reads the file referenced in self.data_file.

        Returns
        -------
        dict
            Decoded arff.

        """

        # TODO: add a partial read method which only returns the attribute
        # headers of the corresponding .arff file!
        import struct

        filename = self.data_file
        bits = (8 * struct.calcsize("P"))
        # Files can be considered too large on a 32-bit system,
        # if it exceeds 120mb (slightly more than covtype dataset size)
        # This number is somewhat arbitrary.
        if bits != 64 and os.path.getsize(filename) > 120000000:
            return NotImplementedError("File too big")

        if format.lower() == 'arff':
            return_type = arff.DENSE
        elif format.lower() == 'sparse_arff':
            return_type = arff.COO
        else:
            raise ValueError('Unknown data format %s' % format)

        def decode_arff(fh):
            decoder = arff.ArffDecoder()
            return decoder.decode(fh, encode_nominal=True,
                                  return_type=return_type)

        if filename[-3:] == ".gz":
            with gzip.open(filename) as fh:
                return decode_arff(fh)
        else:
            with io.open(filename, encoding='utf8') as fh:
                return decode_arff(fh)

    @staticmethod
    def _convert_array_format(data, array_format, attribute_names):
        """Convert a dataset to a given array format.

        By default, the data are stored as a sparse matrix or a pandas
        dataframe. One might be interested to get a pandas SparseDataFrame or a
        NumPy array instead, respectively.
        """
        if array_format == "array" and not scipy.sparse.issparse(data):
            # We encode the categories such that they are integer to be able
            # to make a conversion to numeric for backward compatibility
            def _encode_if_category(column):
                if column.dtype.name == 'category':
                    column = column.cat.codes.astype(np.float32)
                    mask_nan = column == -1
                    column[mask_nan] = np.nan
                return column
            if data.ndim == 2:
                columns = {
                    column_name: _encode_if_category(data.loc[:, column_name])
                    for column_name in data.columns
                }
                data = pd.DataFrame(columns)
            else:
                data = _encode_if_category(data)
            try:
                return np.asarray(data, dtype=np.float32)
            except ValueError:
                raise PyOpenMLError(
                    'PyOpenML cannot handle string when returning numpy'
                    ' arrays. Use dataset_format="dataframe".'
                )
        if array_format == "dataframe" and scipy.sparse.issparse(data):
            return pd.SparseDataFrame(data, columns=attribute_names)
        return data

    @staticmethod
    def _unpack_categories(series, categories):
        col = []
        for x in series:
            try:
                col.append(categories[int(x)])
            except (TypeError, ValueError):
                col.append(np.nan)
        return pd.Series(col, index=series.index, dtype='category',
                         name=series.name)

    def _download_data(self) -> None:
        """ Download ARFF data file to standard cache directory. Set `self.data_file`. """
        # import required here to avoid circular import.
        from .functions import _get_dataset_arff
        self.data_file = _get_dataset_arff(self)

    def get_data(self, target: str = None,
                 include_row_id: bool = False,
                 include_ignore_attributes: bool = False,
                 return_categorical_indicator: bool = False,
                 return_attribute_names: bool = False,
                 dataset_format: str = None):
        """ Returns dataset content as dataframes or sparse matrices.

        Parameters
        ----------
        target : string, list of strings or None (default=None)
            Name of target column(s) to separate from the data.
        include_row_id : boolean (default=False)
            Whether to include row ids in the returned dataset.
        include_ignore_attributes : boolean (default=False)
            Whether to include columns that are marked as "ignore"
            on the server in the dataset.
        return_categorical_indicator : boolean (default=False)
            Whether to return a boolean mask indicating which features are
            categorical.
        return_attribute_names : boolean (default=False)
            Whether to return attribute names.
        dataset_format : string, optional
            The format of returned dataset.
            If ``array``, the returned dataset will be a NumPy array or a SciPy sparse matrix.
            If ``dataframe``, the returned dataset will be a Pandas DataFrame or SparseDataFrame.

        Returns
        -------
        X : ndarray, dataframe, or sparse matrix, shape (n_samples, n_columns)
            Dataset
        y : ndarray or series, shape (n_samples,)
            Target column(s). Only returned if target is not None.
        categorical_indicator : boolean ndarray
            Mask that indicate categorical features.
            Only returned if return_categorical_indicator is True.
        return_attribute_names : list of strings
            List of attribute names.
            Only returned if return_attribute_names is True.
        """
        if dataset_format is None:
            warn('The default of "dataset_format" will change from "array" to'
                 ' "dataframe" in 0.9', FutureWarning)
            dataset_format = 'array'

        rval = []

        if self.data_pickle_file is None:
            if self.data_file is None:
                self._download_data()
            self.data_pickle_file = self._data_arff_to_pickle(self.data_file)

        path = self.data_pickle_file
        if not os.path.exists(path):
            raise ValueError("Cannot find a pickle file for dataset %s at "
                             "location %s " % (self.name, path))
        else:
            with open(path, "rb") as fh:
                data, categorical, attribute_names = pickle.load(fh)

        to_exclude = []
        if include_row_id is False:
            if not self.row_id_attribute:
                pass
            else:
                if isinstance(self.row_id_attribute, str):
                    to_exclude.append(self.row_id_attribute)
                else:
                    to_exclude.extend(self.row_id_attribute)

        if include_ignore_attributes is False:
            if not self.ignore_attributes:
                pass
            else:
                if isinstance(self.ignore_attributes, str):
                    to_exclude.append(self.ignore_attributes)
                else:
                    to_exclude.extend(self.ignore_attributes)

        if len(to_exclude) > 0:
            logger.info("Going to remove the following attributes:"
                        " %s" % to_exclude)
            keep = np.array([True if column not in to_exclude else False
                             for column in attribute_names])
            if hasattr(data, 'iloc'):
                data = data.iloc[:, keep]
            else:
                data = data[:, keep]
            categorical = [cat for cat, k in zip(categorical, keep) if k]
            attribute_names = [att for att, k in
                               zip(attribute_names, keep) if k]

        if target is None:
            data = self._convert_array_format(data, dataset_format,
                                              attribute_names)
            rval.append(data)
        else:
            if isinstance(target, str):
                if ',' in target:
                    target = target.split(',')
                else:
                    target = [target]
            targets = np.array([True if column in target else False
                                for column in attribute_names])
            if np.sum(targets) > 1:
                raise NotImplementedError(
                    "Number of requested targets %d is not implemented." %
                    np.sum(targets)
                )
            target_categorical = [
                cat for cat, column in zip(categorical, attribute_names)
                if column in target
            ]
            target_dtype = int if target_categorical[0] else float

            if hasattr(data, 'iloc'):
                x = data.iloc[:, ~targets]
                y = data.iloc[:, targets]
            else:
                x = data[:, ~targets]
                y = data[:, targets].astype(target_dtype)

            categorical = [cat for cat, t in zip(categorical, targets)
                           if not t]
            attribute_names = [att for att, k in zip(attribute_names, targets)
                               if not k]

            x = self._convert_array_format(x, dataset_format, attribute_names)
            if scipy.sparse.issparse(y):
                y = np.asarray(y.todense()).astype(target_dtype).flatten()
            y = y.squeeze()
            y = self._convert_array_format(y, dataset_format, attribute_names)
            y = y.astype(target_dtype) if dataset_format == 'array' else y

            rval.append(x)
            rval.append(y)

        if return_categorical_indicator:
            rval.append(categorical)
        if return_attribute_names:
            rval.append(attribute_names)

        if len(rval) == 1:
            return rval[0]
        else:
            return rval

    def retrieve_class_labels(self, target_name='class'):
        """Reads the datasets arff to determine the class-labels.

        If the task has no class labels (for example a regression problem)
        it returns None. Necessary because the data returned by get_data
        only contains the indices of the classes, while OpenML needs the real
        classname when uploading the results of a run.

        Parameters
        ----------
        target_name : str
            Name of the target attribute

        Returns
        -------
        list
        """
        for feature in self.features.values():
            if (feature.name == target_name) and (feature.data_type == 'nominal'):
                return feature.nominal_values
        return None

    def get_features_by_type(self, data_type, exclude=None,
                             exclude_ignore_attributes=True,
                             exclude_row_id_attribute=True):
        """
        Return indices of features of a given type, e.g. all nominal features.
        Optional parameters to exclude various features by index or ontology.

        Parameters
        ----------
        data_type : str
            The data type to return (e.g., nominal, numeric, date, string)
        exclude : list(int)
            Indices to exclude (and adapt the return values as if these indices
                        are not present)
        exclude_ignore_attributes : bool
            Whether to exclude the defined ignore attributes (and adapt the
            return values as if these indices are not present)
        exclude_row_id_attribute : bool
            Whether to exclude the defined row id attributes (and adapt the
            return values as if these indices are not present)

        Returns
        -------
        result : list
            a list of indices that have the specified data type
        """
        if data_type not in OpenMLDataFeature.LEGAL_DATA_TYPES:
            raise TypeError("Illegal feature type requested")
        if self.ignore_attributes is not None:
            if not isinstance(self.ignore_attributes, list):
                raise TypeError("ignore_attributes should be a list")
        if self.row_id_attribute is not None:
            if not isinstance(self.row_id_attribute, str):
                raise TypeError("row id attribute should be a str")
        if exclude is not None:
            if not isinstance(exclude, list):
                raise TypeError("Exclude should be a list")
            # assert all(isinstance(elem, str) for elem in exclude),
            #            "Exclude should be a list of strings"
        to_exclude = []
        if exclude is not None:
            to_exclude.extend(exclude)
        if exclude_ignore_attributes and self.ignore_attributes is not None:
            to_exclude.extend(self.ignore_attributes)
        if exclude_row_id_attribute and self.row_id_attribute is not None:
            to_exclude.append(self.row_id_attribute)

        result = []
        offset = 0
        # this function assumes that everything in to_exclude will
        # be 'excluded' from the dataset (hence the offset)
        for idx in self.features:
            name = self.features[idx].name
            if name in to_exclude:
                offset += 1
            else:
                if self.features[idx].data_type == data_type:
                    result.append(idx - offset)
        return result

    def publish(self):
        """Publish the dataset on the OpenML server.

        Upload the dataset description and dataset content to openml.

        Returns
        -------
        dataset_id: int
            Id of the dataset uploaded to the server.
        """
        file_elements = {'description': self._to_xml()}

        # the arff dataset string is available
        if self._dataset is not None:
            file_elements['dataset'] = self._dataset
        else:
            # the path to the arff dataset is given
            if self.data_file is not None:
                path = os.path.abspath(self.data_file)
                if os.path.exists(path):
                    try:
                        # check if arff is valid
                        decoder = arff.ArffDecoder()
                        with io.open(path, encoding='utf8') as fh:
                            decoder.decode(fh, encode_nominal=True)
                    except arff.ArffException:
                        raise ValueError("The file you have provided is not "
                                         "a valid arff file.")

                    file_elements['dataset'] = open(path, 'rb')
            else:
                if self.url is None:
                    raise ValueError("No url/path to the data file was given")

        return_value = openml._api_calls._perform_api_call(
            "data/", 'post',
            file_elements=file_elements,
        )
        response = xmltodict.parse(return_value)
        self.dataset_id = int(response['oml:upload_data_set']['oml:id'])
        return self.dataset_id

    def _to_xml(self):
        """ Serialize object to xml for upload

        Returns
        -------
        xml_dataset : str
            XML description of the data.
        """
        props = ['id', 'name', 'version', 'description', 'format', 'creator',
                 'contributor', 'collection_date', 'upload_date', 'language',
                 'licence', 'url', 'default_target_attribute',
                 'row_id_attribute', 'ignore_attribute', 'version_label',
                 'citation', 'tag', 'visibility', 'original_data_url',
                 'paper_url', 'update_comment', 'md5_checksum']

        data_container = OrderedDict()
        data_dict = OrderedDict([('@xmlns:oml', 'http://openml.org/openml')])
        data_container['oml:data_set_description'] = data_dict

        for prop in props:
            content = getattr(self, prop, None)
            if content is not None:
                data_dict["oml:" + prop] = content

        xml_string = xmltodict.unparse(
            input_dict=data_container,
            pretty=True,
        )
        # A flow may not be uploaded with the xml encoding specification:
        # <?xml version="1.0" encoding="utf-8"?>
        xml_string = xml_string.split('\n', 1)[-1]
        return xml_string


def _check_qualities(qualities):
    if qualities is not None:
        qualities_ = {}
        for xmlquality in qualities:
            name = xmlquality['oml:name']
            if xmlquality.get('oml:value', None) is None:
                value = float('NaN')
            elif xmlquality['oml:value'] == 'null':
                value = float('NaN')
            else:
                value = float(xmlquality['oml:value'])
            qualities_[name] = value
        return qualities_
    else:
        return None
