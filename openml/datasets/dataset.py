import gzip
import io
import logging
import os
import six

import arff

import numpy as np
import scipy.sparse
from six.moves import cPickle as pickle
import xmltodict

from .data_feature import OpenMLDataFeature
from ..exceptions import PyOpenMLError
import openml._api_calls

logger = logging.getLogger(__name__)


class OpenMLDataset(object):
    """Dataset object.

    Allows fetching and uploading datasets to OpenML.

    Parameters
    ----------
    name : str
        Name of the dataset
    description : str
        Description of the dataset
    FIXME : which of these do we actually nee?
    """
    def __init__(self, dataset_id=None, name=None, version=None, description=None,
                 format=None, creator=None, contributor=None,
                 collection_date=None, upload_date=None, language=None,
                 licence=None, url=None, default_target_attribute=None,
                 row_id_attribute=None, ignore_attribute=None,
                 version_label=None, citation=None, tag=None, visibility=None,
                 original_data_url=None, paper_url=None, update_comment=None,
                 md5_checksum=None, data_file=None, features=None, qualities=None):
        # Attributes received by querying the RESTful API
        self.dataset_id = int(dataset_id) if dataset_id is not None else None
        self.name = name
        self.version = int(version)
        self.description = description
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
        self.ignore_attributes = None
        if isinstance(ignore_attribute, six.string_types):
            self.ignore_attributes = [ignore_attribute]
        elif isinstance(ignore_attribute, list):
            self.ignore_attributes = ignore_attribute
        elif ignore_attribute is None:
            pass
        else:
            raise ValueError('wrong data type for ignore_attribute. Should be list. ')
        self.version_label = version_label
        self.citation = citation
        self.tag = tag
        self.visibility = visibility
        self.original_data_url = original_data_url
        self.paper_url = paper_url
        self.update_comment = update_comment
        self.md5_cheksum = md5_checksum
        self.data_file = data_file
        self.features = None
        self.qualities = None

        if features is not None:
            self.features = {}
            for idx, xmlfeature in enumerate(features['oml:feature']):
                feature = OpenMLDataFeature(int(xmlfeature['oml:index']),
                                            xmlfeature['oml:name'],
                                            xmlfeature['oml:data_type'],
                                            None,  # todo add nominal values (currently not in database)
                                            int(xmlfeature.get('oml:number_of_missing_values', 0)))
                if idx != feature.index:
                    raise ValueError('Data features not provided in right order')
                self.features[feature.index] = feature

        self.qualities = _check_qualities(qualities)

        if data_file is not None:
            if self._data_features_supported():
                if six.PY2:
                    self.data_pickle_file = data_file.replace('.arff', '.pkl.py2')
                else:
                    self.data_pickle_file = data_file.replace('.arff', '.pkl.py3')

                if os.path.exists(self.data_pickle_file):
                    logger.debug("Data pickle file already exists.")
                else:
                    try:
                        data = self._get_arff(self.format)
                    except OSError as e:
                        logger.critical("Please check that the data file %s is there "
                                        "and can be read.", self.data_file)
                        raise e

                    categorical = [False if type(type_) != list else True
                                   for name, type_ in data['attributes']]
                    attribute_names = [name for name, type_ in data['attributes']]

                    if isinstance(data['data'], tuple):
                        X = data['data']
                        X_shape = (max(X[1]) + 1, max(X[2]) + 1)
                        X = scipy.sparse.coo_matrix(
                            (X[0], (X[1], X[2])), shape=X_shape, dtype=np.float32)
                        X = X.tocsr()
                    elif isinstance(data['data'], list):
                        X = np.array(data['data'], dtype=np.float32)
                    else:
                        raise Exception()

                    with open(self.data_pickle_file, "wb") as fh:
                        pickle.dump((X, categorical, attribute_names), fh, -1)
                    logger.debug("Saved dataset %d: %s to file %s" %
                                 (int(self.dataset_id or -1), self.name, self.data_pickle_file))

    def push_tag(self, tag):
        """Annotates this data set with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the dataset.
        """
        data = {'data_id': self.dataset_id, 'tag': tag}
        openml._api_calls._perform_api_call("/data/tag", data=data)

    def remove_tag(self, tag):
        """Removes a tag from this dataset on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the dataset.
        """
        data = {'data_id': self.dataset_id, 'tag': tag}
        openml._api_calls._perform_api_call("/data/untag", data=data)

    def __eq__(self, other):
        if type(other) != OpenMLDataset:
            return False
        elif (
            self.dataset_id == other.dataset_id
            or (self.name == other._name and self.version == other._version)
        ):
            return True
        else:
            return False

    def _get_arff(self, format):
        """Read ARFF file and return decoded arff.

        Reads the file referenced in self.data_file.

        Returns
        -------
        arff_string :
            Decoded arff.

        """

        # TODO: add a partial read method which only returns the attribute
        # headers of the corresponding .arff file!

        # A random number after which we consider a file for too large on a
        # 32 bit system...currently 120mb (just a little bit more than covtype)
        import struct

        if not self._data_features_supported():
            raise PyOpenMLError('Dataset not compatible, PyOpenML cannot handle string features')

        filename = self.data_file
        bits = (8 * struct.calcsize("P"))
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

    def get_data(self, target=None,
                 include_row_id=False,
                 include_ignore_attributes=False,
                 return_categorical_indicator=False,
                 return_attribute_names=False
    ):
        """Returns dataset content as numpy arrays / sparse matrices.

        Parameters
        ----------


        Returns
        -------

        """
        rval = []

        if not self._data_features_supported():
            raise PyOpenMLError(
                'Dataset %d not compatible, PyOpenML cannot handle string '
                'features' % self.dataset_id
            )

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
                if isinstance(self.row_id_attribute, six.string_types):
                    to_exclude.append(self.row_id_attribute)
                else:
                    to_exclude.extend(self.row_id_attribute)

        if include_ignore_attributes is False:
            if not self.ignore_attributes:
                pass
            else:
                if isinstance(self.ignore_attributes, six.string_types):
                    to_exclude.append(self.ignore_attributes)
                else:
                    to_exclude.extend(self.ignore_attributes)

        if len(to_exclude) > 0:
            logger.info("Going to remove the following attributes:"
                        " %s" % to_exclude)
            keep = np.array([True if column not in to_exclude else False
                             for column in attribute_names])
            data = data[:, keep]
            categorical = [cat for cat, k in zip(categorical, keep) if k]
            attribute_names = [att for att, k in
                               zip(attribute_names, keep) if k]

        if target is None:
            rval.append(data)
        else:
            if isinstance(target, six.string_types):
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
                cat for cat, column in
                six.moves.zip(categorical, attribute_names)
                if column in target
            ]
            target_dtype = int if target_categorical[0] else float

            try:
                x = data[:, ~targets]
                y = data[:, targets].astype(target_dtype)

                if len(y.shape) == 2 and y.shape[1] == 1:
                    y = y[:, 0]

                categorical = [cat for cat, t in
                               zip(categorical, targets) if not t]
                attribute_names = [att for att, k in
                                   zip(attribute_names, targets) if not k]
            except KeyError as e:
                import sys
                sys.stdout.flush()
                raise e

            if scipy.sparse.issparse(y):
                y = np.asarray(y.todense()).astype(target_dtype).flatten()

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

        # TODO improve performance, currently reads the whole file
        # Should make a method that only reads the attributes
        arffFileName = self.data_file

        if self.format.lower() == 'arff':
            return_type = arff.DENSE
        elif self.format.lower() == 'sparse_arff':
            return_type = arff.COO
        else:
            raise ValueError('Unknown data format %s' % self.format)

        with io.open(arffFileName, encoding='utf8') as fh:
            arffData = arff.ArffDecoder().decode(fh, return_type=return_type)

        dataAttributes = dict(arffData['attributes'])
        if target_name in dataAttributes:
            return dataAttributes[target_name]
        else:
            return None

    def get_features_by_type(self, data_type, exclude=None,
                             exclude_ignore_attributes=True,
                             exclude_row_id_attribute=True):
        '''
        Returns indices of features of a given type, e.g., all nominal features.
        Can use additional parameters to exclude various features by index or ontology.

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
        '''
        if data_type not in OpenMLDataFeature.LEGAL_DATA_TYPES:
            raise TypeError("Illegal feature type requested")
        if self.ignore_attributes is not None:
            if not isinstance(self.ignore_attributes, list):
                raise TypeError("ignore_attributes should be a list")
        if self.row_id_attribute is not None:
            if not isinstance(self.row_id_attribute, six.string_types):
                raise TypeError("row id attribute should be a str")
        if exclude is not None:
            if not isinstance(exclude, list):
                raise TypeError("Exclude should be a list")
            # assert all(isinstance(elem, str) for elem in exclude), "Exclude should be a list of strings"
        to_exclude = []
        if exclude is not None:
            to_exclude.extend(exclude)
        if exclude_ignore_attributes and self.ignore_attributes is not None:
            to_exclude.extend(self.ignore_attributes)
        if exclude_row_id_attribute and self.row_id_attribute is not None:
            to_exclude.append(self.row_id_attribute)

        result = []
        offset = 0
        # this function assumes that everything in to_exclude will be 'excluded' from the dataset (hence the offset)
        for idx in self.features:
            name = self.features[idx].name
            if name in to_exclude:
                offset += 1
            else:
                if self.features[idx].data_type == data_type:
                    result.append(idx-offset)
        return result

    def publish(self):
        """Publish the dataset on the OpenML server.

        Upload the dataset description and dataset content to openml.

        Returns
        -------
        self
        """

        file_elements = {'description': self._to_xml()}
        file_dictionary = {}

        if self.data_file is not None:
            file_dictionary['dataset'] = self.data_file

        return_value = openml._api_calls._perform_api_call(
            "/data/",
            file_dictionary=file_dictionary,
            file_elements=file_elements,
        )

        self.dataset_id = int(xmltodict.parse(return_value)['oml:upload_data_set']['oml:id'])
        return self

    def _to_xml(self):
        """Serialize object to xml for upload

        Returns
        -------
        xml_dataset : str
            XML description of the data.
        """
        xml_dataset = ('<oml:data_set_description '
                       'xmlns:oml="http://openml.org/openml">\n')
        props = ['id', 'name', 'version', 'description', 'format', 'creator',
                 'contributor', 'collection_date', 'upload_date', 'language',
                 'licence', 'url', 'default_target_attribute',
                 'row_id_attribute', 'ignore_attribute', 'version_label',
                 'citation', 'tag', 'visibility', 'original_data_url',
                 'paper_url', 'update_comment', 'md5_checksum']  # , 'data_file']
        for prop in props:
            content = getattr(self, prop, None)
            if content is not None:
                if isinstance(content, (list,set)):
                    for item in content:
                        xml_dataset += "<oml:{0}>{1}</oml:{0}>\n".format(prop, item)
                else:
                    xml_dataset += "<oml:{0}>{1}</oml:{0}>\n".format(prop, content)
        xml_dataset += "</oml:data_set_description>"
        return xml_dataset

    def _data_features_supported(self):
        if self.features is not None:
            for idx in self.features:
                if self.features[idx].data_type not in ['numeric', 'nominal']:
                    return False
            return True
        return True


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
