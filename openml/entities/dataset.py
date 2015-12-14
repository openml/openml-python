import gzip
import os
import sys

import arff

if sys.version_info[0] > 3:
    import pickle
else:
    try:
        import cPickle as pickle
    except:
        import pickle

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse

from ..util import is_string


class OpenMLDataset(object):
    def __init__(self, id, name, version, description, format, creator,
                 contributor, collection_date, upload_date, language,
                 licence, url, default_target_attribute, row_id_attribute,
                 ignore_attribute, version_label, citation, tag, visibility,
                 original_data_url, paper_url, update_comment, md5_checksum,
                 data_file):
        # Attributes received by querying the RESTful API
        self.id = int(id)
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
        self.ignore_attributes = ignore_attribute
        self.version_label = version_label
        self.citation = citation
        self.tag = tag
        self.visibility = visibility
        self.original_data_url = original_data_url
        self.paper_url = paper_url
        self.update_comment = update_comment
        self.md5_cheksum = md5_checksum
        self.data_file = data_file

        self.data_pickle_file = data_file.replace('.arff', '.pkl')

        if os.path.exists(self.data_pickle_file):
            logger.debug("Data pickle file already exists.")
        else:
            try:
                data = self.get_arff()
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
                         (self.id, self.name, self.data_pickle_file))

    def __eq__(self, other):
        if type(other) != OpenMLDataset:
            return False
        elif self.id == other._id or \
                (self.name == other._name and self.version == other._version):
            return True
        else:
            return False

    ############################################################################
    # ARFF related stuff
    def get_arff(self):
        # TODO: add a partial read method which only returns the attribute
        # headers of the corresponding .arff file!

        # A random number after which we consider a file for too large on a
        # 32 bit system...currently 120mb (just a little bit more than covtype)
        import struct

        filename = self.data_file
        bits = ( 8 * struct.calcsize("P"))
        if bits != 64 and os.path.getsize(filename) > 120000000:
            return NotImplementedError("File too big")

        def decode_arff(fh):
            decoder = arff.ArffDecoder()
            return decoder.decode(fh, encode_nominal=True)

        if filename[-3:] == ".gz":
            with gzip.open(filename) as fh:
                return decode_arff(fh)
        else:
            with open(filename) as fh:
                return decode_arff(fh)

    ############################################################################
    # pandas related stuff...
    def get_dataset(self, target=None, include_row_id=False,
                   include_ignore_attributes=False,
                   return_categorical_indicator=False,
                   return_attribute_names=False):
        rval = []

        path = self.data_pickle_file
        if not os.path.exists(path):
            raise ValueError("Cannot find a ndarray file for dataset %s at"
                             "location %s " % (self.name, path))
        else:
            with open(path, "rb") as fh:
                data, categorical, attribute_names = pickle.load(fh)

        to_exclude = []
        if include_row_id is False:
            if not self.row_id_attribute:
                pass
            else:
                if is_string(self.row_id_attribute):
                    to_exclude.append(self.row_id_attribute)
                else:
                    to_exclude.extend(self.row_id_attribute)

        if include_ignore_attributes is False:
            if not self.ignore_attributes:
                pass
            else:
                if is_string(self.ignore_attributes):
                    to_exclude.append(self.ignore_attributes)
                else:
                    to_exclude.extend(self.ignore_attributes)

        if len(to_exclude) > 0:
            logger.info("Going to remove the following row_id_attributes:"
                        " %s" % self.row_id_attribute)
            keep = np.array([True if column not in to_exclude else False
                             for column in attribute_names])
            data = data[:,keep]
            categorical = [cat for cat, k in zip(categorical, keep) if k]
            attribute_names = [att for att, k in
                               zip(attribute_names, keep) if k]

        if target is None:
            rval.append(data)
        else:
            if is_string(target):
                target = [target]
            targets = np.array([True if column in target else False
                                for column in attribute_names])

            try:
                x = data[:,~targets]
                y = data[:,targets].astype(np.int32)

                if len(y.shape) == 2 and y.shape[1] == 1:
                    y = y[:,0]

                categorical = [cat for cat, t in
                               zip(categorical, targets) if not t]
                attribute_names = [att for att, k in
                                   zip(attribute_names, targets) if not k]
            except KeyError as e:
                import sys
                sys.stdout.flush()
                raise e

            if scipy.sparse.issparse(y):
                y = np.asarray(y.todense()).astype(np.int32).flatten()

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
