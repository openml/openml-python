# License: BSD 3-Clause
from __future__ import annotations

import gzip
import logging
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence
from typing_extensions import Literal

import arff
import numpy as np
import pandas as pd
import scipy.sparse
import xmltodict

from openml.base import OpenMLBase
from openml.config import OPENML_SKIP_PARQUET_ENV_VAR
from openml.exceptions import PyOpenMLError

from .data_feature import OpenMLDataFeature

logger = logging.getLogger(__name__)


class OpenMLDataset(OpenMLBase):
    # ... [all of your existing __init__, properties, helper methods, etc.] ...


    def get_data(  # noqa: C901, PLR0912, PLR0915
        self,
        target: list[str] | str | None = None,
        include_row_id: bool = False,  # noqa: FBT001, FBT002
        include_ignore_attribute: bool = False,  # noqa: FBT001, FBT002
        dataset_format: Literal["array", "dataframe"] = "dataframe",
        as_frame: bool = False,
    ) -> tuple[
        np.ndarray | pd.DataFrame | scipy.sparse.csr_matrix,
        np.ndarray | pd.Series | None,
        list[bool],
        list[str],
    ]:
        """
        Returns dataset content as dataframes or sparse matrices.

        Parameters
        ----------
        target : string, List[str] or None (default=None)
            Name of target column to separate from the data.
        include_row_id : boolean (default=False)
            Whether to include row ids in the returned dataset.
        include_ignore_attribute : boolean (default=False)
            Whether to include columns marked as "ignore".
        dataset_format : string (default='dataframe')
            'array' for numpy/sparse, 'dataframe' for pandas.
        as_frame : boolean (default=False)
            If True, always return X as pandas DataFrame and y as pandas Series.

        Returns
        -------
        X : ndarray, DataFrame, or sparse matrix
        y : ndarray, Series, or None
        categorical_indicator : list of bool
        attribute_names : list of str
        """
        if dataset_format == "array":
            warnings.warn(
                "Support for `dataset_format='array'` will be removed in 0.15; "
                "use `dataframe` and then `.to_numpy()`.",
                category=FutureWarning,
                stacklevel=2,
            )

        # load raw data
        data, categorical, attribute_names = self._load_data()

        # exclude row_id / ignore columns
        to_exclude = []
        if not include_row_id and self.row_id_attribute is not None:
            if isinstance(self.row_id_attribute, str):
                to_exclude.append(self.row_id_attribute)
            else:
                to_exclude.extend(self.row_id_attribute)
        if not include_ignore_attribute and self.ignore_attribute is not None:
            to_exclude.extend(self.ignore_attribute)

        if to_exclude:
            mask = [col not in to_exclude for col in attribute_names]
            if isinstance(data, pd.DataFrame):
                data = data.loc[:, mask]
            else:
                data = data[:, mask]
            categorical = [c for c, m in zip(categorical, mask) if m]
            attribute_names = [n for n, m in zip(attribute_names, mask) if m]

        # separate target
        if target is None:
            X = self._convert_array_format(data, dataset_format, attribute_names)  # type: ignore
            y = None
        else:
            tlist = [target] if isinstance(target, str) else target
            sel = [name in tlist for name in attribute_names]
            # X without target columns, y with
            if isinstance(data, pd.DataFrame):
                X = data.loc[:, [not s for s in sel]]
                y = data.loc[:, sel]
            else:
                X = data[:, [not s for s in sel]]
                y = data[:, sel]
            # convert formats
            X = self._convert_array_format(X, dataset_format, attribute_names)  # type: ignore
            y = self._convert_array_format(y, dataset_format, [name for name, s in zip(attribute_names, sel) if s])
            if isinstance(y, np.ndarray):
                y = pd.Series(y) if as_frame else y

            # update categorical and names
            categorical = [c for c, s in zip(categorical, sel) if not s]
            attribute_names = [n for n, s in zip(attribute_names, sel) if not s]

        # --- as_frame override ---
        if as_frame:
            # ensure X is DataFrame
            X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=attribute_names)
            # ensure y is Series or None
            if isinstance(y, np.ndarray):
                y = pd.Series(y, name=(target if isinstance(target, str) else None))
            return X, y, categorical, attribute_names

        return X, y, categorical, attribute_names

    def _dataset_to_dict(self) -> dict[str, Any]:
        dataset_dict = {
            "name": self.name,
            "description": self.description,
            "creator": self.creator,
            "contributor": self.contributor,
            "collection_date": self.collection_date,
            "language": self.language,
            "licence": self.licence,
            "default_target_attribute": self.default_target_attribute,
            "row_id_attribute": self.row_id_attribute,
            "ignore_attribute": self.ignore_attribute,
            "citation": self.citation,
            "version_label": self.version_label,
            "original_data_url": self.original_data_url,
            "paper_url": self.paper_url,
            "update_comment": self.update_comment,
            "format": self.data_format,  # <-- THIS LINE is the key
        }
        return dataset_dict

