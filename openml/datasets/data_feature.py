# License: BSD 3-Clause
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from IPython.lib import pretty


class OpenMLDataFeature:  # noqa: PLW1641
    """
    Data Feature (a.k.a. Attribute) object.

    Parameters
    ----------
    index : int
        The index of this feature. Must be non-negative.
    name : str
        Name of the feature. Must be a non-empty string.
    data_type : str
        can be nominal, numeric, string, date (corresponds to arff)
    nominal_values : list(str)
        list of the possible values, in case of nominal attribute.
        Must be a non-empty list for nominal data types.
    number_missing_values : int
        Number of rows that have a missing value for this feature.
        Must be non-negative.
    ontologies : list(str), optional
        list of ontologies attached to this feature. An ontology describes the
        concept that are described in a feature. An ontology is defined by an
        URL where the information is provided.

    Raises
    ------
    TypeError
        If types are incorrect for any parameter.
    ValueError
        If values are invalid (e.g., negative counts, empty name, invalid data_type).
    """

    LEGAL_DATA_TYPES: ClassVar[Sequence[str]] = ["nominal", "numeric", "string", "date"]

    def __init__(  # noqa: PLR0913
        self,
        index: int,
        name: str,
        data_type: str,
        nominal_values: list[str],
        number_missing_values: int,
        ontologies: list[str] | None = None,
    ):
        # Validate index
        if not isinstance(index, int):
            raise TypeError(
                f"Parameter 'index' must be int, but got {type(index).__name__}. Value: {index!r}"
            )
        if index < 0:
            raise ValueError(
                f"Parameter 'index' must be non-negative, but got {index}. "
                "Feature indices cannot be negative."
            )

        # Validate name
        if not isinstance(name, str):
            raise TypeError(
                f"Parameter 'name' must be str, but got {type(name).__name__}. Value: {name!r}"
            )
        if not name.strip():
            raise ValueError(f"Parameter 'name' cannot be empty or whitespace-only. Got: {name!r}")

        # Validate data_type
        if not isinstance(data_type, str):
            raise TypeError(
                f"Parameter 'data_type' must be str, but got {type(data_type).__name__}. "
                f"Value: {data_type!r}"
            )
        if data_type not in self.LEGAL_DATA_TYPES:
            raise ValueError(
                f"Parameter 'data_type' must be one of {list(self.LEGAL_DATA_TYPES)}, "
                f"but got {data_type!r}."
            )

        # Validate nominal_values
        if data_type == "nominal":
            if nominal_values is None:
                raise TypeError(
                    "Parameter 'nominal_values' is required for nominal data types, but got None. "
                    "Please provide a list of nominal values."
                )
            if not isinstance(nominal_values, list):
                raise TypeError(
                    f"Parameter 'nominal_values' must be list, but got {type(nominal_values).__name__}. "
                    f"Value: {nominal_values!r}"
                )
            if not nominal_values:
                raise ValueError(
                    "Parameter 'nominal_values' cannot be empty for nominal data types. "
                    "Please provide at least one nominal value."
                )
            # Validate that all elements are strings
            non_string_values = [v for v in nominal_values if not isinstance(v, str)]
            if non_string_values:
                raise TypeError(
                    f"All elements in 'nominal_values' must be str, but found non-string values: "
                    f"{non_string_values}. Expected all strings in list."
                )
        elif nominal_values is not None:
            raise TypeError(
                f"Parameter 'nominal_values' must be None for non-nominal data types (got {data_type!r}), "
                f"but got {type(nominal_values).__name__}. Value: {nominal_values!r}"
            )

        # Validate number_missing_values
        if not isinstance(number_missing_values, int):
            raise TypeError(
                f"Parameter 'number_missing_values' must be int, but got "
                f"{type(number_missing_values).__name__}. Value: {number_missing_values!r}"
            )
        if number_missing_values < 0:
            raise ValueError(
                f"Parameter 'number_missing_values' must be non-negative, but got "
                f"{number_missing_values}. Cannot have negative missing values."
            )

        # Validate ontologies (keep simple - just check if list or None)
        if ontologies is not None:
            if not isinstance(ontologies, list):
                raise TypeError(
                    f"Parameter 'ontologies' must be list or None, but got {type(ontologies).__name__}. "
                    f"Value: {ontologies!r}"
                )

        # All validations passed, assign attributes
        self.index = index
        self.name = name
        self.data_type = data_type
        self.nominal_values = nominal_values
        self.number_missing_values = number_missing_values
        self.ontologies = ontologies if ontologies is not None else []

    def __repr__(self) -> str:
        return f"[{self.index} - {self.name} ({self.data_type})]"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, OpenMLDataFeature) and self.__dict__ == other.__dict__

    def _repr_pretty_(self, pp: pretty.PrettyPrinter, cycle: bool) -> None:  # noqa: ARG002
        pp.text(str(self))
