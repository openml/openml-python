"""Command Line Interface for `openml` to configure its settings."""

from __future__ import annotations

import argparse
import string
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import openml
from openml.__version__ import __version__
from openml.enums import APIVersion


def is_hex(string_: str) -> bool:
    return all(c in string.hexdigits for c in string_)


def looks_like_url(url: str) -> bool:
    # There's no thorough url parser, but we only seem to use netloc.
    try:
        return bool(urlparse(url).netloc)
    except Exception:  # noqa: BLE001
        return False


def wait_until_valid_input(
    prompt: str,
    check: Callable[[str], str],
    sanitize: Callable[[str], str] | None,
) -> str:
    """Asks `prompt` until an input is received which returns True for `check`.

    Parameters
    ----------
    prompt: str
        message to display
    check: Callable[[str], str]
        function to call with the given input, that provides an error message if the input is not
        valid otherwise, and False-like otherwise.
    sanitize: Callable[[str], str], optional
        A function which attempts to sanitize the user input (e.g. auto-complete).

    Returns
    -------
    valid input

    """
    while True:
        response = input(prompt)
        if sanitize:
            response = sanitize(response)
        error_message = check(response)
        if error_message:
            print(error_message, end="\n\n")
        else:
            return response


def print_configuration() -> None:
    file = openml.config.determine_config_file_path()
    header = f"File '{file}' contains (or defaults to):"
    print(header)

    max_key_length = max(map(len, openml.config.get_config_as_dict()))
    for field, value in openml.config.get_config_as_dict().items():
        print(f"{field.ljust(max_key_length)}: {value}")


def verbose_set(field: str, value: str) -> None:
    openml.config.set_field_in_config_file(field, value)
    print(f"{field} set to '{value}'.")


def configure_apikey(value: str) -> None:
    def check_apikey(apikey: str) -> str:
        if len(apikey) != 32:
            return f"The key should contain 32 characters but contains {len(apikey)}."
        if not is_hex(apikey):
            return "Some characters are not hexadecimal."
        return ""

    instructions = (
        f"Your current API key is set to: '{openml.config.apikey}'. "
        "You can get an API key at https://new.openml.org. "
        "You must create an account if you don't have one yet:\n"
        "  1. Log in with the account.\n"
        "  2. Navigate to the profile page (top right circle > Your Profile). \n"
        "  3. Click the API Key button to reach the page with your API key.\n"
        "If you have any difficulty following these instructions, let us know on Github."
    )

    configure_field(
        field="apikey",
        value=value,
        check_with_message=check_apikey,
        intro_message=instructions,
        input_message="Please enter your API key:",
    )


def configure_server(value: str) -> None:
    def check_server(server: str) -> str:
        is_shorthand = server in ["test", "production_server"]
        if is_shorthand or looks_like_url(server):
            return ""
        return "Must be 'test', 'production_server' or a url."

    def replace_shorthand(server: str) -> str:
        if server == "test":
            return cast("str", openml.config.get_test_servers()[APIVersion.V1]["server"])
        if server == "production_server":
            return cast("str", openml.config.get_production_servers()[APIVersion.V1]["server"])
        return server

    configure_field(
        field="server",
        value=value,
        check_with_message=check_server,
        intro_message="Specify which server you wish to connect to.",
        input_message="Specify a url or use 'test' or 'production_server' as a shorthand: ",
        sanitize=replace_shorthand,
    )


def configure_cachedir(value: str) -> None:
    def check_cache_dir(path: str) -> str:
        _path = Path(path)
        if _path.is_file():
            return f"'{_path}' is a file, not a directory."

        expanded = _path.expanduser()
        if not expanded.is_absolute():
            return f"'{_path}' is not absolute (even after expanding '~')."

        if not expanded.exists():
            try:
                expanded.mkdir()
            except PermissionError:
                return f"'{path}' does not exist and there are not enough permissions to create it."

        return ""

    configure_field(
        field="cachedir",
        value=value,
        check_with_message=check_cache_dir,
        intro_message="Configuring the cache directory. It can not be a relative path.",
        input_message="Specify the directory to use (or create) as cache directory: ",
    )


def configure_connection_n_retries(value: str) -> None:
    def valid_connection_retries(n: str) -> str:
        if not n.isdigit():
            return f"'{n}' is not a valid positive integer."
        if int(n) <= 0:
            return "connection_n_retries must be positive."
        return ""

    configure_field(
        field="connection_n_retries",
        value=value,
        check_with_message=valid_connection_retries,
        intro_message="Configuring the number of times to attempt to connect to the OpenML Server",
        input_message="Enter a positive integer: ",
    )


def configure_avoid_duplicate_runs(value: str) -> None:
    def is_python_bool(bool_: str) -> str:
        if bool_ in ["True", "False"]:
            return ""
        return "Must be 'True' or 'False' (mind the capital)."

    def autocomplete_bool(bool_: str) -> str:
        if bool_.lower() in ["n", "no", "f", "false", "0"]:
            return "False"
        if bool_.lower() in ["y", "yes", "t", "true", "1"]:
            return "True"
        return bool_

    intro_message = (
        "If set to True, when `run_flow_on_task` or similar methods are called a lookup is "
        "performed to see if there already exists such a run on the server. "
        "If so, download those results instead. "
        "If set to False, runs will always be executed."
    )

    configure_field(
        field="avoid_duplicate_runs",
        value=value,
        check_with_message=is_python_bool,
        intro_message=intro_message,
        input_message="Enter 'True' or 'False': ",
        sanitize=autocomplete_bool,
    )


def configure_verbosity(value: str) -> None:
    def is_zero_through_two(verbosity: str) -> str:
        if verbosity in ["0", "1", "2"]:
            return ""
        return "Must be '0', '1' or '2'."

    intro_message = (
        "Set the verbosity of log messages which should be shown by openml-python."
        " 0: normal output (warnings and errors)"
        " 1: info output (some high-level progress output)"
        " 2: debug output (detailed information (for developers))"
    )

    configure_field(
        field="verbosity",
        value=value,
        check_with_message=is_zero_through_two,
        intro_message=intro_message,
        input_message="Enter '0', '1' or '2': ",
    )


def configure_retry_policy(value: str) -> None:
    def is_known_policy(policy: str) -> str:
        if policy in ["human", "robot"]:
            return ""
        return "Must be 'human' or 'robot'."

    def autocomplete_policy(policy: str) -> str:
        for option in ["human", "robot"]:
            if option.startswith(policy.lower()):
                return option
        return policy

    intro_message = (
        "Set the retry policy which determines how to react if the server is unresponsive."
        "We recommend 'human' for interactive usage and 'robot' for scripts."
        "'human': try a few times in quick succession, less reliable but quicker response."
        "'robot': try many times with increasing intervals, more reliable but slower response."
    )

    configure_field(
        field="retry_policy",
        value=value,
        check_with_message=is_known_policy,
        intro_message=intro_message,
        input_message="Enter 'human' or 'robot': ",
        sanitize=autocomplete_policy,
    )


def configure_field(  # noqa: PLR0913
    field: str,
    value: None | str,
    check_with_message: Callable[[str], str],
    intro_message: str,
    input_message: str,
    sanitize: Callable[[str], str] | None = None,
) -> None:
    """Configure `field` with `value`. If `value` is None ask the user for input.

    `value` and user input are first corrected/auto-completed with `convert_value` if provided,
    then validated with `check_with_message` function.
    If the user input a wrong value in interactive mode, the user gets to input a new value.
    The new valid value is saved in the openml configuration file.
    In case an invalid `value` is supplied directly (non-interactive), no changes are made.

    Parameters
    ----------
    field: str
        Field to set.
    value: str, None
        Value to field to. If `None` will ask user for input.
    check_with_message: Callable[[str], str]
        Function which validates `value` or user input, and returns either an error message if it
        is invalid, or a False-like value if `value` is valid.
    intro_message: str
        Message that is printed once if user input is requested (e.g. instructions).
    input_message: str
        Message that comes with the input prompt.
    sanitize: Union[Callable[[str], str], None]
        A function to convert user input to 'more acceptable' input, e.g. for auto-complete.
        If no correction of user input is possible, return the original value.
        If no function is provided, don't attempt to correct/auto-complete input.
    """
    if value is not None:
        if sanitize:
            value = sanitize(value)
        malformed_input = check_with_message(value)
        if malformed_input:
            print(malformed_input)
            sys.exit()
    else:
        print(intro_message)
        value = wait_until_valid_input(
            prompt=input_message,
            check=check_with_message,
            sanitize=sanitize,
        )
    verbose_set(field, value)


def upload_dataset(args: argparse.Namespace) -> None:
    """Upload a dataset from a CSV or ARFF file to OpenML."""
    import pandas as pd

    file_path = Path(args.file_path)
    if not file_path.is_file():
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(file_path)
    elif suffix == ".arff":
        import arff

        with file_path.open() as fh:
            arff_data = arff.load(fh)
        data = pd.DataFrame(
            arff_data["data"],
            columns=[attr[0] for attr in arff_data["attributes"]],
        )
    else:
        print(f"Error: Unsupported file format '{suffix}'. Supported formats: .csv, .arff")
        sys.exit(1)

    dataset = openml.datasets.create_dataset(
        name=args.name,
        description=args.description,
        creator=args.creator,
        contributor=args.contributor,
        collection_date=args.collection_date,
        language=args.language,
        licence=args.licence,
        attributes="auto",
        data=data,
        default_target_attribute=args.default_target_attribute,
        ignore_attribute=args.ignore_attribute,
        citation=args.citation or "",
        row_id_attribute=args.row_id_attribute,
        original_data_url=args.original_data_url,
        paper_url=args.paper_url,
        version_label=args.version_label,
        update_comment=args.update_comment,
    )
    dataset.publish()
    print(f"Dataset successfully uploaded. ID: {dataset.id}")
    print(f"URL: {dataset.openml_url}")


def upload_flow(args: argparse.Namespace) -> None:
    """Upload a flow from a serialized model file to OpenML."""
    import pickle

    file_path = Path(args.file_path)
    if not file_path.is_file():
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    print(
        "WARNING: Loading pickle files executes arbitrary code. "
        "Only use this with files you trust.",
    )
    with file_path.open("rb") as fh:
        model = pickle.load(fh)  # noqa: S301

    extension = openml.extensions.get_extension_by_model(model, raise_if_no_extension=True)
    assert extension is not None  # guaranteed by raise_if_no_extension=True
    flow = extension.model_to_flow(model)

    if args.name:
        flow.custom_name = args.name
    if args.description:
        flow.description = args.description

    flow.publish()
    print(f"Flow successfully uploaded. ID: {flow.flow_id}")
    print(f"URL: {flow.openml_url}")


def upload_run(args: argparse.Namespace) -> None:
    """Upload a run from a directory containing run files to OpenML."""
    directory = Path(args.file_path)
    if not directory.is_dir():
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)

    expect_model = not args.no_model
    run = openml.runs.OpenMLRun.from_filesystem(directory, expect_model=expect_model)
    run.publish()
    print(f"Run successfully uploaded. ID: {run.run_id}")
    print(f"URL: {run.openml_url}")


def upload(args: argparse.Namespace) -> None:
    """Dispatch upload subcommands."""
    if not openml.config.apikey:
        print(
            "Error: No API key configured. Set your API key with:\n"
            "  openml configure apikey\n"
            "For more information, see: "
            "https://openml.github.io/openml-python/latest/examples/Basics/"
            "introduction_tutorial/#authentication",
        )
        sys.exit(1)

    upload_functions: dict[str, Callable[[argparse.Namespace], None]] = {
        "dataset": upload_dataset,
        "flow": upload_flow,
        "run": upload_run,
    }

    if args.upload_resource not in upload_functions:
        print("Please specify a resource to upload: dataset, flow, or run.")
        sys.exit(1)

    upload_functions[args.upload_resource](args)


def configure(args: argparse.Namespace) -> None:
    """Calls the right submenu(s) to edit `args.field` in the configuration file."""
    set_functions = {
        "apikey": configure_apikey,
        "server": configure_server,
        "cachedir": configure_cachedir,
        "retry_policy": configure_retry_policy,
        "connection_n_retries": configure_connection_n_retries,
        "avoid_duplicate_runs": configure_avoid_duplicate_runs,
        "verbosity": configure_verbosity,
    }

    def not_supported_yet(_: str) -> None:
        print(f"Setting '{args.field}' is not supported yet.")

    if args.field not in ["all", "none"]:
        set_functions.get(args.field, not_supported_yet)(args.value)
    else:
        if args.value is not None:
            print(f"Can not set value ('{args.value}') when field is specified as '{args.field}'.")
            sys.exit()
        print_configuration()

    if args.field == "all":
        for set_field_function in set_functions.values():
            set_field_function(args.value)


def main() -> None:
    subroutines: dict[str, Callable[[argparse.Namespace], None]] = {
        "configure": configure,
        "upload": upload,
    }

    parser = argparse.ArgumentParser()
    # Add a global --version flag to display installed version and exit
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the OpenML version and exit",
    )
    subparsers = parser.add_subparsers(dest="subroutine")

    parser_configure = subparsers.add_parser(
        "configure",
        description="Set or read variables in your configuration file. For more help also see "
        "'https://openml.github.io/openml-python/main/usage.html#configuration'.",
    )

    configurable_fields = [
        f.name for f in fields(openml._config.OpenMLConfig) if f.name not in ["max_retries"]
    ]

    parser_configure.add_argument(
        "field",
        type=str,
        choices=[*configurable_fields, "all", "none"],
        default="all",
        nargs="?",
        help="The field you wish to edit. "
        "Choosing 'all' lets you configure all fields one by one. "
        "Choosing 'none' will print out the current configuration.",
    )

    parser_configure.add_argument(
        "value",
        type=str,
        default=None,
        nargs="?",
        help="The value to set the FIELD to.",
    )

    # --- upload subcommand ---
    parser_upload = subparsers.add_parser(
        "upload",
        description="Upload resources (datasets, flows, or runs) to OpenML.",
    )
    upload_subparsers = parser_upload.add_subparsers(dest="upload_resource")

    # upload dataset
    parser_upload_dataset = upload_subparsers.add_parser(
        "dataset",
        description="Upload a dataset from a CSV or ARFF file.",
    )
    parser_upload_dataset.add_argument(
        "file_path",
        type=str,
        help="Path to the dataset file (.csv or .arff).",
    )
    _dataset_args: list[tuple[str, str, bool]] = [
        ("--name", "Name of the dataset.", True),
        ("--description", "Description of the dataset.", True),
        ("--default_target_attribute", "The default target attribute.", False),
        ("--creator", "The person who created the dataset.", False),
        ("--contributor", "People who contributed to the dataset.", False),
        ("--collection_date", "The date the data was originally collected.", False),
        ("--language", "Language in which the data is represented.", False),
        ("--licence", "License of the data.", False),
        ("--ignore_attribute", "Attributes to exclude in modelling (comma separated).", False),
        ("--citation", "Reference(s) that should be cited.", False),
        ("--row_id_attribute", "The attribute that represents the row-id column.", False),
        ("--original_data_url", "URL to the original dataset (for derived data).", False),
        ("--paper_url", "Link to a paper describing the dataset.", False),
        ("--version_label", "Version label (e.g. date, hash).", False),
        ("--update_comment", "An explanation for when the dataset is uploaded.", False),
    ]
    for flag, help_text, required in _dataset_args:
        parser_upload_dataset.add_argument(
            flag,
            type=str,
            required=required,
            default=None,
            help=help_text,
        )

    # upload flow
    parser_upload_flow = upload_subparsers.add_parser(
        "flow",
        description="Upload a flow from a serialized model file (.pkl). "
        "WARNING: pickle files can execute arbitrary code. Only use trusted files.",
    )
    parser_upload_flow.add_argument(
        "file_path",
        type=str,
        help="Path to the serialized model file (.pkl). WARNING: only use trusted pickle files.",
    )
    parser_upload_flow.add_argument("--name", type=str, default=None, help="Custom flow name.")
    parser_upload_flow.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description of the flow.",
    )

    # upload run
    parser_upload_run = upload_subparsers.add_parser(
        "run",
        description="Upload a run from a directory containing run files.",
    )
    parser_upload_run.add_argument(
        "file_path",
        type=str,
        help="Path to directory with run files (description.xml, predictions.arff, etc.).",
    )
    parser_upload_run.add_argument(
        "--no_model",
        action="store_true",
        default=False,
        help="If set, do not require model.pkl in the run directory.",
    )

    args = parser.parse_args()
    subroutines.get(args.subroutine, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
