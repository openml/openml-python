"""Command Line Interface for `openml` to configure its settings."""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlparse

from openml import config, datasets

if TYPE_CHECKING:
    import pandas as pd


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
    file = config.determine_config_file_path()
    header = f"File '{file}' contains (or defaults to):"
    print(header)

    max_key_length = max(map(len, config.get_config_as_dict()))
    for field, value in config.get_config_as_dict().items():
        print(f"{field.ljust(max_key_length)}: {value}")


def verbose_set(field: str, value: str) -> None:
    config.set_field_in_config_file(field, value)
    print(f"{field} set to '{value}'.")


def configure_apikey(value: str) -> None:
    def check_apikey(apikey: str) -> str:
        if len(apikey) != 32:
            return f"The key should contain 32 characters but contains {len(apikey)}."
        if not is_hex(apikey):
            return "Some characters are not hexadecimal."
        return ""

    instructions = (
        f"Your current API key is set to: '{config.apikey}'. "
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
        is_shorthand = server in ["test", "production"]
        if is_shorthand or looks_like_url(server):
            return ""
        return "Must be 'test', 'production' or a url."

    def replace_shorthand(server: str) -> str:
        if server == "test":
            return "https://test.openml.org/api/v1/xml"
        if server == "production":
            return "https://www.openml.org/api/v1/xml"
        return server

    configure_field(
        field="server",
        value=value,
        check_with_message=check_server,
        intro_message="Specify which server you wish to connect to.",
        input_message="Specify a url or use 'test' or 'production' as a shorthand: ",
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


def _format_output(
    datasets_df: pd.DataFrame,
    format: str = "table",  # noqa: A002
    verbose: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Format and print DataFrame output.

    Parameters
    ----------
    df : pd.DataFrame
        Data to display
    format : str
        Output format: 'table' or 'json'
    verbose : bool
        Include all columns if True, otherwise show subset
    """
    if datasets_df.empty:
        print("No results found.")
        return

    if not verbose and len(datasets_df.columns) > 6:
        # Show only key columns for non-verbose output
        key_cols = [
            "did",
            "name",
            "status",
            "NumberOfInstances",
            "NumberOfFeatures",
            "NumberOfClasses",
        ]
        cols_to_show = [c for c in key_cols if c in datasets_df.columns]
        datasets_df = datasets_df[cols_to_show]

    if format == "json":
        print(datasets_df.to_json(orient="records", indent=2))
    else:  # table format
        print(datasets_df.to_string(index=False))


def datasets_list(args: argparse.Namespace) -> None:
    """List datasets with optional filtering."""
    # Build filter parameters
    kwargs = {}
    if args.tag:
        kwargs["tag"] = args.tag
    if args.status:
        kwargs["status"] = args.status
    if args.data_name:
        kwargs["data_name"] = args.data_name
    if args.number_instances:
        kwargs["number_instances"] = args.number_instances
    if args.number_features:
        kwargs["number_features"] = args.number_features
    if args.number_classes:
        kwargs["number_classes"] = args.number_classes

    # Fetch datasets
    try:
        datasets_df = datasets.list_datasets(
            offset=args.offset,
            size=args.size,
            **kwargs,
        )
        _format_output(datasets_df, format=args.format, verbose=args.verbose)
    except Exception as e:  # noqa: BLE001
        print(f"Error listing datasets: {e}", file=sys.stderr)
        sys.exit(1)


def datasets_info(args: argparse.Namespace) -> None:
    """Display detailed information about a specific dataset."""
    try:
        dataset = datasets.get_dataset(
            args.dataset_id,
            download_data=False,
            download_qualities=True,
            download_features_meta_data=True,
        )

        # Print basic information
        print(f"Dataset ID: {dataset.dataset_id}")
        print(f"Name: {dataset.name}")
        print(f"Version: {dataset.version}")
        print(f"Format: {dataset.format}")
        print(f"Upload Date: {dataset.upload_date}")
        if dataset.visibility:
            print(f"Visibility: {dataset.visibility}")
        description = dataset.description or "No description available"
        if len(description) > 200:
            print(f"Description: {description[:200]}...")
        else:
            print(f"Description: {description}")

        # Print qualities if available
        if dataset.qualities:
            print("\nQualities:")
            for key, value in dataset.qualities.items():
                print(f"  {key}: {value}")

        # Print feature information
        if dataset.features:
            print(f"\nFeatures ({len(dataset.features)}):")
            for feat_name, feat in list(dataset.features.items())[:10]:  # Show first 10
                print(f"  - {feat_name} ({feat.data_type})")
            if len(dataset.features) > 10:
                print(f"  ... and {len(dataset.features) - 10} more")

    except Exception as e:  # noqa: BLE001
        print(f"Error fetching dataset info: {e}", file=sys.stderr)
        sys.exit(1)


def datasets_search(args: argparse.Namespace) -> None:
    """Search datasets by name (case-insensitive)."""
    try:
        # First try exact match with the API
        datasets_df = datasets.list_datasets(data_name=args.query, size=args.size or 100)

        # If no exact match, do case-insensitive client-side filtering
        if datasets_df.empty:
            all_datasets = datasets.list_datasets(size=1000)  # Get more datasets
            if "name" in all_datasets.columns:
                mask = all_datasets["name"].str.contains(args.query, case=False, na=False)
                datasets_df = all_datasets[mask].head(args.size or 20)

        if datasets_df.empty:
            print(f"No datasets found matching '{args.query}'")
        else:
            print(f"Found {len(datasets_df)} dataset(s) matching '{args.query}':\n")
            _format_output(datasets_df, format=args.format, verbose=args.verbose)
    except Exception as e:  # noqa: BLE001
        print(f"Error searching datasets: {e}", file=sys.stderr)
        sys.exit(1)


def datasets_handler(args: argparse.Namespace) -> None:
    """Route datasets subcommands to appropriate handlers."""
    actions = {
        "list": datasets_list,
        "info": datasets_info,
        "search": datasets_search,
    }
    action_func = actions.get(args.datasets_action)
    if action_func:
        action_func(args)
    else:
        print("Please specify a datasets action: list, info, or search")
        sys.exit(1)


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
    subroutines = {"configure": configure, "datasets": datasets_handler}

    parser = argparse.ArgumentParser(prog="openml")
    subparsers = parser.add_subparsers(dest="subroutine")

    # Configure subparser (existing)
    parser_configure = subparsers.add_parser(
        "configure",
        description="Set or read variables in your configuration file. For more help also see "
        "'https://openml.github.io/openml-python/main/usage.html#configuration'.",
    )
    configurable_fields = [f for f in config._defaults if f not in ["max_retries"]]
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

    # Datasets subparser (NEW)
    parser_datasets = subparsers.add_parser(
        "datasets",
        description="Browse and search OpenML datasets from the command line.",
    )
    datasets_subparsers = parser_datasets.add_subparsers(dest="datasets_action")

    # datasets list
    parser_datasets_list = datasets_subparsers.add_parser(
        "list",
        help="List datasets with optional filtering",
    )
    parser_datasets_list.add_argument("--offset", type=int, help="Number of datasets to skip")
    parser_datasets_list.add_argument("--size", type=int, help="Maximum number of datasets to show")
    parser_datasets_list.add_argument("--tag", type=str, help="Filter by tag")
    parser_datasets_list.add_argument(
        "--status",
        type=str,
        choices=["active", "in_preparation", "deactivated"],
        help="Filter by status",
    )
    parser_datasets_list.add_argument("--data-name", type=str, help="Filter by dataset name")
    parser_datasets_list.add_argument(
        "--number-instances",
        type=str,
        help="Filter by number of instances (e.g., '100..1000')",
    )
    parser_datasets_list.add_argument(
        "--number-features",
        type=str,
        help="Filter by number of features",
    )
    parser_datasets_list.add_argument(
        "--number-classes",
        type=str,
        help="Filter by number of classes",
    )
    parser_datasets_list.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    parser_datasets_list.add_argument(
        "--verbose",
        action="store_true",
        help="Show all columns",
    )

    # datasets info
    parser_datasets_info = datasets_subparsers.add_parser(
        "info",
        help="Display detailed information about a specific dataset",
    )
    parser_datasets_info.add_argument("dataset_id", type=str, help="Dataset ID or name")

    # datasets search
    parser_datasets_search = datasets_subparsers.add_parser(
        "search",
        help="Search datasets by name (case-insensitive)",
    )
    parser_datasets_search.add_argument("query", type=str, help="Search query")
    parser_datasets_search.add_argument("--size", type=int, help="Maximum number of results")
    parser_datasets_search.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    parser_datasets_search.add_argument(
        "--verbose",
        action="store_true",
        help="Show all columns",
    )

    args = parser.parse_args()
    subroutines.get(args.subroutine, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
