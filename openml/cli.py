"""Command Line Interface for `openml` to configure its settings."""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlparse

import pandas as pd  # Used at runtime for CLI output formatting

from openml import config
from openml.study import functions as study_functions

if TYPE_CHECKING:
    from openml.study.study import OpenMLBenchmarkSuite, OpenMLStudy


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


def _format_studies_output(
    studies_df: pd.DataFrame,
    output_format: str,
    *,
    verbose: bool = False,
) -> None:
    """Format and print studies output based on requested format.

    Parameters
    ----------
    studies_df : pd.DataFrame
        DataFrame containing studies information
    output_format : str
        Output format: 'json', 'table', or 'list'
    verbose : bool
        Whether to show detailed information
    """
    if output_format == "json":
        # Convert to JSON format
        output = studies_df.to_json(orient="records", indent=2)
        print(output)
    elif output_format == "table":
        _format_studies_table(studies_df, verbose=verbose)
    else:  # default: simple list
        _format_studies_list(studies_df, verbose=verbose)


def _format_studies_table(studies_df: pd.DataFrame, *, verbose: bool = False) -> None:
    """Format studies as a table.

    Parameters
    ----------
    studies_df : pd.DataFrame
        DataFrame containing studies information
    verbose : bool
        Whether to show all columns
    """
    if verbose:
        print(studies_df.to_string(index=False))
    else:
        # Show only key columns for compact view
        columns_to_show = ["id", "name", "main_entity_type", "status", "creator", "creation_date"]
        available_columns = [col for col in columns_to_show if col in studies_df.columns]
        print(studies_df[available_columns].to_string(index=False))


def _format_studies_list(studies_df: pd.DataFrame, *, verbose: bool = False) -> None:
    """Format studies as a simple list.

    Parameters
    ----------
    studies_df : pd.DataFrame
        DataFrame containing studies information
    verbose : bool
        Whether to show detailed information
    """
    if verbose:
        # Verbose: show detailed info for each study
        for _, study in studies_df.iterrows():
            print(f"Study ID: {study['id']}")
            print(f"  Name: {study['name']}")
            print(f"  Type: {study.get('main_entity_type', 'N/A')}")
            print(f"  Status: {study.get('status', 'N/A')}")
            print(f"  Creator: {study.get('creator', 'N/A')}")
            if "creation_date" in study and pd.notna(study["creation_date"]):
                print(f"  Created: {study['creation_date']}")
            if "alias" in study and pd.notna(study["alias"]):
                print(f"  Alias: {study['alias']}")
            print()
    else:
        # Simple: just list study IDs and names
        for _, study in studies_df.iterrows():
            study_type = study.get("main_entity_type", "")
            type_label = " (suite)" if study_type == "task" else ""
            print(f"{study['id']}: {study['name']}{type_label}")


def studies_list(args: argparse.Namespace) -> None:
    """List studies with optional filtering.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing filtering criteria: status, uploader, type, size, offset, format
    """
    # Build filter arguments, excluding None values
    kwargs = {}
    if args.status is not None:
        kwargs["status"] = args.status
    if args.uploader is not None:
        kwargs["uploader"] = args.uploader
    if args.size is not None:
        kwargs["size"] = args.size
    if args.offset is not None:
        kwargs["offset"] = args.offset

    try:
        # Fetch based on type
        if args.type == "all":
            # Fetch both studies and suites
            studies_df = study_functions.list_studies(**kwargs)
            suites_df = study_functions.list_suites(**kwargs)
            # Combine results
            combined_df = pd.concat([studies_df, suites_df], ignore_index=True)
            _format_studies_output(combined_df, args.format, verbose=args.verbose)
        elif args.type == "study":
            # Fetch only studies (runs)
            studies_df = study_functions.list_studies(**kwargs)
            _format_studies_output(studies_df, args.format, verbose=args.verbose)
        else:  # suite
            # Fetch only suites (tasks)
            suites_df = study_functions.list_suites(**kwargs)
            _format_studies_output(suites_df, args.format, verbose=args.verbose)
    except Exception as e:  # noqa: BLE001
        print(f"Error listing studies: {e}", file=sys.stderr)
        sys.exit(1)


def _display_study_entity_counts(study: OpenMLStudy | OpenMLBenchmarkSuite) -> None:
    """Display entity counts for a study.

    Parameters
    ----------
    study : Union[OpenMLStudy, OpenMLBenchmarkSuite]
        Study or suite object
    """
    print("\nEntities:")
    if hasattr(study, "data") and study.data:
        print(f"  Datasets: {len(study.data)}")
    if hasattr(study, "tasks") and study.tasks:
        print(f"  Tasks: {len(study.tasks)}")
    if hasattr(study, "flows") and study.flows:
        print(f"  Flows: {len(study.flows)}")
    if hasattr(study, "runs") and study.runs:
        print(f"  Runs: {len(study.runs)}")
    if hasattr(study, "setups") and study.setups:
        print(f"  Setups: {len(study.setups)}")


def _display_study_entity_ids(study: OpenMLStudy | OpenMLBenchmarkSuite) -> None:
    """Display first 10 entity IDs for a study.

    Parameters
    ----------
    study : Union[OpenMLStudy, OpenMLBenchmarkSuite]
        Study or suite object
    """
    if hasattr(study, "data") and study.data:
        print(f"\nDataset IDs (first 10): {study.data[:10]}")
    if hasattr(study, "tasks") and study.tasks:
        print(f"Task IDs (first 10): {study.tasks[:10]}")
    if hasattr(study, "flows") and study.flows:
        print(f"Flow IDs (first 10): {study.flows[:10]}")
    if hasattr(study, "runs") and study.runs:
        print(f"Run IDs (first 10): {study.runs[:10]}")


def studies_info(args: argparse.Namespace) -> None:
    """Display detailed information about a specific study.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing the study_id to fetch
    """
    try:
        # Get study from server - try as study first, then as suite
        study: OpenMLStudy | OpenMLBenchmarkSuite
        try:
            study = study_functions.get_study(args.study_id)
        except Exception:  # noqa: BLE001
            # Might be a suite (benchmark suite)
            study = study_functions.get_suite(args.study_id)

        # Display study information
        print(f"Study ID: {study.study_id}")
        print(f"Name: {study.name}")
        print(f"Main Entity Type: {study.main_entity_type}")
        print(f"Status: {study.status}")

        if hasattr(study, "alias") and study.alias:
            print(f"Alias: {study.alias}")

        if study.creator:
            print(f"Creator: {study.creator}")

        if study.creation_date:
            print(f"Creation Date: {study.creation_date}")

        if hasattr(study, "benchmark_suite") and study.benchmark_suite:
            print(f"Benchmark Suite: {study.benchmark_suite}")

        # Display description
        if study.description:
            print("\nDescription:")
            print(f"  {study.description}")

        # Display entity counts and IDs
        _display_study_entity_counts(study)

        if args.verbose:
            _display_study_entity_ids(study)

    except Exception as e:  # noqa: BLE001
        print(f"Error fetching study information: {e}", file=sys.stderr)
        sys.exit(1)


def studies_search(args: argparse.Namespace) -> None:
    """Search studies by name or alias.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing the search query
    """
    try:
        # Get all studies (both types)
        kwargs = {}
        if args.status:
            kwargs["status"] = args.status

        studies_df_runs = study_functions.list_studies(**kwargs)
        studies_df_suites = study_functions.list_suites(**kwargs)

        # Combine both dataframes
        if not studies_df_runs.empty and not studies_df_suites.empty:
            all_studies = pd.concat([studies_df_runs, studies_df_suites], ignore_index=True)
        elif not studies_df_runs.empty:
            all_studies = studies_df_runs
        elif not studies_df_suites.empty:
            all_studies = studies_df_suites
        else:
            print("No studies found.")
            return

        # Search by name (case-insensitive)
        search_term = args.query.lower()
        mask = all_studies["name"].str.lower().str.contains(search_term, na=False)

        # Also search by alias if available
        if "alias" in all_studies.columns:
            mask |= all_studies["alias"].str.lower().str.contains(search_term, na=False)

        results = all_studies[mask]

        if results.empty:
            print(f"No studies found matching '{args.query}'.")
            return

        print(f"Found {len(results)} study(ies) matching '{args.query}':\n")

        # Format output
        _format_studies_output(results, args.format, verbose=args.verbose)

    except Exception as e:  # noqa: BLE001
        print(f"Error searching studies: {e}", file=sys.stderr)
        sys.exit(1)


def studies(args: argparse.Namespace) -> None:
    """Route studies subcommands to the appropriate handler.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing the subcommand and its arguments
    """
    subcommands = {
        "list": studies_list,
        "info": studies_info,
        "search": studies_search,
    }

    handler = subcommands.get(args.studies_subcommand)
    if handler:
        handler(args)
    else:
        print(f"Unknown studies subcommand: {args.studies_subcommand}")
        sys.exit(1)


def main() -> None:
    subroutines = {"configure": configure, "studies": studies}

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subroutine")

    # Configure subcommand
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

    # Studies subcommand
    parser_studies = subparsers.add_parser(
        "studies",
        description="Browse and search OpenML studies and benchmark suites from the command line.",
    )
    studies_subparsers = parser_studies.add_subparsers(dest="studies_subcommand")

    # studies list subcommand
    parser_studies_list = studies_subparsers.add_parser(
        "list",
        description="List studies/suites with optional filtering.",
        help="List studies/suites with optional filtering.",
    )
    parser_studies_list.add_argument(
        "--status",
        type=str,
        choices=["active", "in_preparation", "deactivated", "all"],
        help="Filter by status (default: active)",
    )
    parser_studies_list.add_argument(
        "--uploader",
        type=int,
        help="Filter by uploader ID",
    )
    parser_studies_list.add_argument(
        "--type",
        type=str,
        choices=["all", "study", "suite"],
        default="all",
        help="Type to list: all, study (runs), or suite (tasks) (default: all)",
    )
    parser_studies_list.add_argument(
        "--size",
        type=int,
        default=10,
        help="Number of studies to retrieve (default: 10)",
    )
    parser_studies_list.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for pagination (default: 0)",
    )
    parser_studies_list.add_argument(
        "--format",
        type=str,
        choices=["list", "table", "json"],
        default="list",
        help="Output format (default: list)",
    )
    parser_studies_list.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information",
    )

    # studies info subcommand
    parser_studies_info = studies_subparsers.add_parser(
        "info",
        description="Display detailed information about a specific study.",
        help="Display detailed information about a specific study.",
    )
    parser_studies_info.add_argument(
        "study_id",
        type=str,
        help="Study ID (numeric or alias) to fetch information for",
    )
    parser_studies_info.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional details including entity IDs",
    )

    # studies search subcommand
    parser_studies_search = studies_subparsers.add_parser(
        "search",
        description="Search studies by name or alias.",
        help="Search studies by name or alias.",
    )
    parser_studies_search.add_argument(
        "query",
        type=str,
        help="Search query (case-insensitive substring match)",
    )
    parser_studies_search.add_argument(
        "--status",
        type=str,
        choices=["active", "in_preparation", "deactivated", "all"],
        help="Filter by status (default: active)",
    )
    parser_studies_search.add_argument(
        "--format",
        type=str,
        choices=["list", "table", "json"],
        default="list",
        help="Output format (default: list)",
    )
    parser_studies_search.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information",
    )

    args = parser.parse_args()
    subroutines.get(args.subroutine, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
