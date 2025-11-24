"""Command Line Interface for `openml` to configure its settings."""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlparse

from openml import config
from openml.runs import functions as run_functions

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


def _format_runs_output(
    runs_df: pd.DataFrame,
    output_format: str,
    *,
    verbose: bool = False,
) -> None:
    """Format and print runs output based on requested format.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame containing runs information
    output_format : str
        Output format: 'json', 'table', or 'list'
    verbose : bool
        Whether to show detailed information
    """
    if output_format == "json":
        # Convert to JSON format
        output = runs_df.to_json(orient="records", indent=2)
        print(output)
    elif output_format == "table":
        _format_runs_table(runs_df, verbose=verbose)
    else:  # default: simple list
        _format_runs_list(runs_df, verbose=verbose)


def _format_runs_table(runs_df: pd.DataFrame, *, verbose: bool = False) -> None:
    """Format runs as a table.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame containing runs information
    verbose : bool
        Whether to show all columns
    """
    if verbose:
        print(runs_df.to_string(index=False))
    else:
        # Show only key columns for compact view
        columns_to_show = ["run_id", "task_id", "flow_id", "uploader", "upload_time"]
        available_columns = [col for col in columns_to_show if col in runs_df.columns]
        print(runs_df[available_columns].to_string(index=False))


def _format_runs_list(runs_df: pd.DataFrame, *, verbose: bool = False) -> None:
    """Format runs as a simple list.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame containing runs information
    verbose : bool
        Whether to show detailed information
    """
    if verbose:
        # Verbose: show detailed info for each run
        for _, run in runs_df.iterrows():
            print(f"Run ID: {run['run_id']}")
            print(f"  Task ID: {run['task_id']}")
            print(f"  Flow ID: {run['flow_id']}")
            print(f"  Setup ID: {run['setup_id']}")
            print(f"  Uploader: {run['uploader']}")
            print(f"  Upload Time: {run['upload_time']}")
            if run.get("error_message"):
                print(f"  Error: {run['error_message']}")
            print()
    else:
        # Simple: just list run IDs
        for run_id in runs_df["run_id"]:
            print(f"{run_id}: Task {runs_df[runs_df['run_id'] == run_id]['task_id'].iloc[0]}")


def runs_list(args: argparse.Namespace) -> None:
    """List runs with optional filtering.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing filtering criteria: task, flow, uploader, tag, size, offset, format
    """
    # Build filter arguments, excluding None values
    kwargs = {}
    if args.task is not None:
        kwargs["task"] = [args.task]
    if args.flow is not None:
        kwargs["flow"] = [args.flow]
    if args.uploader is not None:
        kwargs["uploader"] = [args.uploader]
    if args.tag is not None:
        kwargs["tag"] = args.tag
    if args.size is not None:
        kwargs["size"] = args.size
    if args.offset is not None:
        kwargs["offset"] = args.offset

    try:
        # Get runs from server
        runs_df = run_functions.list_runs(**kwargs)  # type: ignore[arg-type]

        if runs_df.empty:
            print("No runs found matching the criteria.")
            return

        # Format output based on requested format
        _format_runs_output(runs_df, args.format, verbose=args.verbose)

    except Exception as e:  # noqa: BLE001
        print(f"Error listing runs: {e}", file=sys.stderr)
        sys.exit(1)


def _print_run_evaluations(run: object) -> None:
    """Print evaluation information for a run.

    Parameters
    ----------
    run : OpenMLRun
        The run object containing evaluation data
    """
    # Display evaluations if available
    if hasattr(run, "evaluations") and run.evaluations:
        print("\nEvaluations:")
        for measure, value in run.evaluations.items():
            print(f"  {measure}: {value}")

    # Display fold evaluations if available (summary)
    if hasattr(run, "fold_evaluations") and run.fold_evaluations:
        print("\nFold Evaluations (Summary):")
        for measure, repeats in run.fold_evaluations.items():
            # Calculate average across all folds and repeats
            all_values = []
            for repeat_dict in repeats.values():
                all_values.extend(repeat_dict.values())
            if all_values:
                avg_value = sum(all_values) / len(all_values)
                print(f"  {measure}: {avg_value:.4f} (avg over {len(all_values)} folds)")


def runs_info(args: argparse.Namespace) -> None:
    """Display detailed information about a specific run.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing the run_id to fetch
    """
    try:
        # Get run from server
        run = run_functions.get_run(args.run_id)

        # Display run information
        print(f"Run ID: {run.run_id}")
        print(f"Task ID: {run.task_id}")
        print(f"Task Type: {run.task_type}")
        print(f"Flow ID: {run.flow_id}")
        print(f"Flow Name: {run.flow_name}")
        print(f"Setup ID: {run.setup_id}")
        print(f"Dataset ID: {run.dataset_id}")
        print(f"Uploader: {run.uploader_name} (ID: {run.uploader})")

        # Display parameter settings if available
        if run.parameter_settings:
            print("\nParameter Settings:")
            for param in run.parameter_settings:
                component = param.get("oml:component", "")
                name = param.get("oml:name", "")
                value = param.get("oml:value", "")
                if component:
                    print(f"  {component}.{name}: {value}")
                else:
                    print(f"  {name}: {value}")

        # Display evaluations
        _print_run_evaluations(run)

        # Display tags if available
        if run.tags:
            print(f"\nTags: {', '.join(run.tags)}")

        # Display predictions URL if available
        if run.predictions_url:
            print(f"\nPredictions URL: {run.predictions_url}")

        # Display output files if available
        if run.output_files:
            print("\nOutput Files:")
            for file_name, file_id in run.output_files.items():
                print(f"  {file_name}: {file_id}")

    except Exception as e:  # noqa: BLE001
        print(f"Error fetching run information: {e}", file=sys.stderr)
        sys.exit(1)


def runs_download(args: argparse.Namespace) -> None:
    """Download a run and cache it locally.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing the run_id to download
    """
    try:
        # Get run from server (this will download and cache it)
        run = run_functions.get_run(args.run_id, ignore_cache=True)

        print(f"Successfully downloaded run {run.run_id}")
        print(f"Task ID: {run.task_id}")
        print(f"Flow ID: {run.flow_id}")
        print(f"Dataset ID: {run.dataset_id}")

        # Display cache location
        cache_dir = config.get_cache_directory()
        run_cache_dir = Path(cache_dir) / "runs" / str(run.run_id)
        if run_cache_dir.exists():
            print(f"\nRun cached at: {run_cache_dir}")
            # List cached files
            cached_files = list(run_cache_dir.iterdir())
            if cached_files:
                print("Cached files:")
                for file in cached_files:
                    print(f"  - {file.name}")

        if run.predictions_url:
            print(f"\nPredictions available at: {run.predictions_url}")

    except Exception as e:  # noqa: BLE001
        print(f"Error downloading run: {e}", file=sys.stderr)
        sys.exit(1)


def runs(args: argparse.Namespace) -> None:
    """Route runs subcommands to the appropriate handler.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing the subcommand and its arguments
    """
    subcommands = {
        "list": runs_list,
        "info": runs_info,
        "download": runs_download,
    }

    handler = subcommands.get(args.runs_subcommand)
    if handler:
        handler(args)
    else:
        print(f"Unknown runs subcommand: {args.runs_subcommand}")
        sys.exit(1)


def main() -> None:
    subroutines = {"configure": configure, "runs": runs}

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

    # Runs subcommand
    parser_runs = subparsers.add_parser(
        "runs",
        description="Browse and search OpenML runs from the command line.",
    )
    runs_subparsers = parser_runs.add_subparsers(dest="runs_subcommand")

    # runs list subcommand
    parser_runs_list = runs_subparsers.add_parser(
        "list",
        description="List runs with optional filtering.",
        help="List runs with optional filtering.",
    )
    parser_runs_list.add_argument(
        "--task",
        type=int,
        help="Filter by task ID",
    )
    parser_runs_list.add_argument(
        "--flow",
        type=int,
        help="Filter by flow ID",
    )
    parser_runs_list.add_argument(
        "--uploader",
        type=str,
        help="Filter by uploader name or ID",
    )
    parser_runs_list.add_argument(
        "--tag",
        type=str,
        help="Filter by tag",
    )
    parser_runs_list.add_argument(
        "--size",
        type=int,
        default=10,
        help="Number of runs to retrieve (default: 10)",
    )
    parser_runs_list.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for pagination (default: 0)",
    )
    parser_runs_list.add_argument(
        "--format",
        type=str,
        choices=["list", "table", "json"],
        default="list",
        help="Output format (default: list)",
    )
    parser_runs_list.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information",
    )

    # runs info subcommand
    parser_runs_info = runs_subparsers.add_parser(
        "info",
        description="Display detailed information about a specific run.",
        help="Display detailed information about a specific run.",
    )
    parser_runs_info.add_argument(
        "run_id",
        type=int,
        help="Run ID to fetch information for",
    )

    # runs download subcommand
    parser_runs_download = runs_subparsers.add_parser(
        "download",
        description="Download a run and cache it locally.",
        help="Download a run and cache it locally.",
    )
    parser_runs_download.add_argument(
        "run_id",
        type=int,
        help="Run ID to download",
    )

    args = parser.parse_args()
    subroutines.get(args.subroutine, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
