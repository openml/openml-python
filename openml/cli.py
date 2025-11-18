"""Command Line Interface for `openml` to configure settings and browse resources."""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import openml
from openml import config


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


def flows_list(args: argparse.Namespace) -> None:
    """List flows from OpenML with optional filtering."""
    try:
        flows_df = openml.flows.list_flows(
            offset=args.offset,
            size=args.size,
            tag=args.tag,
            uploader=args.uploader,
        )

        if flows_df.empty:
            print("No flows found matching the criteria.")
            return

        # Display flows in a readable format
        if args.format == "table":
            # Print as a formatted table
            print(f"\nFound {len(flows_df)} flow(s):\n")
            print(flows_df.to_string(index=False))
        else:
            # Print in a more compact format
            print(f"\nFound {len(flows_df)} flow(s):\n")
            for _, row in flows_df.iterrows():
                print(f"ID: {row['id']:>6} | {row['name']:<40} | Version: {row['version']}")
                if args.verbose:
                    print(f"      Full Name: {row['full_name']}")
                    print(f"      External Version: {row['external_version']}")
                    print(f"      Uploader: {row['uploader']}")
                    print()

    except Exception as e:  # noqa: BLE001
        print(f"Error listing flows: {e}", file=sys.stderr)
        sys.exit(1)


def flows_info(args: argparse.Namespace) -> None:
    """Display detailed information about a specific flow."""
    try:
        flow_id = int(args.flow_id)
        flow = openml.flows.get_flow(flow_id)

        # Display flow information using its __repr__ method
        print(flow)

        # Additional information
        if flow.parameters:
            print("\nParameters:")
            for param_name, param_value in flow.parameters.items():
                meta_info = flow.parameters_meta_info.get(param_name, {})
                param_desc = meta_info.get("description", "No description")
                param_type = meta_info.get("data_type", "unknown")
                print(f"  {param_name}: {param_value} ({param_type})")
                if param_desc != "No description":
                    print(f"    Description: {param_desc}")

        if flow.components:
            print(f"\nComponents: {len(flow.components)}")
            for comp_name, comp_flow in flow.components.items():
                comp_id = comp_flow.flow_id if comp_flow.flow_id else "Not uploaded"
                print(f"  {comp_name}: Flow ID {comp_id}")

        if flow.tags:
            print(f"\nTags: {', '.join(flow.tags)}")

    except ValueError:
        print(
            f"Error: '{args.flow_id}' is not a valid flow ID. Please provide a number.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"Error retrieving flow information: {e}", file=sys.stderr)
        sys.exit(1)


def flows_search(args: argparse.Namespace) -> None:
    """Search for flows by name."""
    try:
        # Get all flows (or a reasonable subset)
        flows_df = openml.flows.list_flows(
            offset=0,
            size=args.max_results if args.max_results else 1000,
            tag=args.tag,
        )

        if flows_df.empty:
            print("No flows found.")
            return

        # Filter by search query (case-insensitive)
        query_lower = args.query.lower()
        matching_flows = flows_df[
            flows_df["name"].str.lower().str.contains(query_lower, na=False)
            | flows_df["full_name"].str.lower().str.contains(query_lower, na=False)
        ]

        if matching_flows.empty:
            print(f"No flows found matching '{args.query}'.")
            return

        print(f"\nFound {len(matching_flows)} flow(s) matching '{args.query}':\n")
        for _, row in matching_flows.iterrows():
            print(f"ID: {row['id']:>6} | {row['name']:<40} | Version: {row['version']}")
            if args.verbose:
                print(f"      Full Name: {row['full_name']}")
                print(f"      External Version: {row['external_version']}")
                print(f"      Uploader: {row['uploader']}")
                print()

    except Exception as e:  # noqa: BLE001
        print(f"Error searching flows: {e}", file=sys.stderr)
        sys.exit(1)


def flows(args: argparse.Namespace) -> None:
    """Handle flow subcommands."""
    subcommands = {
        "list": flows_list,
        "info": flows_info,
        "search": flows_search,
    }

    if args.flows_subcommand in subcommands:
        subcommands[args.flows_subcommand](args)
    else:
        print(f"Unknown flows subcommand: {args.flows_subcommand}")
        sys.exit(1)


def datasets_list(args: argparse.Namespace) -> None:
    """List datasets from OpenML with optional filtering."""
    try:
        datasets_df = openml.datasets.list_datasets(
            offset=args.offset,
            size=args.size,
            status=args.status,
            tag=args.tag,
            data_name=args.name,
        )

        if datasets_df.empty:
            print("No datasets found matching the criteria.")
            return

        if args.format == "table":
            print(f"\nFound {len(datasets_df)} dataset(s):\n")
            print(datasets_df.to_string(index=False))
        else:
            print(f"\nFound {len(datasets_df)} dataset(s):\n")
            for _, row in datasets_df.iterrows():
                name = row.get("name", "unknown")
                did = row.get("did", row.get("id", "unknown"))
                status = row.get("status", "unknown")
                print(f"ID: {did:>6} | {name:<40} | Status: {status}")
                if args.verbose:
                    format_ = row.get("format", "unknown")
                    version = row.get("version", "unknown")
                    print(f"      Format: {format_}")
                    print(f"      Version: {version}")
                    print()

    except Exception as e:  # noqa: BLE001
        print(f"Error listing datasets: {e}", file=sys.stderr)
        sys.exit(1)


def datasets_info(args: argparse.Namespace) -> None:
    """Display detailed information about a specific dataset."""
    try:
        dataset_id = int(args.dataset_id)
        dataset = openml.datasets.get_dataset(dataset_id, download_data=False)

        metadata = [
            ("Dataset ID", dataset.dataset_id),
            ("Name", dataset.name),
            ("Version", dataset.version),
            ("Format", dataset.format),
            ("Creator", dataset.creator),
            ("Collected", dataset.collection_date),
            ("Citation", dataset.citation),
        ]

        for label, field_value in metadata:
            if field_value:
                print(f"{label:<10}: {field_value}")

        if dataset.description:
            print("\nDescription:\n")
            print(dataset.description)

        if dataset.qualities:
            print("\nQualities:")
            for key, quality_value in sorted(dataset.qualities.items()):
                print(f"  {key}: {quality_value}")

    except ValueError:
        print(
            f"Error: '{args.dataset_id}' is not a valid dataset ID. Please provide a number.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"Error retrieving dataset information: {e}", file=sys.stderr)
        sys.exit(1)


def datasets_search(args: argparse.Namespace) -> None:
    """Search for datasets by name."""
    try:
        datasets_df = openml.datasets.list_datasets(
            offset=0,
            size=args.max_results if args.max_results else 1000,
            tag=args.tag,
        )

        if datasets_df.empty:
            print("No datasets found.")
            return

        query_lower = args.query.lower()
        name_series = datasets_df["name"].astype(str).str.lower()
        matching = datasets_df[name_series.str.contains(query_lower, na=False)]

        if matching.empty:
            print(f"No datasets found matching '{args.query}'.")
            return

        print(f"\nFound {len(matching)} dataset(s) matching '{args.query}':\n")
        for _, row in matching.iterrows():
            name = row.get("name", "unknown")
            did = row.get("did", row.get("id", "unknown"))
            status = row.get("status", "unknown")
            print(f"ID: {did:>6} | {name:<40} | Status: {status}")
            if args.verbose:
                format_ = row.get("format", "unknown")
                version = row.get("version", "unknown")
                print(f"      Format: {format_}")
                print(f"      Version: {version}")
                print()

    except Exception as e:  # noqa: BLE001
        print(f"Error searching datasets: {e}", file=sys.stderr)
        sys.exit(1)


def datasets(args: argparse.Namespace) -> None:
    """Handle dataset subcommands."""
    subcommands = {
        "list": datasets_list,
        "info": datasets_info,
        "search": datasets_search,
    }

    if args.datasets_subcommand in subcommands:
        subcommands[args.datasets_subcommand](args)
    else:
        print(f"Unknown datasets subcommand: {args.datasets_subcommand}")
        sys.exit(1)


def main() -> None:
    subroutines = {"configure": configure, "flows": flows, "datasets": datasets}

    parser = argparse.ArgumentParser(
        description="OpenML Python CLI - Access OpenML datasets, tasks, flows, and more.",
    )
    subparsers = parser.add_subparsers(dest="subroutine", help="Available commands")

    # Configure command
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

    # Flows command
    parser_flows = subparsers.add_parser(
        "flows",
        description="Browse and search OpenML flows.",
    )

    flows_subparsers = parser_flows.add_subparsers(
        dest="flows_subcommand",
        help="Available flow subcommands",
    )

    # Flows list command
    parser_flows_list = flows_subparsers.add_parser(
        "list",
        description="List flows from OpenML.",
    )
    parser_flows_list.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Number of flows to skip (for pagination).",
    )
    parser_flows_list.add_argument(
        "--size",
        type=int,
        default=None,
        help="Maximum number of flows to return.",
    )
    parser_flows_list.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filter flows by tag.",
    )
    parser_flows_list.add_argument(
        "--uploader",
        type=str,
        default=None,
        help="Filter flows by uploader ID.",
    )
    parser_flows_list.add_argument(
        "--format",
        type=str,
        choices=["table", "compact"],
        default="compact",
        help="Output format (default: compact).",
    )
    parser_flows_list.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each flow.",
    )

    # Flows info command
    parser_flows_info = flows_subparsers.add_parser(
        "info",
        description="Display detailed information about a specific flow.",
    )
    parser_flows_info.add_argument(
        "flow_id",
        type=str,
        help="The ID of the flow to display.",
    )

    # Flows search command
    parser_flows_search = flows_subparsers.add_parser(
        "search",
        description="Search for flows by name.",
    )
    parser_flows_search.add_argument(
        "query",
        type=str,
        help="Search query (searches in flow names).",
    )
    parser_flows_search.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum number of results to search through (default: 1000).",
    )
    parser_flows_search.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filter flows by tag before searching.",
    )
    parser_flows_search.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each flow.",
    )

    # Datasets command
    parser_datasets = subparsers.add_parser(
        "datasets",
        description="Browse and search OpenML datasets.",
    )

    datasets_subparsers = parser_datasets.add_subparsers(
        dest="datasets_subcommand",
        help="Available dataset subcommands",
    )

    parser_datasets_list = datasets_subparsers.add_parser(
        "list",
        description="List datasets from OpenML.",
    )
    parser_datasets_list.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Number of datasets to skip (for pagination).",
    )
    parser_datasets_list.add_argument(
        "--size",
        type=int,
        default=None,
        help="Maximum number of datasets to return.",
    )
    parser_datasets_list.add_argument(
        "--status",
        type=str,
        default=None,
        help="Filter datasets by status (e.g., active).",
    )
    parser_datasets_list.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filter datasets by tag.",
    )
    parser_datasets_list.add_argument(
        "--name",
        type=str,
        default=None,
        help="Filter datasets by (partial) name.",
    )
    parser_datasets_list.add_argument(
        "--format",
        type=str,
        choices=["table", "compact"],
        default="compact",
        help="Output format (default: compact).",
    )
    parser_datasets_list.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each dataset.",
    )

    parser_datasets_info = datasets_subparsers.add_parser(
        "info",
        description="Display detailed information about a specific dataset.",
    )
    parser_datasets_info.add_argument(
        "dataset_id",
        type=str,
        help="The ID of the dataset to display.",
    )

    parser_datasets_search = datasets_subparsers.add_parser(
        "search",
        description="Search for datasets by name.",
    )
    parser_datasets_search.add_argument(
        "query",
        type=str,
        help="Search query (searches in dataset names).",
    )
    parser_datasets_search.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum number of results to search through (default: 1000).",
    )
    parser_datasets_search.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filter datasets by tag before searching.",
    )
    parser_datasets_search.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each dataset.",
    )

    args = parser.parse_args()
    if args.subroutine:
        subroutines.get(args.subroutine, lambda _: parser.print_help())(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
