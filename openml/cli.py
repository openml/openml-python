"""" Command Line Interface for `openml` to configure its settings. """

import argparse

# import os
# import pathlib
import string
from typing import Callable  # Union, Callable
from urllib.parse import urlparse


from openml import config


def is_hex(string_: str) -> bool:
    return all(c in string.hexdigits for c in string_)


def looks_like_url(url: str) -> bool:
    # There's no thorough url parser, but we only seem to use netloc.
    try:
        return bool(urlparse(url).netloc)
    except Exception:
        return False


def wait_until_valid_input(prompt: str, check: Callable[[str], str],) -> str:
    """  Asks `prompt` until an input is received which returns True for `check`.

    Parameters
    ----------
    prompt: str
        message to display
    check: Callable[[str], str]
        function to call with the given input, that provides an error message if the input is not
        valid otherwise, and False-like otherwise.

    Returns
    -------
    valid input

    """

    response = input(prompt)
    error_message = check(response)
    while error_message:
        print(error_message, end="\n\n")
        response = input(prompt)
        error_message = check(response)

    return response


def print_configuration():
    file = config.determine_config_file_path()
    header = f"File '{file}' contains (or defaults to):"
    print(header)

    max_key_length = max(map(len, config.get_config_as_dict()))
    for field, value in config.get_config_as_dict().items():
        print(f"{field.ljust(max_key_length)}: {value}")


def verbose_set(field, value):
    config.set_field_in_config_file(field, value)
    print(f"{field} set to '{value}'.")


def configure_apikey(value: str) -> None:
    def check_apikey(apikey: str) -> str:
        if len(apikey) != 32:
            return f"The key should contain 32 characters but contains {len(apikey)}."
        if not is_hex(apikey):
            return "Some characters are not hexadecimal."
        return ""

    if value is not None:
        apikey_malformed = check_apikey(value)
        if apikey_malformed:
            print(apikey_malformed)
            quit(-1)
    else:
        print(f"\nYour current API key is set to: '{config.apikey}'")
        print("You can get an API key at https://new.openml.org")
        print("You must create an account if you don't have one yet.")
        print("  1. Log in with the account.")
        print("  2. Navigate to the profile page (top right circle > Your Profile). ")
        print("  3. Click the API Key button to reach the page with your API key.")
        print("If you have any difficulty following these instructions, let us know on Github.")

        value = wait_until_valid_input(prompt="Please enter your API key:", check=check_apikey,)
    verbose_set("apikey", value)


def configure_server(value: str) -> None:
    def check_server(server: str) -> str:
        is_shorthand = server in ["test", "production"]
        if is_shorthand or looks_like_url(server):
            return ""
        return "Must be 'test', 'production' or a url."

    if value is not None:
        server_malformed = check_server(value)
        if server_malformed:
            print(server_malformed)
            quit(-1)
    else:
        print("\nSpecify which server you wish to connect to.")
        value = wait_until_valid_input(
            prompt="Specify a url or use 'test' or 'production' as a shorthand:",
            check=check_server,
        )

    if value == "test":
        value = "https://test.openml.org/api/v1/xml"
    elif value == "production":
        value = "https://www.openml.org/api/v1/xml"

    verbose_set("server", value)


# def configure_cachedir(value: str) -> None:
#     def is_valid_cachedir(path: str) -> bool:
#         return True
#
#     def cache_dir_error(path: str) -> str:
#         p = pathlib.Path(path)
#         if p.is_file():
#             return f"'{path}' is a file, not a directory."
#         expanded = p.expanduser()
#         if not expanded.is_absolute():
#             return f"'{path}' is not absolute (even after expanding '~')."
#         if expanded.exists():
#             try:
#                 os.mkdir(expanded)
#             except PermissionError:
#               return f"'{path}' does not exist and there are not enough permissions to create it."
#         return "is not valid"
#
#     if value is not None:
#         if not is_valid_cachedir(value):
#             print(cache_dir_error(value))
#             quit(-1)
#     else:
#         print("\nConfiguring the cache directory. It can not be a relative path.")
#         value = wait_until_valid_input(
#             prompt="Specify the directory to use (or create) as cache directory:",
#             check=is_valid_cachedir,
#         )
#
#     print("NOTE: Data from your old cache directory is not moved over.")
#     verbose_set("cachedir", value)


def configure(args: argparse.Namespace):
    """ Calls the right submenu(s) to edit `args.field` in the configuration file. """
    set_functions = {
        "apikey": configure_apikey,
        "server": configure_server,
    }

    def not_supported_yet(_):
        print(f"Setting '{args.field}' is not supported yet.")

    if args.field not in ["all", "none"]:
        set_functions.get(args.field, not_supported_yet)(args.value)
    else:
        if args.value is not None:
            print(f"Can not set value ('{args.value}') when field is specified as '{args.field}'.")
            quit()
        print_configuration()

    if args.field == "all":
        for set_field_function in set_functions.values():
            set_field_function(args.value)


def main() -> None:
    subroutines = {"configure": configure}

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subroutine")

    parser_configure = subparsers.add_parser(
        "configure",
        description="Set or read variables in your configuration file. For more help also see "
        "'https://openml.github.io/openml-python/master/usage.html#configuration'.",
    )

    parser_configure.add_argument(
        "field",
        type=str,
        choices=[*config._defaults.keys(), "all", "none"],
        default="all",
        nargs="?",
        help="The field you wish to edit."
        "Choosing 'all' lets you configure all fields one by one."
        "Choosing 'none' will print out the current configuration.",
    )

    parser_configure.add_argument(
        "value", type=str, default=None, nargs="?", help="The value to set the FIELD to.",
    )

    args = parser.parse_args()
    subroutines.get(args.subroutine, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
