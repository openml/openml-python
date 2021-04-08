"""" Command Line Interface for `openml` to configure its settings. """

import argparse
import string
from typing import Union, Callable


from openml import config


def is_hex(string_: str) -> bool:
    return all(c in string.hexdigits for c in string_)


def looks_like_api_key(apikey: str) -> bool:
    return len(apikey) == 32 and is_hex(apikey)


def wait_until_valid_input(
    prompt: str,
    check: Callable[[str], bool],
    error_message: Union[str, Callable[[str], str]] = "That is not a valid response.",
) -> str:
    """  Asks `prompt` until an input is received which returns True for `check`.

    Parameters
    ----------
    prompt: str
        message to display
    check: Callable[[str], bool]
        function to call with the given input, should return true only if the input is valid.
    error_message: Union[str, Callable[[str], str]
        a message to display on invalid input, or a `str`->`str` function that can give feedback
        specific to the error.

    Returns
    -------

    """

    response = input(prompt)
    while not check(response):
        if isinstance(error_message, str):
            print(error_message)
        else:
            print(error_message(response), end="\n\n")
        response = input(prompt)

    return response


def print_configuration():
    file = config.determine_config_file_path()
    header = f"File '{file}' contains (or defaults to):"
    print(header)

    max_key_length = max(map(len, config.get_config_as_dict()))
    for field, value in config.get_config_as_dict().items():
        print(f"{field.ljust(max_key_length)}: {value}")


def configure_apikey() -> None:
    print(f"\nYour current API key is set to: '{config.apikey}'")
    print("You can get an API key at https://new.openml.org")
    print("You must create an account if you don't have one yet.")
    print("  1. Log in with the account.")
    print("  2. Navigate to the profile page (top right circle > Your Profile). ")
    print("  3. Click the API Key button to reach the page with your API key.")
    print("If you have any difficulty following these instructions, please let us know on Github.")

    def apikey_error(apikey: str) -> str:
        if len(apikey) != 32:
            return f"The key should contain 32 characters but contains {len(apikey)}."
        if not is_hex(apikey):
            return "Some characters are not hexadecimal."
        return "This does not look like an API key."

    response = wait_until_valid_input(
        prompt="Please enter your API key:", check=looks_like_api_key, error_message=apikey_error,
    )

    config.set_field_in_config_file("apikey", response)
    print("Key set.")


def configure(args: argparse.Namespace):
    """ Configures the openml configuration file. """
    print_configuration()
    set_functions = {
        "apikey": configure_apikey,
    }
    set_functions.get(args.field, quit)()


def main() -> None:
    subroutines = {"configure": configure}

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subroutine")

    parser_configure = subparsers.add_parser(
        "configure", description="Set or read variables in your configuration file.",
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

    args = parser.parse_args()
    subroutines.get(args.subroutine, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
