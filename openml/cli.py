"""" Command Line Interface for `openml` to configure its settings. """

import argparse

# from openml import config


def configure(args: argparse.Namespace):
    """ Configures the openml configuration file. """
    print("Configuring", args.file)

    # check if API key exists, if so ask to overwrite.


def main() -> None:
    subroutines = {"configure": configure}

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subroutine")

    parser_configure = subparsers.add_parser(
        "configure", description="Set or read variables in your configuration file.",
    )
    parser_configure.add_argument(
        "file", default="~/.openml/config", help="The configuration file to edit or read."
    )

    args = parser.parse_args()
    subroutines.get(args.subroutine, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
