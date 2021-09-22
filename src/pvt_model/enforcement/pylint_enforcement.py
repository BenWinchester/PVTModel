#!/usr/bin/python3.7
########################################################################################
# enforcement/pylint_enforcement.py
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
Script for ensuring that all pylint errors are declared correctly in the pylint yaml file.

All uses of the `pylint: disable` pylint declaration must be declared in pylint_enforcement.yaml
in the enforcement folder.

"""

import os
import subprocess

from dataclasses import dataclass

import yaml

# String required in all enforcement entries.
JUSTIFICATION_STRING = "justification"
# Path to the YAML enforcement file.
YAML_ENFORCEMENT_FILE = os.path.join("enforcement", "pylint_enforcement.yaml")


class EnforcementError(Exception):
    """
    Raised when an enforcement error occurs.

    """

    def __init__(self, msg: str) -> None:
        """
        Instantiate a :class:`EnforcementError` instance.

        """

        super().__init__(
            f"Not all pylint 'pylint: disable' uses were declared in the yaml file: {msg}"
        )


@dataclass
class PylintDisable:
    """
    Represents a single use of the "pylint: disable" declaration.

    .. attribute:: filename
        The name of the file containing the usage.

    .. attribute:: flag
        The flag disabled.

    .. attribute:: line
        The line on which the flag is used.

    """

    filename: str
    flag: str
    line: int

    def __eq__(self, other) -> bool:
        """
        Determines if two instances are equal.

        """

        return str(self) == str(other)

    def __lt__(self, other) -> bool:
        """
        Determines if the instance is less than the other.

        """

        return str(self) < str(other)

    def __repr__(self) -> str:
        """
        Representation of the type ignore usage.

        """

        return f"{self.filename}:{self.line}:{self.flag}"

    def __str__(self) -> str:
        """
        Representation of the type ignore usage.

        """

        return f"{self.filename}:{self.line}:{self.flag}"


def main() -> None:
    """
    Main method for the pylint enforcement module.

    """

    # Load the declared uses of "pylint: disable".
    with open(YAML_ENFORCEMENT_FILE) as f:
        pylint_disable_declarations = yaml.safe_load(f)

    processed_pylint_disable_declarations = sorted(
        [
            PylintDisable(entry["file"], entry["flag"], entry["line"])
            for entry in pylint_disable_declarations
        ]
    )

    if not all(
        (JUSTIFICATION_STRING in entry for entry in pylint_disable_declarations)
    ):
        raise EnforcementError(
            "Not all entries were justified: {}".format(
                "; ".join(
                    [
                        entry
                        for entry in pylint_disable_declarations
                        if JUSTIFICATION_STRING not in entry
                    ]
                )
            )
        )

    # Run from the above directory if needed.
    if "pvt_model" not in os.listdir():
        directory_prefix: str = ".." + os.path.sep
    else:
        directory_prefix = ""

    # Determine the actual uses
    pylint_disable_uses = (
        subprocess.run(
            f"grep 'pylint: disable' {directory_prefix}pvt_model -rn",
            capture_output=True,
            check=True,
            shell=True,
        )
        .stdout.decode("utf-8")
        .split("\n")
    )
    pylint_disable_uses = [
        entry.replace("# pylint: disable", "") for entry in pylint_disable_uses
    ]
    pylint_disable_uses = [entry for entry in pylint_disable_uses if entry != ""]

    # Process this into usable data.
    processed_pylint_disable_uses = sorted(
        [
            PylintDisable(
                entry.split(":")[0],
                ":".join(entry.split(":")[2:]).split("=")[1],
                int(entry.split(":")[1]),
            )
            for entry in pylint_disable_uses
        ]
    )

    if processed_pylint_disable_declarations != processed_pylint_disable_uses:
        raise EnforcementError(
            "Missing declarations:\n{}\nExcess declarations:\n{}".format(
                "\n".join(
                    [
                        str(entry)
                        for entry in processed_pylint_disable_uses
                        if entry not in processed_pylint_disable_declarations
                    ]
                ),
                "\n".join(
                    [
                        str(entry)
                        for entry in processed_pylint_disable_declarations
                        if entry not in processed_pylint_disable_uses
                    ]
                ),
            )
        )


if __name__ == "__main__":
    main()
