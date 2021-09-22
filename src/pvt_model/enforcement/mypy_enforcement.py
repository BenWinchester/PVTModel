#!/usr/bin/python3.7
########################################################################################
# enforcement/mypy_enforcement.py
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
Script for ensuring that all mypy errors are declared correctly in the mypy yaml file.

All uses of the `type:ignore` mypy declaration must be declared in mypy_enforcement.yaml
in the enforcement folder.

"""

import os
import subprocess

from dataclasses import dataclass

import yaml

# String required in all enforcement entries.
JUSTIFICATION_STRING = "justification"
# Path to the YAML enforcement file.
YAML_ENFORCEMENT_FILE = os.path.join(
    "pvt_model", "enforcement", "mypy_enforcement.yaml"
)


class EnforcementError(Exception):
    """
    Raised when an enforcement error occurs.

    """

    def __init__(self, msg: str) -> None:
        """
        Instantiate a :class:`EnforcementError` instance.

        """

        super().__init__(
            f"Not all mypy 'type:ignore' uses were declared in the yaml file: {msg}"
        )


@dataclass
class TypeIgnoreUsage:
    """
    Represents a single use of the "type: ignore" declaration.

    .. attribute:: filename
        The name of the file containing the usage.

    .. attribute:: usage
        The actual usage within the file.

    """

    filename: str
    usage: str

    def __eq__(self, other) -> bool:
        """
        Determines if two instances are equal.

        """

        return str(self) == str(other)

    def __hash__(self) -> int:
        """
        Returns a unique string.

        """

        return hash(self.__repr__())

    def __lt__(self, other) -> bool:
        """
        Determines if the instance is less than the other.

        """

        return str(self) < str(other)

    def __repr__(self) -> str:
        """
        Representation of the type ignore usage.

        """

        return f"{self.filename}:{self.usage}"

    def __str__(self) -> str:
        """
        Representation of the type ignore usage.

        """

        return f"{self.filename}:{self.usage}"


def main() -> None:
    """
    Main method for the mypy enforcement module.

    """

    # Load the declared uses of "type:ignore".
    with open(YAML_ENFORCEMENT_FILE) as f:
        type_ignore_declarations = yaml.safe_load(f)

    try:
        processed_type_ignore_declarations = {
            TypeIgnoreUsage(entry["file"], entry["usage"])
            for entry in type_ignore_declarations
        }

    except KeyError as e:
        print(f"Not all data entries conform in the enforcement file: {str(e)}")
        raise

    if not all((JUSTIFICATION_STRING in entry for entry in type_ignore_declarations)):
        raise EnforcementError(
            "Not all entries were justified:\n  {}".format(
                "\n  ".join(
                    [
                        f"{entry['file']}: {entry['usage']}"
                        for entry in type_ignore_declarations
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
    type_ignore_uses = (
        subprocess.run(
            f"grep 'type: ignore' {directory_prefix}pvt_model -rn --exclude \\*.yaml",
            capture_output=True,
            check=True,
            shell=True,
        )
        .stdout.decode("utf-8")
        .split("\n")
    )
    type_ignore_uses = [
        entry.replace("# type: ignore", "") for entry in type_ignore_uses
    ]
    type_ignore_uses = [entry for entry in type_ignore_uses if entry != ""]

    # Process this into usable data.
    processed_type_ignore_uses = {
        TypeIgnoreUsage(
            entry.split(":")[0],
            ":".join(entry.split(":")[2:]).strip(),
        )
        for entry in type_ignore_uses
    }

    if processed_type_ignore_declarations != processed_type_ignore_uses:
        raise EnforcementError(
            "Missing declarations:\n{}\nExcess declarations:\n{}".format(
                "\n".join(
                    [
                        str(entry)
                        for entry in processed_type_ignore_uses
                        if entry not in processed_type_ignore_declarations
                    ]
                ),
                "\n".join(
                    [
                        str(entry)
                        for entry in processed_type_ignore_declarations
                        if entry not in processed_type_ignore_uses
                    ]
                ),
            )
        )


if __name__ == "__main__":
    main()
