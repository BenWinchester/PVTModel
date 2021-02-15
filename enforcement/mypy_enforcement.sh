#!/usr/bin/env bash
########################################################################################
# enforcement/mypy_enforcement.sh
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

# Script for ensuring that all mypy errors are declared correctly in the mypy yaml file.

# Determine the mypy `type:ignore` declarations.
type_ignore_uses="$(grep 'type: ignore' pvt_model -rn)"
type_ignore_declarations="$(cat enforcement/mypy_enforcement.txt)"
if [[ "$type_ignore_uses" == "$type_ignore_declarations" ]]
then
    exit 0
else
    echo "Not all `type:ignore` instances referenced in enforcement/my_enforcement.txt."
    exit 1
fi
