#!/bin/bash

set -e -E -u -o pipefail

# shellcheck source=.ci/retry.sh
source "$(dirname "${BASH_SOURCE[0]}")/retry.sh"

retry pwsh -command "Install-Module -Name PSScriptAnalyzer -Scope CurrentUser -SkipPublisherCheck"
echo "Linting PowerShell code"
pwsh -file ./.ci/lint-powershell.ps1 || exit 1

retry conda create -q -y -n test-env \
    "python=3.14[build=*_cp*]" \
    'pre-commit>=3.8.0' \
    'r-lintr>=3.3.0'

# shellcheck disable=SC1091
source activate test-env

echo "Running pre-commit checks"
pre-commit run --all-files || exit 1

echo "Linting R code"
Rscript ./.ci/lint-r-code.R "$(pwd)" || exit 1
