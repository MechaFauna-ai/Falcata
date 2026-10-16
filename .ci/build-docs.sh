#!/bin/bash

set -e -E -u -o pipefail

# shellcheck source=.ci/retry.sh
source "$(dirname "${BASH_SOURCE[0]}")/retry.sh"

retry conda env create \
    --name test-env \
    --file ./docs/env.yml \
|| exit 1

# shellcheck disable=SC1091
source activate test-env

make -C docs html || exit 1
