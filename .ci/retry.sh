#!/bin/bash
#
# retry: run a network-dependent setup command up to 3 times, 25 s apart.
#
# For downloads and package-manager calls only (curl, conda create/update,
# Install-Module). Never wrap a lint, test or docs check with this: those
# must fail on the first error.
#
# usage:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/retry.sh"
#   retry conda update -q -y conda
#
retry() {
    local max_attempts=3
    local delay_seconds=25
    local attempt=1
    while true; do
        local exit_code=0
        "$@" || exit_code=$?
        if [[ ${exit_code} -eq 0 ]]; then
            return 0
        fi
        if [[ ${attempt} -ge ${max_attempts} ]]; then
            echo "[retry] '$*' failed with exit code ${exit_code} on attempt ${attempt}/${max_attempts}, giving up" >&2
            return "${exit_code}"
        fi
        echo "[retry] '$*' failed with exit code ${exit_code} on attempt ${attempt}/${max_attempts}, retrying in ${delay_seconds}s" >&2
        sleep "${delay_seconds}"
        attempt=$((attempt + 1))
    done
}
