#!/bin/bash
# This script will be run on the *host* machine before VS Code starts building
# devcontainer image. Use it to prepare anything necessary for building the dev
# container or to check if the host environment meets requirements.

# Unofficial Bash Strict Mode
set -euo pipefail
IFS=$'\n\t'

ENV_FILE=".env"
ENV_FILE_TEMPLATE=".env.default"

# Create .env file if necessary
if [ ! -e $ENV_FILE ] ; then
  cp $ENV_FILE_TEMPLATE $ENV_FILE
fi

if [ -z "${CODESPACES:+x}" ] && [[ "$OSTYPE" == "darwin"* ]] ; then
    # Macs only add keys to agent on demand (no-op if already loaded)
    ssh-add --apple-load-keychain 2> /dev/null
fi
