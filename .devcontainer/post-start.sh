#!/bin/bash
# This script is executed every time the dev container starts

# Unofficial Bash Strict Mode
set -euo pipefail
IFS=$'\n\t'

# Make sure Git sees workspace as safe
#
# This can't be run in post-create.sh as it would prevent VS Code from copying
# user's .gitconfig and would decrease Dev Container's personalisation.
#
# Note: According to the docs, this shouldn't be necessary, as /workspace and
# everything below is owned by `vscode` user, but somehow git has problems with
# this dev container setup. Docs:
# https://git-scm.com/docs/git-config/2.35.2#Documentation/git-config.txt-safedirectory
if [[ ! `git config --global --get-all safe.directory | grep '^/workspace$'` ]]; then
  # Adding only when not added already
  git config --global --add safe.directory /workspace
fi
