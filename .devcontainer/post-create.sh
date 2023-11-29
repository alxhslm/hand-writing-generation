#!/bin/bash

# Unofficial Bash Strict Mode
set -euo pipefail
IFS=$'\n\t'

WORKSPACE="/workspace"

# Copy bash/zsh history from persistent cache
touch /home/vscode/.cache/.bash_history /home/vscode/.cache/.zsh_history
ln -Fs /home/vscode/.cache/.bash_history /home/vscode/.bash_history
ln -Fs /home/vscode/.cache/.zsh_history /home/vscode/.zsh_history

# Start in a well-know location
cd $WORKSPACE

# Install pre-commit hooks
if [ ! -e ".git/hooks/pre-commit" ] ; then
  pre-commit install
fi

# Install direnv hooks
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
direnv allow
eval "$(direnv export bash)"

# Ensure poetry uses project's virtual env, regardless if one is activated
unset VIRTUAL_ENV

if [ ! -e "$HOME/.config/pypoetry/config.toml" ] ; then
  echo "Configuring poetry..."
  poetry config virtualenvs.in-project true
fi

make init
echo '
Dev container is ready to use!
'
