#!/bin/bash

# This script installs the UV tool from Astral.sh
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add UV to the PATH
echo 'export PATH="$HOME/.uv/bin:$PATH"' >> ~/.zshrc
# Source the updated .bashrc to make the changes effective immediately
source ~/.zshrc