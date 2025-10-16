#!/bin/bash

# Change the ownership of the /app directory to appuser
# This ensures that any mounted volumes are also owned by appuser
chown -R appuser:appgroup /app

# Execute the command passed to the script (the CMD from the Dockerfile)
# 'su-exec' switches from the root user to 'appuser' before running the command
exec gosu appuser "$@"