#!/usr/bin/env bash
# SessionStart hook: pull latest, then show what's here so every session
# opens synced and shows what changed since last time.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 0

echo "=== git pull ==="
if ! git pull 2>&1; then
  echo "WARNING: git pull failed — see output above. Working tree may be out of sync."
fi

echo
echo "=== recent commits ==="
git log --oneline -10 2>&1

echo
echo "=== git status ==="
git status 2>&1
