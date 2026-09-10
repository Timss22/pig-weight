#!/usr/bin/env bash
# Stop hook: commit and push any changes so work is never stranded on one
# machine. Never fails silently — a failed commit or push says so clearly,
# since that's exactly the "forgot to sync" problem this exists to prevent.
set -uo pipefail
cd "$(dirname "$0")/../.." || {
  echo '{"systemMessage": "Stop hook: could not locate repo root — nothing committed."}'
  exit 0
}

if [ -z "$(git status --porcelain 2>/dev/null)" ]; then
  echo '{"systemMessage": "Stop hook: no changes to commit — working tree already clean, nothing pushed."}'
  exit 0
fi

CHANGED=$(git status --porcelain | awk '{print $2}' | tr '\n' ' ' | cut -c1-150)
git add -A

if ! COMMIT_OUT=$(git commit -m "Auto-sync: $CHANGED" -m "Automated commit from Claude Code Stop hook." 2>&1); then
  DETAIL=$(echo "$COMMIT_OUT" | tail -2 | tr '\n' ' ' | tr -d '"')
  printf '{"systemMessage": "Stop hook: COMMIT FAILED - nothing pushed. %s"}\n' "$DETAIL"
  exit 0
fi

if ! PUSH_OUT=$(git push 2>&1); then
  DETAIL=$(echo "$PUSH_OUT" | tail -3 | tr '\n' ' ' | tr -d '"')
  printf '{"systemMessage": "Stop hook: commit succeeded but PUSH FAILED - changes are committed locally only, run git push manually. %s"}\n' "$DETAIL"
  exit 0
fi

printf '{"systemMessage": "Stop hook: committed and pushed (%s)."}\n' "$CHANGED"
