#!/bin/sh
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
GAME="$ROOT_DIR/doom/target/release/doom"
POLICY="${1:-training/output/best_policy.json}"

if [ ! -x "$GAME" ]; then
  "$ROOT_DIR/exec_mgmt.sh" -b || exit $?
fi

cd "$ROOT_DIR" || exit 1
exec "$GAME" --watch-agent "$POLICY"
