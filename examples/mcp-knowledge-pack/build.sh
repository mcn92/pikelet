#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

node "$REPO_ROOT/packages/pikelet/bin/pikelet.mjs" compile \
  --source "$DIR/sources/modest-proposal" \
  --out "$DIR/modest-proposal.pikelet" \
  --name modest-proposal --force
