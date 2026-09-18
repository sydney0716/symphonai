#!/bin/sh
# Run every concrete claim a spec makes about the tree, before handing it over.
# Usage: ./scripts/spec_claims.sh specs/22/22a-....md
set -eu
spec="$1"
echo "=== commands this spec asserts, and what they really return ==="
grep -oE '`(grep|rg|ls|wc|test|python3|node)[^`]*`' "$spec" | tr -d '`' | sort -u | while read -r cmd; do
    case "$cmd" in *"<"*|*">"*|*"…"*) continue;; esac
    printf '  %-62s -> ' "$cmd"
    out=$(eval "$cmd" 2>&1 | head -1)
    [ -n "$out" ] && echo "$out" || echo "(no output)"
done
echo
echo "=== file paths the spec names that do not exist ==="
grep -oE '`[A-Za-z0-9_./-]+\.(py|js|json|md|sh|toml|rs|css|html)`' "$spec" | tr -d '`' | sort -u | while read -r f; do
    [ -e "$f" ] || echo "  MISSING: $f"
done
echo "  (nothing above = every named file exists)"
echo
echo "=== symbols the spec names that are not in the tree ==="
grep -oE '`[a-z_][a-z0-9_]{4,}`' "$spec" | tr -d '`' | sort -u | while read -r s; do
    grep -rqF "$s" symphonai_api symphonai_host symphonai_app/src scripts publish.sh 2>/dev/null || echo "  NOT FOUND: $s"
done
echo "  (nothing above = every named symbol exists)"
