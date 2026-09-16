#!/bin/sh
# Every assertion about one name, in both languages, before scoping a spec.
# Usage: ./scripts/preflight.sh repo_root
set -eu
name="$1"
echo "=== $name: definitions and uses ==="
grep -rn "\b$name\b" symphonai_api symphonai_host symphonai_app/src scripts --include="*.py" --include="*.js" --include="*.rs" 2>/dev/null | head -30
echo
echo "=== exact-shape assertions that mention it (Python checks) ==="
grep -rnE "set\([a-z_]+(\[0\])?\) [!=]= \{|json\.loads\([a-z_]+\) [!=]= \{|== \{\"" scripts/checks/*.py | grep -i "$name" || echo "  (none naming it — check the ones below by hand)"
echo
echo "=== exact-shape assertions that mention it (JavaScript tests) ==="
grep -rnE "deepEqual\(" symphonai_app/test/*.js | grep -i "$name" || echo "  (none naming it — check the ones below by hand)"
echo
echo "=== every exact-shape assertion in the repo, for eyeballing ==="
grep -rcE "set\([a-z_]+(\[0\])?\) [!=]= \{|json\.loads\([a-z_]+\) [!=]= \{" scripts/checks/*.py | grep -v ":0" || true
grep -rcE "deepEqual\([^,]+, *\{" symphonai_app/test/*.js | grep -v ":0" || true
echo
echo "=== lists that enumerate siblings (adding a file of a known kind?) ==="
grep -rn "NODE_TESTS\|expected_names\|EXPECTED_REPOSITORY_CHECKS" scripts/checks/*.py | head -5
