#!/usr/bin/env bash
set -euo pipefail

CHANGELOG="$(dirname "$0")/../CHANGELOG.md"
ISSUES=0

if [ ! -f "$CHANGELOG" ]; then
	echo "CHANGELOG.md not found." >&2
	exit 1
fi

# Check that the file starts with "# Changelog"
head -1 "$CHANGELOG" | grep -q '^# Changelog' || {
	echo "ERROR: CHANGELOG.md must start with '# Changelog'" >&2
	ISSUES=$((ISSUES + 1))
}

# Check that at least one version entry exists (## [X.Y.Z] - YYYY-MM-DD)
grep -q '^## \[[0-9]\+\.[0-9]\+\.[0-9]\+\] - [0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}$' "$CHANGELOG" || {
	echo "ERROR: No valid version entry found. Expected format: '## [X.Y.Z] - YYYY-MM-DD'" >&2
	ISSUES=$((ISSUES + 1))
}

# Check for required metadata in the first version entry
grep -qE '^- \*\*why:\*\*' "$CHANGELOG" || {
	echo "ERROR: First version entry missing '**why:**' metadata" >&2
	ISSUES=$((ISSUES + 1))
}

grep -qE '^- \*\*model:\*\*' "$CHANGELOG" || {
	echo "ERROR: First version entry missing '**model:**' metadata" >&2
	ISSUES=$((ISSUES + 1))
}

grep -qE '^- \*\*tags:\*\*' "$CHANGELOG" || {
	echo "ERROR: First version entry missing '**tags:**' metadata" >&2
	ISSUES=$((ISSUES + 1))
}

# Check for at least one category heading
grep -q '^### \(Added\|Changed\|Fixed\|Removed\|Security\)$' "$CHANGELOG" || {
	echo "ERROR: No category heading found (### Added, ### Changed, ### Fixed, ### Removed, ### Security)" >&2
	ISSUES=$((ISSUES + 1))
}

if [ "$ISSUES" -gt 0 ]; then
	echo "Validation failed with $ISSUES issue(s)." >&2
	exit 1
fi

echo "CHANGELOG.md validates successfully."
