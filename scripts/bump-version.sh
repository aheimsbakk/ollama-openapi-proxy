#!/usr/bin/env bash
set -euo pipefail

VERSION_FILE="$(dirname "$0")/../VERSION"

if [ ! -f "$VERSION_FILE" ]; then
	echo "Error: VERSION file not found at $VERSION_FILE" >&2
	exit 1
fi

CURRENT=$(cat "$VERSION_FILE")
BUMP="${1:-patch}"

if [ "$BUMP" != "major" ] && [ "$BUMP" != "minor" ] && [ "$BUMP" != "patch" ]; then
	echo "Usage: $0 [major|minor|patch]" >&2
	exit 1
fi

major=$(echo "$CURRENT" | cut -d. -f1)
minor=$(echo "$CURRENT" | cut -d. -f2)
patch=$(echo "$CURRENT" | cut -d. -f3)

case "$BUMP" in
major)
	major=$((major + 1))
	minor=0
	patch=0
	;;
minor)
	minor=$((minor + 1))
	patch=0
	;;
patch)
	patch=$((patch + 1))
	;;
esac

NEW_VERSION="$major.$minor.$patch"
echo "$NEW_VERSION" >"$VERSION_FILE"
echo "$NEW_VERSION"
