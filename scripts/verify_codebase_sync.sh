#!/usr/bin/env bash
# verify_codebase_sync.sh — Validates that all paths listed in CODEBASE.md exist.
#
# Usage:
#   ./scripts/verify_codebase_sync.sh
#
# Exit codes:
#   0  All paths exist
#   1  One or more paths are missing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODEBASE="$REPO_ROOT/CODEBASE.md"

if [[ ! -f "$CODEBASE" ]]; then
	echo "ERROR: CODEBASE.md not found at $CODEBASE" >&2
	exit 1
fi

missing=0
count=0

# Extract file paths from the directory tree block using Python.
# We track directory context by following the tree structure line by line.
while IFS= read -r filepath; do
	[[ -z "$filepath" ]] && continue
	[[ "$filepath" =~ /$ ]] && continue

	fullpath="$REPO_ROOT/$filepath"
	if [[ ! -e "$fullpath" ]]; then
		echo "MISSING: $filepath"
		missing=$((missing + 1))
	else
		count=$((count + 1))
	fi
done < <(
	python3 <<'PYEOF'
import re

with open("CODEBASE.md") as f:
    content = f.read()

# Find the code block that contains tree-drawing characters
blocks = re.split(r'\x60\x60\x60\s*\n', content)
tree_block = None
for block in blocks:
    if '├──' in block or '└──' in block:
        tree_block = block
        break

if not tree_block:
    import sys
    sys.exit(0)

# Parse tree line by line, tracking directory context.
# Each line has a structure: [│ ... │] [spaces] (├──|└──) [spaces] entry
# We track which directory we're currently in by following the tree hierarchy.

# Build a list of (depth, entry) tuples by counting the number of │ characters
# and using the spaces after the last │ to determine sub-level.
lines_info = []
for line in tree_block.split('\n'):
    if '├──' not in line and '└──' not in line:
        continue
    
    # Count │ characters
    pipe_count = line.count('│')
    
    # Find the last │ and count spaces after it before the tree character
    last_pipe = line.rfind('│')
    if last_pipe >= 0:
        after_last_pipe = line[last_pipe + 1:]
        # Count leading spaces in the part after the last │
        leading_spaces = len(after_last_pipe) - len(after_last_pipe.lstrip(' '))
    else:
        # No │ character - count leading spaces before the tree character
        leading_spaces = len(line) - len(line.lstrip(' '))
        # But only if there's a tree character after the spaces
        stripped = line.lstrip(' ')
        if not stripped.startswith('├──') and not stripped.startswith('└──'):
            leading_spaces = 0
    
    # Extract entry
    match = re.search(r'(?:│\s*)?(?:├──|└──)\s+(.+)', line)
    if not match:
        continue
    
    entry = match.group(1).strip().split('#')[0].strip()
    if not entry:
        continue
    
    lines_info.append((pipe_count, leading_spaces, entry))

# Now reconstruct paths using a stack-based approach.
# We track the current directory path and push/pop as we encounter directories.
current_path = ""
dir_stack = []  # Stack of (pipe_count, leading_spaces, dir_name, full_path)

for pipe_count, leading_spaces, entry in lines_info:
    is_dir = entry.endswith('/')
    name = entry.rstrip('/')
    
    # Determine if this entry is a child of the current directory or a sibling/parent
    # A child has more │ characters than the parent, or same │ count but more leading spaces
    if dir_stack:
        parent = dir_stack[-1]
        parent_pipes, parent_spaces, parent_name, parent_path = parent
        
        # Check if this is a child of the current directory
        is_child = (pipe_count > parent_pipes) or (pipe_count == parent_pipes and leading_spaces > parent_spaces)
        
        if not is_child:
            # Pop back to find the correct parent
            while dir_stack:
                parent = dir_stack[-1]
                parent_pipes, parent_spaces, parent_name, parent_path = parent
                is_child = (pipe_count > parent_pipes) or (pipe_count == parent_pipes and leading_spaces > parent_spaces)
                if is_child:
                    break
                dir_stack.pop()
    
    if is_dir:
        # Push directory onto stack
        if dir_stack:
            _, _, _, parent_path = dir_stack[-1]
            new_path = f"{parent_path}/{name}"
        else:
            new_path = name if name else ""
        dir_stack.append((pipe_count, leading_spaces, name, new_path))
    else:
        # Output file path
        if dir_stack:
            _, _, _, parent_path = dir_stack[-1]
            if parent_path:
                print(f"{parent_path}/{name}")
            else:
                print(name)
        else:
            print(name)
PYEOF
)

if [[ $missing -gt 0 ]]; then
	echo ""
	echo "FAIL: $missing path(s) missing from CODEBASE.md" >&2
	exit 1
fi

echo "OK: All $count paths verified."
exit 0
