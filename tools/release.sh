#!/usr/bin/env bash
# tools/release.sh — release one workspace crate via cargo-release + create
# a GitHub Release from the CHANGELOG entry.
#
# Usage:
#   tools/release.sh <crate> <bump>
#
# Where:
#   <crate>  is the package name (e.g., nexus-collections)
#   <bump>   is one of: patch | minor | major | <explicit-version>
#
# Workflow:
#   1. cargo-release bumps Cargo.toml, renames `## [Unreleased]` in
#      CHANGELOG.md to `## [<version>] — <date>`, commits, tags as
#      `<crate>-v<version>`, pushes the tag, publishes to crates.io.
#   2. This script then extracts the `## [<version>]` section from
#      CHANGELOG.md and creates a GitHub Release with that as notes.
#
# Pre-requisites:
#   - On main, working tree clean.
#   - cargo-release installed (`cargo install cargo-release`).
#   - gh CLI authenticated (`gh auth status`).
#   - CARGO_REGISTRY_TOKEN set or `cargo login` done previously.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <crate> <bump>" >&2
    echo "  e.g.: $0 nexus-collections patch" >&2
    exit 1
fi

crate="$1"
bump="$2"

if [ ! -f "$crate/Cargo.toml" ]; then
    echo "Error: $crate/Cargo.toml not found (run from workspace root)" >&2
    exit 1
fi

if [ ! -f "$crate/CHANGELOG.md" ]; then
    echo "Error: $crate/CHANGELOG.md not found" >&2
    exit 1
fi

if [ "$(git symbolic-ref --short HEAD 2>/dev/null)" != "main" ]; then
    echo "Error: not on main branch" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "Error: working tree not clean" >&2
    exit 1
fi

echo "==> cargo release $bump --execute -p $crate"
cargo release "$bump" --execute -p "$crate"

# After cargo-release: read the new version from Cargo.toml.
version=$(grep -m1 '^version' "$crate/Cargo.toml" | cut -d'"' -f2)
tag="${crate}-v${version}"

echo "==> Extracting CHANGELOG section for $version"
notes=$(awk -v ver="$version" '
    $0 ~ "^## \\[" ver "\\]" { p=1; print; next }
    p && $0 ~ "^## \\[" { exit }
    p { print }
' "$crate/CHANGELOG.md")

if [ -z "$notes" ]; then
    echo "Warning: could not extract CHANGELOG section for [$version] in $crate/CHANGELOG.md" >&2
    echo "Creating release with empty notes — edit on GitHub if needed." >&2
fi

echo "==> gh release create $tag"
gh release create "$tag" \
    --title "$crate v$version" \
    --notes "$notes"

echo
echo "Released: $tag"
echo "  crates.io: https://crates.io/crates/$crate/$version"
echo "  GitHub:    https://github.com/Abso1ut3Zer0/nexus/releases/tag/$tag"
