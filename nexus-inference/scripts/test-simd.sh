#!/usr/bin/env bash
#
# Run nexus-inference tests across all SIMD tiers.
#
# Tiers:
#   scalar  — baseline x86-64 (SSE2 only), no SIMD matmul kernels
#   avx2    — compile-time AVX2, maddubs i8 kernel active
#   avx512  — compile-time AVX-512BW, requires Intel SDE to run
#
# Usage:
#   ./scripts/test-simd.sh           # run all tiers (avx512 requires SDE)
#   ./scripts/test-simd.sh scalar    # scalar only
#   ./scripts/test-simd.sh avx2     # avx2 only
#   ./scripts/test-simd.sh avx512   # avx512 only (requires SDE)

set -euo pipefail

CRATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_DIR="$(cd "$CRATE_DIR/.." && pwd)"
SDE="${SDE:-/opt/intel-sde/sde64}"
TARGET="x86_64-unknown-linux-gnu"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; }
skip() { echo -e "  ${YELLOW}SKIP${NC} $1"; }

run_tier() {
    local tier="$1"
    echo ""
    echo "=== $tier ==="

    case "$tier" in
        scalar)
            echo "  Building (baseline x86-64, SSE2 only)..."
            if cargo test -p nexus-inference --target "$TARGET" --quiet 2>&1; then
                pass "unit + loader tests"
            else
                fail "unit + loader tests"
                return 1
            fi
            ;;

        avx2)
            echo "  Building (compile-time AVX2)..."
            if CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-feature=+avx2" \
                cargo test -p nexus-inference --target "$TARGET" --quiet 2>&1; then
                pass "unit + loader tests"
            else
                fail "unit + loader tests"
                return 1
            fi
            ;;

        avx512)
            if [ ! -x "$SDE" ]; then
                skip "Intel SDE not found at $SDE (set SDE= to override)"
                return 0
            fi

            echo "  Building (compile-time AVX-512BW)..."
            CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-feature=+avx512bw" \
                cargo test -p nexus-inference --lib --target "$TARGET" --no-run --quiet 2>&1

            local bin
            bin=$(CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-feature=+avx512bw" \
                cargo test -p nexus-inference --lib --target "$TARGET" --no-run --message-format=json 2>/dev/null \
                | jq -r 'select(.executable != null) | .executable' | head -1)

            if [ -z "$bin" ]; then
                fail "could not locate test binary"
                return 1
            fi

            echo "  Running under SDE (Sapphire Rapids emulation)..."
            if "$SDE" -spr -- "$bin" quantized 2>&1; then
                pass "unit + loader tests (SDE)"
            else
                fail "unit + loader tests (SDE)"
                return 1
            fi
            ;;

        *)
            echo "Unknown tier: $tier"
            echo "Valid tiers: scalar, avx2, avx512"
            return 1
            ;;
    esac
}

cd "$WORKSPACE_DIR"

tiers=("${@:-scalar avx2 avx512}")
if [ $# -eq 0 ]; then
    tiers=(scalar avx2 avx512)
fi

failures=0
for tier in "${tiers[@]}"; do
    if ! run_tier "$tier"; then
        ((failures++))
    fi
done

echo ""
if [ "$failures" -eq 0 ]; then
    echo -e "${GREEN}All tiers passed.${NC}"
else
    echo -e "${RED}${failures} tier(s) failed.${NC}"
    exit 1
fi
