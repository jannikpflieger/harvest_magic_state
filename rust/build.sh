#!/usr/bin/env bash
# Build the Rust router and install it as the extension module harvest._router.
#
#   rust/build.sh            # release build (needs cargo; uses the active python)
#
# The module lands in src/harvest/_router.so; HarvestConfig(backend="python")
# keeps the pure-Python router.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
crate="$here/harvest_router"
PYO3_PYTHON="${PYO3_PYTHON:-$(command -v python)}" cargo build --release --manifest-path "$crate/Cargo.toml" "$@"
cp "$crate/target/release/libharvest_router.so" "$here/../src/harvest/_router.so"
echo "installed $here/../src/harvest/_router.so"
