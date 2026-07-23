#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FINAL_BIN="${PROJECT_DIR}/target/linux-release/npyz-viewer"

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "ERROR: pkg-config is required." >&2
  exit 1
fi

if pkg-config --exists webkit2gtk-4.1 &&
  pkg-config --atleast-version=2.70 glib-2.0; then
  VARIANT="Tauri 2 / WebKitGTK 4.1"
  MANIFEST="${PROJECT_DIR}/Cargo.toml"
  TARGET_DIR="${PROJECT_DIR}/target"
elif pkg-config --exists webkit2gtk-4.0; then
  VARIANT="Tauri 1 compatibility / WebKitGTK 4.0"
  MANIFEST="${PROJECT_DIR}/compat/rocky8/Cargo.toml"
  TARGET_DIR="${PROJECT_DIR}/target/rocky8"
else
  GLIB_VERSION="$(pkg-config --modversion glib-2.0 2>/dev/null || echo unavailable)"
  echo "ERROR: No supported WebKitGTK development package was found." >&2
  echo "Detected GLib: ${GLIB_VERSION}" >&2
  echo "Modern Linux: install webkit2gtk-4.1 development files." >&2
  echo "Rocky/RHEL 8: sudo dnf install webkit2gtk3-devel gtk3-devel openssl-devel" >&2
  exit 1
fi

echo "Selected: ${VARIANT}"
cargo build \
  --locked \
  --release \
  --bin npyz-viewer \
  --manifest-path "${MANIFEST}" \
  --target-dir "${TARGET_DIR}"

BUILT_BIN="${TARGET_DIR}/release/npyz-viewer"
if [[ ! -x "${BUILT_BIN}" ]]; then
  echo "ERROR: Release executable not found: ${BUILT_BIN}" >&2
  exit 1
fi

install -Dm755 "${BUILT_BIN}" "${FINAL_BIN}"
echo "Built: ${FINAL_BIN}"
