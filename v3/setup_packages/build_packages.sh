#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_ROOT="${ROOT}/setup_packages"
DIST="${PKG_ROOT}/dist"
TS="$(date -u +%Y%m%d_%H%M%S)"

mkdir -p "${DIST}"

_mk_pkg() {
  local name="$1"
  local src_dir="${PKG_ROOT}/${name}"
  local out="${DIST}/${name}_${TS}.tar.gz"
  tar -C "${src_dir}" -czf "${out}" .
  echo "${out}"
}

echo "[build] building packages into ${DIST}"
_mk_pkg setup_v4
_mk_pkg setup_v6e_us
_mk_pkg setup_v6e_eu
echo "[build] done"

