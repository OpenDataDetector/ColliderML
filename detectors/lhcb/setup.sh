#!/usr/bin/env bash
# Fetch LHCb Run 3 geometry from gitlab.cern.ch/lhcb/Detector.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_URL="https://gitlab.cern.ch/lhcb/Detector.git"
UPSTREAM_REV="v1r41"
COMPACT_SUBPATH="compact/run3"
CACHE_DIR="${COLLIDERML_BUNDLE_CACHE:-$HOME/.cache/colliderml/bundles}"
SRC_DIR="${CACHE_DIR}/lhcb-Detector-${UPSTREAM_REV}"

mkdir -p "${CACHE_DIR}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    git clone --depth 1 --branch "${UPSTREAM_REV}" "${UPSTREAM_URL}" "${SRC_DIR}"
else
    git -C "${SRC_DIR}" fetch --depth 1 origin "${UPSTREAM_REV}"
    git -C "${SRC_DIR}" checkout "${UPSTREAM_REV}"
fi

mkdir -p "${BUNDLE_DIR}/geometry"
ln -sfn "${SRC_DIR}/${COMPACT_SUBPATH}" "${BUNDLE_DIR}/geometry/run3"
ln -sfn "run3/LHCb.xml" "${BUNDLE_DIR}/geometry/LHCb.xml"

cat <<'EOF'
LHCb bundle staged.

WARNING: this bundle is "sketch" status. Running it through the standard
ColliderML pipeline will fail at the ACTS tracking step because the forward
custom tracking builder is not yet implemented. See README.md.

The geometry itself loads fine in DD4hep and ddsim — useful for material
scans and visualisation in the meantime.
EOF
