#!/usr/bin/env bash
# Fetch CLD geometry from k4geo into the bundle layout.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_URL="https://github.com/key4hep/k4geo"
UPSTREAM_REV="v00-19-01"
COMPACT_SUBPATH="FCCee/CLD/compact/CLD_o2_v05"
CACHE_DIR="${COLLIDERML_BUNDLE_CACHE:-$HOME/.cache/colliderml/bundles}"
SRC_DIR="${CACHE_DIR}/k4geo-${UPSTREAM_REV}"

mkdir -p "${CACHE_DIR}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    git clone --depth 1 --branch "${UPSTREAM_REV}" "${UPSTREAM_URL}" "${SRC_DIR}"
else
    git -C "${SRC_DIR}" fetch --depth 1 origin "${UPSTREAM_REV}"
    git -C "${SRC_DIR}" checkout "${UPSTREAM_REV}"
fi

mkdir -p "${BUNDLE_DIR}/geometry"
ln -sfn "${SRC_DIR}/${COMPACT_SUBPATH}" "${BUNDLE_DIR}/geometry/CLD_o2_v05"

# Convenience root pointer matching `geometry.compact_xml` in detector.yaml.
ln -sfn "CLD_o2_v05/CLD_o2_v05.xml" "${BUNDLE_DIR}/geometry/CLD_o2_v05.xml"

echo "CLD bundle ready at ${BUNDLE_DIR}"
echo "NOTE: an ACTS material map is not provided upstream and must be"
echo "      generated locally before reco quality can be quoted."
