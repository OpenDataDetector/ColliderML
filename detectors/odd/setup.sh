#!/usr/bin/env bash
# Fetch ODD geometry assets and link them into the bundle layout expected
# by detector.yaml. Idempotent.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_URL="https://github.com/OpenDataDetector/OpenDataDetector"
UPSTREAM_REV="v3.0.0"
CACHE_DIR="${COLLIDERML_BUNDLE_CACHE:-$HOME/.cache/colliderml/bundles}"
SRC_DIR="${CACHE_DIR}/OpenDataDetector-${UPSTREAM_REV}"

mkdir -p "${CACHE_DIR}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    git clone --depth 1 --branch "${UPSTREAM_REV}" "${UPSTREAM_URL}" "${SRC_DIR}"
else
    git -C "${SRC_DIR}" fetch --depth 1 origin "${UPSTREAM_REV}"
    git -C "${SRC_DIR}" checkout "${UPSTREAM_REV}"
fi

mkdir -p "${BUNDLE_DIR}/geometry" "${BUNDLE_DIR}/acts"

# Symlink the entire xml/ tree so includes resolve.
ln -sfn "${SRC_DIR}/xml/OpenDataDetector.xml" "${BUNDLE_DIR}/geometry/OpenDataDetector.xml"
for inc in "${SRC_DIR}"/xml/*.xml; do
    name="$(basename "${inc}")"
    [[ "${name}" == "OpenDataDetector.xml" ]] && continue
    ln -sfn "${inc}" "${BUNDLE_DIR}/geometry/${name}"
done

ln -sfn "${SRC_DIR}/config/odd-digi-geometric-config.json" "${BUNDLE_DIR}/acts/digi.json"
ln -sfn "${SRC_DIR}/data/odd-material-maps.root"           "${BUNDLE_DIR}/acts/odd-material-maps.root"

echo "ODD bundle ready at ${BUNDLE_DIR}"
