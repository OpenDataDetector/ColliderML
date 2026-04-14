#!/usr/bin/env bash
#
# One-shot setup for the ColliderML platform tutorial.
#
# Starts the backend (Postgres + FastAPI) via the colliderml-production
# docker-compose stack, waits for the service to come up, and grants 100
# tutorial credits to a HuggingFace user so they can drive the
# simulate / leaderboard / webhook flows against the local backend
# instead of api.colliderml.com.
#
# This script is opinionated about:
#   * using docker-compose OR podman-compose (autodetected),
#   * talking to http://localhost:8000 by default,
#   * granting credits with the default dev admin token
#     (`dev-admin-token`), which is fine for localhost but NOT for any
#     deployed backend.
#
# Usage:
#
#   ./scripts/setup_tutorial_env.sh path/to/colliderml-production/backend [hf-username]
#
# If hf-username is omitted we read it from `huggingface-cli whoami` or
# the HF_USERNAME env var.
#
# On success, prints the env-var snippet to paste into the shell that
# will run the tutorial notebook:
#
#   export COLLIDERML_BACKEND=http://localhost:8000
#   export COLLIDERML_DATA_DIR=...
#   export HF_HOME=...
#
# Re-running the script is safe — compose is idempotent, and
# /admin/grant adds credits rather than setting them (so a re-run
# gives another 100).

set -euo pipefail

BACKEND_DIR="${1:-}"
HF_USER="${2:-${HF_USERNAME:-}}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-dev-admin-token}"
BACKEND_URL="http://localhost:${BACKEND_PORT}"
TUTORIAL_CREDITS="${TUTORIAL_CREDITS:-100}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-180}"

# --- argument checks --------------------------------------------------

if [[ -z "${BACKEND_DIR}" || ! -d "${BACKEND_DIR}" ]]; then
    echo "error: first argument must be the backend directory of the" >&2
    echo "       colliderml-production repo (contains docker-compose.yml)." >&2
    echo "usage: $0 path/to/colliderml-production/backend [hf-username]" >&2
    exit 2
fi

if [[ -z "${HF_USER}" ]]; then
    if command -v huggingface-cli >/dev/null 2>&1; then
        HF_USER=$(huggingface-cli whoami 2>/dev/null | head -n1 || true)
    fi
fi
if [[ -z "${HF_USER}" ]]; then
    echo "error: no HuggingFace username supplied." >&2
    echo "       pass it as the second argument, set \$HF_USERNAME," >&2
    echo "       or run 'huggingface-cli login' first." >&2
    exit 2
fi

# --- compose tool detection -------------------------------------------

COMPOSE=""
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v podman-compose >/dev/null 2>&1; then
    COMPOSE="podman-compose"
else
    echo "error: no compose tool found on PATH." >&2
    echo "       install one of: docker-compose, 'docker compose'," >&2
    echo "       or 'pip install podman-compose'." >&2
    exit 2
fi

echo "[setup] compose tool: ${COMPOSE}"
echo "[setup] backend dir:  ${BACKEND_DIR}"
echo "[setup] backend url:  ${BACKEND_URL}"
echo "[setup] HF user:      ${HF_USER}"

# --- bring the stack up -----------------------------------------------

pushd "${BACKEND_DIR}" >/dev/null
${COMPOSE} up -d
popd >/dev/null

# --- wait for /healthz ------------------------------------------------

echo "[setup] waiting for ${BACKEND_URL}/healthz (timeout ${HEALTH_TIMEOUT_SEC}s) ..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SEC ))
while true; do
    status=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/healthz" || echo "000")
    if [[ "${status}" == "200" ]]; then
        echo "[setup] backend is healthy."
        break
    fi
    if (( $(date +%s) > deadline )); then
        echo "error: backend did not become healthy within ${HEALTH_TIMEOUT_SEC}s." >&2
        echo "       last /healthz status was ${status}. see 'docker logs backend_backend_1'." >&2
        exit 3
    fi
    sleep 2
done

# --- grant tutorial credits -------------------------------------------

echo "[setup] granting ${TUTORIAL_CREDITS} tutorial credits to ${HF_USER} ..."
grant_status=$(curl -s -o /tmp/setup-grant-resp.$$ -w "%{http_code}" \
    -X POST "${BACKEND_URL}/admin/grant" \
    -H "Authorization: Bearer ${ADMIN_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"hf_username\": \"${HF_USER}\", \"credits\": ${TUTORIAL_CREDITS}}")
if [[ "${grant_status}" != "200" ]]; then
    echo "error: /admin/grant returned ${grant_status}:" >&2
    cat /tmp/setup-grant-resp.$$ >&2 || true
    rm -f /tmp/setup-grant-resp.$$
    exit 4
fi
rm -f /tmp/setup-grant-resp.$$

# --- print env vars for the caller ------------------------------------

cat <<EOF

[setup] ready. export the following in the shell that runs the tutorial:

    export COLLIDERML_BACKEND=${BACKEND_URL}
    # If you're on a machine with a small \$HOME (e.g. NERSC), redirect caches:
    # export COLLIDERML_DATA_DIR=/path/on/scratch/colliderml
    # export HF_HOME=/path/on/scratch/hf

Next: open notebooks/tutorial.ipynb.
EOF
