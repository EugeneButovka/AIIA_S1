#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:?Set your Azure VM public IP: export VM_HOST=<public-ip>}"
VM_USER="${VM_USER:-azureuser}"
REPO_URL="${REPO_URL:?Set your Git repository URL: export REPO_URL=https://github.com/your-username/your-repo.git}"
REMOTE_DIR="${REMOTE_DIR:-}"

ssh "${VM_USER}@${VM_HOST}" bash -s -- "${REPO_URL}" "${REMOTE_DIR}" <<'EOF'
set -euo pipefail

REPO_URL="$1"
REMOTE_DIR="${2:-${HOME}/yolo-api}"

if ! command -v docker > /dev/null 2>&1; then
    echo "Error: Docker is not installed on the VM." >&2
    exit 1
fi

if ! command -v git > /dev/null 2>&1; then
    echo "Error: git is not installed on the VM." >&2
    exit 1
fi

if [ ! -d "${REMOTE_DIR}" ]; then
    echo "First deployment: cloning ${REPO_URL} into ${REMOTE_DIR}"
    git clone "${REPO_URL}" "${REMOTE_DIR}"
else
    echo "Updating existing deployment in ${REMOTE_DIR}"
fi

cd "${REMOTE_DIR}"
git pull --ff-only

echo "Building and starting the service..."
docker compose up -d --build
docker compose ps
EOF