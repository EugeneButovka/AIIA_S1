#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f deploy.env ]; then
    set -a
    . ./deploy.env
    set +a
fi

VM_HOST="${VM_HOST:?Set your Azure VM public IP: export VM_HOST=<public-ip> or put it in deploy.env}"
VM_USER="${VM_USER:-azureuser}"
REPO_URL="${REPO_URL:?Set your Git repository URL: export REPO_URL=https://github.com/your-username/your-repo.git or put it in deploy.env}"
REMOTE_DIR="${REMOTE_DIR:-}"

ssh -t "${VM_USER}@${VM_HOST}" bash -s -- "${REPO_URL}" "${REMOTE_DIR}" <<'EOF'
set -euo pipefail

REPO_URL="$1"
REMOTE_DIR="${2:-${HOME}/yolo-api}"
REMOTE_DIR="${REMOTE_DIR/#\~\//${HOME}/}"
REMOTE_DIR="${REMOTE_DIR/#\~/${HOME}}"
case "${REMOTE_DIR}" in
    /*) ;;
    *) REMOTE_DIR="${HOME}/${REMOTE_DIR}" ;;
esac

if ! command -v git > /dev/null 2>&1; then
    echo "Error: git is not installed on the VM." >&2
    exit 1
fi

if ! command -v docker > /dev/null 2>&1; then
    echo "Docker not found on the VM — installing Docker Engine..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "${USER}"
fi

DOCKER="docker"
if ! docker info > /dev/null 2>&1; then
    DOCKER="sudo docker"
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
${DOCKER} compose up -d --build
${DOCKER} compose ps
EOF