#!/usr/bin/env bash
set -euo pipefail

PROJECTS_DIR="${VERTEX_PROJECTS_DIR:-/opt/vertex-projects}"
PROJECTS_REPO_URL="git@github.com:ISARICResearch/VERTEX-projects.git"
PROJECTS_REPO_BRANCH="${PROJECTS_REPO_BRANCH:-main}"
PROJECTS_GIT_SSH_KEY_PATH="${PROJECTS_GIT_SSH_KEY_PATH:-/root/.ssh/vertex_projects_deploy_key}"
PROJECTS_GIT_KNOWN_HOSTS_PATH="${PROJECTS_GIT_KNOWN_HOSTS_PATH:-/root/.ssh/known_hosts}"

if [[ ! -f "${PROJECTS_GIT_SSH_KEY_PATH}" ]]; then
  echo "Missing SSH deploy key: ${PROJECTS_GIT_SSH_KEY_PATH}" >&2
  exit 1
fi

export GIT_SSH_COMMAND="ssh -i ${PROJECTS_GIT_SSH_KEY_PATH} -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=${PROJECTS_GIT_KNOWN_HOSTS_PATH}"

mkdir -p "${PROJECTS_DIR}"

if [[ -d "${PROJECTS_DIR}/.git" ]]; then
  git -C "${PROJECTS_DIR}" fetch --all --prune
  git -C "${PROJECTS_DIR}" checkout "${PROJECTS_REPO_BRANCH}"
  git -C "${PROJECTS_DIR}" pull --ff-only origin "${PROJECTS_REPO_BRANCH}"
else
  rm -rf "${PROJECTS_DIR}"
  git clone --branch "${PROJECTS_REPO_BRANCH}" "${PROJECTS_REPO_URL}" "${PROJECTS_DIR}"
fi

echo "Projects repo synced at ${PROJECTS_DIR} (branch: ${PROJECTS_REPO_BRANCH})"
