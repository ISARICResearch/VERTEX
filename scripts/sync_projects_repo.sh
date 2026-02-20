#!/usr/bin/env bash
set -euo pipefail

PROJECTS_DIR="${VERTEX_PROJECTS_DIR:-/opt/vertex-projects}"
PROJECTS_REPO_URL="${PROJECTS_REPO_URL:-}"
PROJECTS_REPO_BRANCH="${PROJECTS_REPO_BRANCH:-main}"

if [[ -z "${PROJECTS_REPO_URL}" ]]; then
  echo "PROJECTS_REPO_URL is not set; cannot sync projects repo." >&2
  exit 1
fi

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
