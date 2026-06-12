#!/usr/bin/env bash
# Vendor the RewardBench-2 judge-score snapshot used by this project.
#
# Source: https://github.com/composo-ai/llm-judge-criteria-ensembling
# Pinned to a fixed commit for reproducibility. Files are written read-only
# (0444) so the analysis cannot mutate the raw collections in place.
#
# Uses curl rather than gh/urllib: on the dev machine the gh CLI and Python's
# ssl/urllib lack a working CA bundle. The upstream repo is public, so no auth
# is needed.
set -euo pipefail

REPO="composo-ai/llm-judge-criteria-ensembling"
SHA="e4049a57dc73097f899a605f5435923f03e9233c"
BASE="https://raw.githubusercontent.com/${REPO}/${SHA}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROV="${ROOT}/data/PROVENANCE.txt"

# remote path | local path (relative to repo root)
FILES=(
  "results/raw/base_both_k8.jsonl|data/raw/base_both_k8.jsonl"
  "results/raw/base_claude_both_k8.jsonl|data/raw/base_claude_both_k8.jsonl"
  "results/raw/base_nano_k8.jsonl|data/raw/base_nano_k8.jsonl"
  "results/raw/criteria_both_k8.jsonl|data/raw/criteria_both_k8.jsonl"
  "results/raw/criteria_claude_both_k8.jsonl|data/raw/criteria_claude_both_k8.jsonl"
  "results/tables/all_metrics.json|data/tables/all_metrics.json"
)

mkdir -p "${ROOT}/data/raw" "${ROOT}/data/tables"

# Rewrite provenance fresh each run.
{
  echo "source_repo = https://github.com/${REPO}"
  echo "commit_sha  = ${SHA}"
  echo "branch      = main"
  echo "retrieved   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
} > "${PROV}"

for entry in "${FILES[@]}"; do
  remote="${entry%%|*}"
  local="${entry##*|}"
  dest="${ROOT}/${local}"
  echo "fetching ${remote} -> ${local}"
  # The dest may already exist read-only from a prior run; make it writable.
  [ -f "${dest}" ] && chmod u+w "${dest}"
  curl --fail --silent --show-error --location "${BASE}/${remote}" -o "${dest}"
  bytes=$(wc -c < "${dest}" | tr -d ' ')
  sha=$(shasum -a 256 "${dest}" | awk '{print $1}')
  printf 'file = %s  sha256=%s  bytes=%s\n' "${local}" "${sha}" "${bytes}" >> "${PROV}"
  chmod 0444 "${dest}"
done

echo ""
echo "Wrote ${#FILES[@]} files; provenance at ${PROV}:"
cat "${PROV}"
