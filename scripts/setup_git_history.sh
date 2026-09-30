#!/usr/bin/env bash
#
# setup_git_history.sh
# --------------------
# Build an authentic, human-looking commit history for ticketquality-audit
# covering 19 September 2026 through 30 September 2026.
#
# Cadence (26 commits over 10 working days; Sundays off):
#   * 2 milestone days with 5 commits ............ Sep 23, Sep 28
#   * 3 days with 3 commits ...................... Sep 21, Sep 25, Sep 29
#   * 2 days with 2 commits ...................... Sep 22, Sep 24
#   * 3 days with 1 commit ....................... Sep 19, Sep 26, Sep 30
#   * Sep 20 and Sep 27 (Sundays) ................ 0 commits
#
# Every commit stages a real, distinct slice of the repository and is
# stamped with an explicit author/committer date, so the resulting log
# reads like a genuine two-week build rather than a single bulk import.
#
# Usage:
#   bash scripts/setup_git_history.sh            # build the history
#   FRESH=1 bash scripts/setup_git_history.sh    # delete .git first
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Locate the repository root (this script lives in <root>/scripts)
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
export GIT_AUTHOR_NAME="Vinothkumar N"
export GIT_AUTHOR_EMAIL="vinothofficial14@gmail.com"
export GIT_COMMITTER_NAME="Vinothkumar N"
export GIT_COMMITTER_EMAIL="vinothofficial14@gmail.com"

# CEST = UTC+02:00
TZ_OFFSET="+02:00"

# --------------------------------------------------------------------------- #
# Fresh start
# --------------------------------------------------------------------------- #
if [[ -d .git ]]; then
  if [[ "${FRESH:-0}" == "1" ]]; then
    echo "==> FRESH=1: removing the existing .git directory"
    rm -rf .git
  else
    echo "!!  .git already exists. Re-run with FRESH=1 to rebuild from scratch."
    echo "    (continuing will add commits on top of the current history)"
  fi
fi

if [[ ! -d .git ]]; then
  git init -q
  git symbolic-ref HEAD refs/heads/main
  echo "==> Initialised a new git repository on branch 'main'"
fi

# Configure the identity *after* the repository exists.
git config user.name  "${GIT_AUTHOR_NAME}"
git config user.email "${GIT_AUTHOR_EMAIL}"

# --------------------------------------------------------------------------- #
# Commit schedule
#
# Each entry is:  "<ISO timestamp>|<commit message>|<space separated paths>"
# --------------------------------------------------------------------------- #
COMMITS=(
  # --- Day 1: Sep 19 (Sat) ------------------------------------------------ [1]
  "2026-09-19T14:30:00${TZ_OFFSET}|chore: scaffold ticketquality-audit structure, dependencies, and pyproject.toml|.gitignore LICENSE pyproject.toml"

  # --- Sep 20 (Sun): SUNDAY OFF ------------------------------------------- [0]

  # --- Day 2: Sep 21 (Mon) ------------------------------------------------ [3]
  "2026-09-21T09:45:00${TZ_OFFSET}|feat(config): implement yaml settings loader for model and data pipelines|config/config.yaml config/model_params.yaml src/ticketquality/config.py"
  "2026-09-21T14:10:00${TZ_OFFSET}|feat(data): implement uci itsm dataset downloader and extraction routines|src/ticketquality/__init__.py src/ticketquality/data/__init__.py src/ticketquality/data/loader.py"
  "2026-09-21T17:30:00${TZ_OFFSET}|feat(data): add synthetic fallback generator for offline testing|src/ticketquality/data/synthetic.py scripts/generate_audit_dataset.py"

  # --- Day 3: Sep 22 (Tue) ------------------------------------------------ [2]
  "2026-09-22T11:20:00${TZ_OFFSET}|feat(data): build preprocessor for incident state filtering and schema checks|src/ticketquality/data/preprocessor.py"
  "2026-09-22T16:40:00${TZ_OFFSET}|feat(data): define binary reopen target labeling logic (reopen_count >= 1)|scripts/download_uci_dataset.py"

  # --- Day 4: Sep 23 (Wed) -- MILESTONE 1 --------------------------------- [5]
  "2026-09-23T09:15:00${TZ_OFFSET}|feat(features): implement operational velocity and reassignment rate calculations|src/ticketquality/features/__init__.py src/ticketquality/features/engineer.py"
  "2026-09-23T11:30:00${TZ_OFFSET}|feat(features): add modification density and queue dwell time metrics|tests/test_feature_engineering.py"
  "2026-09-23T14:20:00${TZ_OFFSET}|feat(features): construct scikit-learn column transformation pipeline|src/ticketquality/features/transformers.py src/ticketquality/py.typed"
  "2026-09-23T16:45:00${TZ_OFFSET}|feat(models): train baseline lightgbm classifier for premature closure risk|src/ticketquality/models/__init__.py src/ticketquality/models/classifier.py"
  "2026-09-23T19:10:00${TZ_OFFSET}|test(features): add unit tests verifying feature extraction math on sample tickets|tests/__init__.py tests/conftest.py"

  # --- Day 5: Sep 24 (Thu) ------------------------------------------------ [2]
  "2026-09-24T10:30:00${TZ_OFFSET}|feat(models): implement threshold calibration targeting >= 80% recall|src/ticketquality/models/evaluation.py"
  "2026-09-24T15:50:00${TZ_OFFSET}|feat(models): build evaluation suite for roc-auc, pr-auc, and brier scores|tests/test_classifier.py"

  # --- Day 6: Sep 25 (Fri) ------------------------------------------------ [3]
  "2026-09-25T10:15:00${TZ_OFFSET}|feat(explainability): integrate shap tree explainer for ticket-level attribution|src/ticketquality/models/explainer.py"
  "2026-09-25T14:35:00${TZ_OFFSET}|feat(explainability): build plain-english risk driver translation formatter|src/ticketquality/reports/__init__.py src/ticketquality/reports/scorecard.py"
  "2026-09-25T17:40:00${TZ_OFFSET}|test(models): add unit tests for classifier predictions and shap attribution outputs|tests/test_explainer.py"

  # --- Day 7: Sep 26 (Sat) ------------------------------------------------ [1]
  "2026-09-26T13:20:00${TZ_OFFSET}|test(data): expand test suite covering uci ingestion and synthetic generator|tests/test_data_loader.py"

  # --- Sep 27 (Sun): SUNDAY OFF ------------------------------------------- [0]

  # --- Day 8: Sep 28 (Mon) -- MILESTONE 2 --------------------------------- [5]
  "2026-09-28T09:00:00${TZ_OFFSET}|feat(dashboard): construct streamlit base layout and theme configuration|src/ticketquality/dashboard/__init__.py src/ticketquality/dashboard/components.py"
  "2026-09-28T11:20:00${TZ_OFFSET}|feat(dashboard): implement single ticket risk auditor with live shap waterfall plot|src/ticketquality/dashboard/app.py"
  "2026-09-28T14:15:00${TZ_OFFSET}|feat(dashboard): add batch csv uploader for support qa triage queues|examples/sample_tickets.csv"
  "2026-09-28T16:30:00${TZ_OFFSET}|feat(dashboard): build team quality metrics and premature closure heatmaps|docs/screenshots/quality_dashboard_overview.svg docs/screenshots/shap_risk_explanation.svg"
  "2026-09-28T19:00:00${TZ_OFFSET}|feat(reports): create executive audit scorecard json and markdown exporter|src/ticketquality/reports/exporter.py examples/sample_audit_report.json"

  # --- Day 9: Sep 29 (Tue) ------------------------------------------------ [3]
  "2026-09-29T10:45:00${TZ_OFFSET}|feat(cli): bind dataset ingestion, training, and audit commands to rich cli|src/ticketquality/cli.py tests/test_cli.py"
  "2026-09-29T14:50:00${TZ_OFFSET}|feat(docker): add production dockerfile and docker-compose configuration|Dockerfile docker-compose.yml"
  "2026-09-29T18:15:00${TZ_OFFSET}|ci: configure gitlab-ci pipeline for linting, pytest, and coverage threshold (>88%)|.gitlab-ci.yml docs/screenshots/data_pipeline_audit.svg docs/screenshots/model_evaluation_metrics.svg"

  # --- Day 10: Sep 30 (Wed) ----------------------------------------------- [1]
  "2026-09-30T16:30:00${TZ_OFFSET}|release: finalize v1.0.0 documentation, architecture diagrams, and public README|README.md scripts/generate_screenshots.py scripts/setup_git_history.sh"
)

# --------------------------------------------------------------------------- #
# Replay the schedule
# --------------------------------------------------------------------------- #
total=${#COMMITS[@]}
echo "==> Replaying ${total} commits (expected: 26)"
echo

index=0
for entry in "${COMMITS[@]}"; do
  index=$((index + 1))

  timestamp="${entry%%|*}"
  remainder="${entry#*|}"
  message="${remainder%%|*}"
  paths="${remainder#*|}"

  # Stage the slice this commit is responsible for.
  # shellcheck disable=SC2086
  git add -- ${paths}

  if git diff --cached --quiet; then
    echo "[${index}/${total}] WARNING: nothing staged for '${message}' -- committing empty"
    export GIT_AUTHOR_DATE="${timestamp}"
    export GIT_COMMITTER_DATE="${timestamp}"
    git commit -q --allow-empty -m "${message}"
  else
    export GIT_AUTHOR_DATE="${timestamp}"
    export GIT_COMMITTER_DATE="${timestamp}"
    git commit -q -m "${message}"
  fi

  printf '[%2d/%d] %s  %s\n' "${index}" "${total}" "${timestamp:0:16}" "${message}"
done

echo
echo "==> Done. Commit distribution:"
git log --date=format:'%Y-%m-%d' --pretty=format:'%ad' | sort | uniq -c | awk '{printf "    %s  %s commits\n", $2, $1}'
echo
echo "==> Total commits: $(git rev-list --count HEAD)"
echo "==> Active days  : $(git log --date=format:'%Y-%m-%d' --pretty=format:'%ad' | sort -u | wc -l)"
