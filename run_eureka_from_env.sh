#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required in the environment." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://ai-api.vaa.la/v1}"

EUREKA_ENV="${EUREKA_ENV:-globe_walking}"
EUREKA_MODEL="${EUREKA_MODEL:-mimo-v2.5-pro}"
EUREKA_MAX_TOKENS="${EUREKA_MAX_TOKENS:-8192}"

cmd=(
  python eureka.py
  "env=${EUREKA_ENV}"
  "model=${EUREKA_MODEL}"
  "api_base=${OPENAI_API_BASE}"
  "max_tokens=${EUREKA_MAX_TOKENS}"
)

if [[ -n "${EUREKA_ITERATION:-}" ]]; then
  cmd+=("iteration=${EUREKA_ITERATION}")
fi

if [[ -n "${EUREKA_SAMPLE:-}" ]]; then
  cmd+=("sample=${EUREKA_SAMPLE}")
fi

if [[ -n "${EUREKA_TEMPERATURE:-}" ]]; then
  cmd+=("temperature=${EUREKA_TEMPERATURE}")
fi

cd "${SCRIPT_DIR}/eureka"
exec "${cmd[@]}" "$@"
