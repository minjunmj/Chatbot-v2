#!/usr/bin/env bash

set -Eeuo pipefail

readonly DATASET_KEY="71723"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly TOOL_DIR="${PROJECT_ROOT}/.tools"
readonly AIHUB_SHELL="${TOOL_DIR}/aihubshell"
readonly DOWNLOAD_DIR="${PROJECT_ROOT}/data"
readonly COMPLETE_MARKER="${DOWNLOAD_DIR}/.aihubshell-${DATASET_KEY}-complete"
readonly AIHUB_SHELL_URL="https://api.aihub.or.kr/api/aihubshell.do"

if [[ "${1:-}" == "--force" ]]; then
  force_download=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--force]" >&2
  exit 2
else
  force_download=false
fi

if [[ -z "${AIHUB_API_KEY:-}" ]]; then
  cat >&2 <<'EOF'
AIHUB_API_KEY 환경변수가 필요합니다.

1. AI Hub에서 데이터셋 71723의 다운로드를 신청하고 승인을 받으세요.
2. AI Hub 오픈 API 페이지에서 API Key를 발급받으세요.
3. 아래처럼 실행하세요.

   AIHUB_API_KEY='2DAEB1F1-4451-4B72-99DA-E1C32161B67D' ./scripts/download_aihub_data.sh
EOF
  exit 1
fi

if [[ -f "${COMPLETE_MARKER}" && "${force_download}" == false ]]; then
  echo "이미 다운로드가 완료되었습니다: ${DOWNLOAD_DIR}"
  echo "다시 받으려면 --force 옵션을 사용하세요."
  exit 0
fi

for command_name in curl unzip; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "필수 명령어를 찾을 수 없습니다: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p "${TOOL_DIR}" "${DOWNLOAD_DIR}"

if [[ ! -x "${AIHUB_SHELL}" ]]; then
  echo "AI Hub 공식 다운로더를 설치합니다: ${AIHUB_SHELL}"
  curl --fail --location --retry 3 --output "${AIHUB_SHELL}.tmp" "${AIHUB_SHELL_URL}"
  chmod 700 "${AIHUB_SHELL}.tmp"
  mv "${AIHUB_SHELL}.tmp" "${AIHUB_SHELL}"
fi

echo "AI Hub 데이터셋 ${DATASET_KEY} 다운로드를 시작합니다."
echo "저장 위치: ${DOWNLOAD_DIR}"

(
  cd "${DOWNLOAD_DIR}"
  "${AIHUB_SHELL}" \
    -mode d \
    -datasetkey "${DATASET_KEY}" \
    -aihubapikey "${AIHUB_API_KEY}"
)

touch "${COMPLETE_MARKER}"
echo "다운로드가 완료되었습니다: ${DOWNLOAD_DIR}"
