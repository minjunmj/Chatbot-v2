#!/usr/bin/env bash

set -Eeuo pipefail

readonly DATASET_KEY="71723"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly DOWNLOAD_DIR="${PROJECT_ROOT}/data"
readonly COMPLETE_MARKER="${DOWNLOAD_DIR}/.aihubshell-${DATASET_KEY}-complete"
readonly DOWNLOAD_ARCHIVE="${DOWNLOAD_DIR}/.aihubshell-${DATASET_KEY}.tar"
readonly DOWNLOAD_URL="https://api.aihub.or.kr/down/0.6/${DATASET_KEY}.do?fileSn=all"

extract_zip_files() {
  python3 "${SCRIPT_DIR}/extract_zip_files.py" "${DOWNLOAD_DIR}"
}

force_download=false
extract_only=false

case "${1:-}" in
  --force)
    force_download=true
    ;;
  --extract-only)
    extract_only=true
    ;;
  '')
    ;;
  *)
    echo "Usage: $0 [--force|--extract-only]" >&2
    exit 2
    ;;
esac

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--force|--extract-only]" >&2
  exit 2
fi

if [[ "${extract_only}" == true ]]; then
  if [[ ! -d "${DOWNLOAD_DIR}" ]]; then
    echo "데이터 디렉터리를 찾을 수 없습니다: ${DOWNLOAD_DIR}" >&2
    exit 1
  fi

  extract_zip_files
  exit 0
fi

if [[ -z "${AIHUB_API_KEY:-}" ]]; then
  cat >&2 <<'EOF'
AIHUB_API_KEY 환경변수가 필요합니다.

1. AI Hub에서 데이터셋 71723의 다운로드를 신청하고 승인을 받으세요.
2. AI Hub 오픈 API 페이지에서 API Key를 발급받으세요.
3. 아래처럼 실행하세요.

   AIHUB_API_KEY='발급받은-키' ./scripts/download_aihub_data.sh
EOF
  exit 1
fi

if [[ -f "${COMPLETE_MARKER}" && "${force_download}" == false ]]; then
  echo "이미 다운로드가 완료되었습니다: ${DOWNLOAD_DIR}"
  echo "다시 받으려면 --force 옵션을 사용하세요."
  exit 0
fi

for command_name in curl python3 tar; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "필수 명령어를 찾을 수 없습니다: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p "${DOWNLOAD_DIR}"

echo "AI Hub 데이터셋 ${DATASET_KEY} 다운로드를 시작합니다."
echo "저장 위치: ${DOWNLOAD_DIR}"

rm -f "${COMPLETE_MARKER}"

curl \
  --fail \
  --location \
  --retry 3 \
  --continue-at - \
  --output "${DOWNLOAD_ARCHIVE}" \
  --header "apikey:${AIHUB_API_KEY}" \
  "${DOWNLOAD_URL}"

echo "다운로드 묶음을 해제합니다."
tar -xf "${DOWNLOAD_ARCHIVE}" -C "${DOWNLOAD_DIR}"

echo "분할 파일을 안전하게 병합합니다."
python3 "${SCRIPT_DIR}/merge_aihub_parts.py" "${DOWNLOAD_DIR}"

if [[ -n "$(find "${DOWNLOAD_DIR}" -type f -iname '*.zip' -size 0 -print -quit)" ]]; then
  echo "0바이트 ZIP 파일이 발견되어 다운로드를 완료 처리하지 않습니다." >&2
  echo "다시 다운로드하려면 --force 옵션을 사용하세요." >&2
  exit 1
fi

extract_zip_files
rm -f "${DOWNLOAD_ARCHIVE}"
touch "${COMPLETE_MARKER}"
echo "다운로드가 완료되었습니다: ${DOWNLOAD_DIR}"
