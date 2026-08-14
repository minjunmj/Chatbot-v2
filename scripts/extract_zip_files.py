#!/usr/bin/env python3
"""Safely extract AI Hub ZIP files with Unicode filenames."""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


def safe_destination(base: Path, archive_name: str) -> Path:
    # AI Hub archives sometimes contain names beginning with '/'. Treat those
    # as paths relative to the ZIP instead of absolute filesystem paths.
    normalized = archive_name.replace("\\", "/").lstrip("/")
    parts = PurePosixPath(normalized).parts

    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"안전하지 않은 ZIP 내부 경로: {archive_name!r}")

    destination = base.joinpath(*parts)
    resolved_base = base.resolve()
    resolved_destination = destination.resolve()
    if resolved_destination != resolved_base and resolved_base not in resolved_destination.parents:
        raise ValueError(f"압축 해제 경로를 벗어나는 항목: {archive_name!r}")
    return destination


def extract_zip(archive: Path) -> int:
    extracted = 0
    print(f"ZIP 압축 해제: {archive}", flush=True)

    with zipfile.ZipFile(archive) as zip_file:
        bad_member = zip_file.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"손상된 ZIP 항목: {bad_member}")

        for member in zip_file.infolist():
            destination = safe_destination(archive.parent, member.filename)

            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue

            # Do not materialize symlinks stored in a ZIP archive.
            unix_mode = member.external_attr >> 16
            if (unix_mode & 0o170000) == 0o120000:
                raise ValueError(f"심볼릭 링크 ZIP 항목은 허용하지 않습니다: {member.filename!r}")

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.extracting")
            try:
                with zip_file.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.replace(temporary, destination)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            extracted += 1

    return extracted


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} DATA_DIRECTORY", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"데이터 디렉터리를 찾을 수 없습니다: {root}", file=sys.stderr)
        return 1

    archives = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".zip")
    total_files = 0
    for archive in archives:
        total_files += extract_zip(archive)

    print(f"ZIP 파일 {len(archives)}개에서 파일 {total_files}개를 압축 해제했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
