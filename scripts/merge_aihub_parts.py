#!/usr/bin/env python3
"""Merge AI Hub files named <filename>.part<offset> without corrupting them."""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


PART_PATTERN = re.compile(r"^(?P<target>.+)\.part(?P<offset>\d+)$")


def merge_parts(root: Path) -> int:
    groups: dict[Path, list[tuple[int, Path]]] = defaultdict(list)

    for part in root.rglob("*.part*"):
        match = PART_PATTERN.match(part.name)
        if match and part.is_file():
            target = part.with_name(match.group("target"))
            groups[target].append((int(match.group("offset")), part))

    for target, parts in groups.items():
        parts.sort(key=lambda item: item[0])
        expected_offset = 0

        for offset, part in parts:
            if offset != expected_offset:
                raise RuntimeError(
                    f"분할 파일이 누락되었습니다: {part} "
                    f"(예상 offset={expected_offset}, 실제 offset={offset})"
                )
            expected_offset += part.stat().st_size

        temporary = target.with_name(f".{target.name}.merging")
        print(f"분할 파일 {len(parts)}개 병합: {target}", flush=True)

        try:
            with temporary.open("wb") as destination:
                for _, part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        for _, part in parts:
            part.unlink()

    return len(groups)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} DATA_DIRECTORY", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"데이터 디렉터리를 찾을 수 없습니다: {root}", file=sys.stderr)
        return 1

    count = merge_parts(root)
    print(f"분할 파일 묶음 {count}개를 병합했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
