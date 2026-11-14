"""Fetch the MovieLens archive once, check it against the pinned MD5, unpack it."""
from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from src.common.io import file_md5
from src.common.logging_setup import get_logger

log = get_logger(__name__)


def download_archive(url: str, dest: Path, expected_md5: str | None) -> Path:
    if dest.exists() and (expected_md5 is None or file_md5(dest) == expected_md5):
        log.info("Archive already present and verified: %s", dest.name)
        return dest

    log.info("Downloading %s", url)
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    got = file_md5(tmp)
    if expected_md5 and got != expected_md5:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {url}: expected {expected_md5}, got {got}")
    tmp.replace(dest)
    log.info("Saved %s (%.1f MB, md5 %s)", dest.name, dest.stat().st_size / 1e6, got)
    return dest


def extract_archive(archive: Path, target_dir: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            # refuse anything that would land outside the target folder
            resolved = (target_dir / member).resolve()
            if not str(resolved).startswith(str(target_dir.resolve())):
                raise ValueError(f"Unsafe path in archive: {member}")
        zf.extractall(target_dir)
    return target_dir
