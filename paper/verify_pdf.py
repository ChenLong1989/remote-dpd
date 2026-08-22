"""Verify text integrity and key facts in the generated paper PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


REQUIRED_FACTS = (
    "2240",
    "54.0%",
    "69.0%",
    "-116.6",
    "-318.7",
    "4008/4008",
    "125/125",
    "artifacts/frozen_release",
)
REQUIRED_UNICODE_TEXT = ("正向模型", "摘要", "预注册", "局限性", "参考文献")
FORBIDDEN_TEXT = ("RESULT_", "ResultPlaceholder", "PLACEHOLDER", "\ufffd")


def verify(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size < 100_000:
        raise ValueError("the final paper PDF is missing or unexpectedly small")
    reader = PdfReader(str(path))
    page_text = [(page.extract_text() or "").strip() for page in reader.pages]
    if not page_text or any(len(text) < 50 for text in page_text):
        raise ValueError("one or more PDF pages have no meaningful extractable text")
    combined = "\n".join(page_text)
    for token in FORBIDDEN_TEXT:
        if token in combined:
            raise ValueError(f"forbidden placeholder or replacement token found: {token!r}")
    missing_facts = [fact for fact in REQUIRED_FACTS if fact not in combined]
    if missing_facts:
        raise ValueError(f"required paper facts are missing: {missing_facts}")
    missing_unicode = [text for text in REQUIRED_UNICODE_TEXT if text not in combined]
    if missing_unicode:
        raise ValueError(f"required Unicode text is not extractable: {missing_unicode}")
    with pdfplumber.open(path) as document:
        plumber_lengths = [len((page.extract_text() or "").strip()) for page in document.pages]
        page_sizes = [(round(page.width, 2), round(page.height, 2)) for page in document.pages]
    if len(set(page_sizes)) != 1:
        raise ValueError("PDF page sizes are inconsistent")
    if len(plumber_lengths) != len(page_text) or any(length < 50 for length in plumber_lengths):
        raise ValueError("pdfplumber extraction found an empty page")
    metadata = reader.metadata or {}
    if "在线功率放大器" not in str(metadata.get("/Title", "")):
        raise ValueError("PDF title metadata is missing")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "pages": len(page_text),
        "page_size_points": page_sizes[0],
        "pypdf_characters": sum(len(text) for text in page_text),
        "pdfplumber_characters": sum(plumber_lengths),
        "required_facts_verified": list(REQUIRED_FACTS),
        "required_unicode_text_verified": list(REQUIRED_UNICODE_TEXT),
        "forbidden_tokens_absent": list(FORBIDDEN_TEXT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=Path("output") / "pdf" / "pa_model_backprop_ilc.pdf",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.pdf), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
