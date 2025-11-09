#!/usr/bin/env python3
"""
Split all FASTA files in the current folder into parts of N sequences (default 4999).

- Keeps script and FASTA files in the same folder.
- Processes any number of FASTA files automatically.
- Skips files that look like they were produced by this script (e.g., *.part001.fasta).
- Supports .fasta, .fa, .fas, .fna (and optionally .gz).
- Writes sequences with 60-char line wrapping.
"""

import os
import re
import argparse
import gzip
from typing import Iterator, Tuple, TextIO

# File extensions to consider
PLAIN_EXTS = (".fasta", ".fa", ".fas", ".fna")
PART_PATTERN = re.compile(r"\.part\d{3}\.", re.IGNORECASE)

def open_maybe_gzip(path: str, mode: str = "rt") -> TextIO:
    """Open plain text or gzip transparently."""
    if path.lower().endswith(".gz"):
        return gzip.open(path, mode=mode, encoding="utf-8", newline="")
    return open(path, mode=mode, encoding="utf-8", newline="")

def fasta_records(handle: TextIO) -> Iterator[Tuple[str, str]]:
    """
    Stream-parse FASTA. Yields (header_without_>, sequence_string).
    Preserves only A-Z/0-9 and standard FASTA characters as-is; trims whitespace.
    """
    header = None
    seq_chunks = []
    for line in handle:
        if not line:
            continue
        line = line.rstrip("\r\n")
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_chunks)
            header = line[1:].strip()
            seq_chunks = []
        else:
            seq_chunks.append(line.strip())
    if header is not None:
        yield header, "".join(seq_chunks)

def wrap60(seq: str) -> str:
    return "\n".join(seq[i:i+60] for i in range(0, len(seq), 60)) or ""

def safe_out_name(in_path: str, part_idx: int, force_ext: str = ".fasta") -> str:
    """
    Create output name like: basename.part001.fasta (next to input).
    If input is foo.fa.gz → base 'foo.fa' first, then add .part###
    """
    folder, fname = os.path.split(in_path)
    # strip .gz if present
    if fname.lower().endswith(".gz"):
        fname = fname[:-3]
    # ensure we don’t carry over previous part segments
    base, ext = os.path.splitext(fname)
    if PART_PATTERN.search(fname):
        # rare case: user fed an already-parted file
        base = PART_PATTERN.sub(".", base)
    # keep original extension if it's a plain fasta-like ext, else force .fasta
    ext = ext if ext.lower() in PLAIN_EXTS else force_ext
    out = f"{base}.part{part_idx:03d}{ext}"
    return os.path.join(folder, out)

def should_skip(name: str) -> bool:
    """Skip previously split parts and non-target extensions."""
    low = name.lower()
    if PART_PATTERN.search(low):
        return True
    if low.endswith(".gz"):
        low = low[:-3]
    return not low.endswith(PLAIN_EXTS)

def split_one_file(path: str, chunk_size: int, dry_run: bool = False) -> Tuple[int, int]:
    """
    Split a single FASTA file; returns (num_parts_written, total_records).
    """
    total = 0
    part_count = 0
    record_in_part = 0
    out_handle = None
    part_idx = 1

    def open_next_part():
        nonlocal out_handle, part_idx, part_count, record_in_part
        if out_handle:
            out_handle.close()
        out_path = safe_out_name(path, part_idx)
        if not dry_run:
            out_handle = open(out_path, "w", encoding="utf-8", newline="\n")
        part_count += 1
        record_in_part = 0
        part_idx += 1
        return out_path

    with open_maybe_gzip(path, "rt") as fh:
        for header, seq in fasta_records(fh):
            if record_in_part == 0:
                out_path = open_next_part()
                print(f"  -> Writing {out_path}")
            # write record
            if not dry_run:
                out_handle.write(f">{header}\n")
                if seq:
                    out_handle.write(wrap60(seq) + "\n")
                else:
                    out_handle.write("\n")
            record_in_part += 1
            total += 1
            if record_in_part >= chunk_size:
                # close and move to next part on next record
                if out_handle:
                    out_handle.close()
                    out_handle = None

    # Close last part if open but had no records? (should not happen)
    if out_handle:
        out_handle.close()

    # If the file had fewer than chunk_size records, we still created exactly one .part001
    return part_count, total

def find_fasta_files(include_gz: bool) -> list:
    files = []
    for name in os.listdir("."):
        if os.path.isfile(name):
            if PART_PATTERN.search(name):
                continue
            low = name.lower()
            if include_gz and low.endswith(".gz"):
                core = low[:-3]
                if core.endswith(PLAIN_EXTS):
                    files.append(name)
            elif low.endswith(PLAIN_EXTS):
                files.append(name)
    return sorted(files)

def main():
    ap = argparse.ArgumentParser(
        description="Split all FASTA files in the current folder into parts of N sequences."
    )
    ap.add_argument("--chunk", type=int, default=4999,
                    help="Number of FASTA records per output file (default: 4999)")
    ap.add_argument("--include-gz", action="store_true",
                    help="Also process .gz-compressed FASTA files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan and report, but do not write files")
    args = ap.parse_args()

    files = find_fasta_files(include_gz=args.include_gz)
    if not files:
        print("No FASTA files found in this folder.")
        return

    print(f"Found {len(files)} FASTA file(s):")
    for f in files:
        print(" -", f)

    for f in files:
        print(f"\nProcessing: {f}")
        parts, total = split_one_file(f, args.chunk, dry_run=args.dry_run)
        if total == 0:
            print("  (No records found; skipped.)")
        else:
            print(f"  Done: {total} sequences → {parts} part file(s).")

if __name__ == "__main__":
    main()