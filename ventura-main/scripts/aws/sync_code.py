#!/usr/bin/env python3
# selective_rsync.py   (updated)

import argparse, subprocess, shlex, sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Selective rsync by extension.")
    p.add_argument("--src",  required=True,
                   help="Where to mirror the data locally.")
    p.add_argument("--dst",  required=True,
                   help="Remote source (user@host:/path OR /path).")
    p.add_argument("--subdirs",     required=False, default="",
                   help="Comma-separated list of sub-dirs to sync, relative to root.")
    p.add_argument("--exts",        required=False, default="",
                   help="Comma-separated list of file extensions to keep (without dots). "
                        "If omitted, defaults will be used.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print rsync command but don’t execute.")
    return p.parse_args()


def rsync_subdir(local_root: Path, remote_root: str, subdir: str,
                 exts: list[str], dry: bool):
    local_path  = local_root / subdir
    remote_path = f"{remote_root.rstrip('/')}/{subdir}"

    include_rules = [f"--include=*.{ext.lstrip('.')}" for ext in exts]

    cmd = [
        "rsync", "-azvhP",
        "--include=*/",        # always walk directories
        *include_rules,
        "--exclude=*",         # drop anything else
        str(local_path) + "/",     # source
        remote_path + "/", # destination
    ]
    if dry:
        cmd.insert(1, "--dry-run")

    print("[rsync]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)


"""
python scripts/aws/sync_code.py --src ./ --dst ec2-user@52.13.84.237:/home/ec2-user/playground/spinflow
"""
def main():
    args = parse_args()

    src = Path(args.src).expanduser().resolve()
    src.mkdir(parents=True, exist_ok=True)

    default_subdirs = ["scripts", "spinflow", "config", "docs"]
    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    subdirs = default_subdirs + subdirs if subdirs else default_subdirs

    default_exts = ["py", "yaml", "md", "sh", "txt", "json", "csv"]
    exts = [e.strip().lstrip(".") for e in args.exts.split(",") if e.strip()]
    exts = default_exts + exts if exts else default_exts

    if not subdirs:
        sys.exit("No subdirectories given.")
    if not exts:
        sys.exit("No extensions given.")

    for sd in subdirs:
        rsync_subdir(src, args.dst, sd, exts, args.dry_run)


if __name__ == "__main__":
    main()
