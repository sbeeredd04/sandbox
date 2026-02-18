#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Error on line $LINENO: ${BASH_COMMAND:-}"; exit 1' ERR

# Export a unified environment.yml that includes:
# - channels (conda-forge + robostack-noetic if detected)
# - conda dependencies (from history + detected meta packages like ros-noetic-desktop)
# - pip dependencies (portable; preserves external VCS/URL; drops local file refs)
# - optional PIP_EXTRA_INDEX_URL for PyTorch CUDA wheels (cu121|cu124|cu126)
#
# Usage:
#   ./export_unified_env_yaml.sh
#   ./export_unified_env_yaml.sh --name spinflow --out environment.yml --torch-index cu126
#
# Notes:
# - This is best run *inside the active environment* you want to export.
# - We purposely avoid listing hundreds of conda transitive packages; we use
#   "from-history" specs plus key meta packages (e.g., ros-noetic-desktop).
# - Pip lines preserve Git/HTTP URLs but drop local paths (file:/// or absolute).
# - We also drop ROS Python dists that came from robostack-noetic from pip section.

OUT_YML="environment.yml"
ENV_NAME=""            # if empty, derive from conda; else use provided name
TORCH_INDEX="auto"     # auto|none|cpu|cu121|cu124|cu126

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_YML="$2"; shift 2;;
    --name) ENV_NAME="$2"; shift 2;;
    --torch-index) TORCH_INDEX="$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $0 [--out environment.yml] [--name ENV_NAME] [--torch-index auto|cpu|cu121|cu124|cu126]
EOF
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# --- sanity: tools
command -v python >/dev/null || { echo "python not found"; exit 1; }
command -v pip >/dev/null || { echo "pip not found"; exit 1; }

HAS_CONDA="false"
if command -v conda >/dev/null 2>&1; then
  HAS_CONDA="true"
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
FREEZE="$TMP/pip_freeze.txt"
CONDA_LIST_JSON="$TMP/conda_list.json"
CONDA_ENV_HISTORY="$TMP/conda_env_hist.yml"
PIP_BODY="$TMP/pip_body.txt"
YAML_BODY="$TMP/yaml_body.txt"

# --- gather pip inventory (to capture Git/HTTP lines)
pip freeze > "$FREEZE"

# --- gather conda context (if available)
ROBOSTACK_PRESENT="false"
ROS_META_PRESENT="false"
PYVER=""

if [[ "$HAS_CONDA" == "true" ]]; then
  # Python version (major.minor)
  PYVER="$(python - <<'PY' 2>/dev/null || echo ''
import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  conda list --json > "$CONDA_LIST_JSON"

  # Detect robostack-noetic presence
  if grep -q '"channel": *"robostack-noetic"' "$CONDA_LIST_JSON"; then
    ROBOSTACK_PRESENT="true"
  fi

  # Export explicit specs from history (what user asked to install)
  # (ok if it fails; we’ll fall back on minimal)
  conda env export --from-history > "$CONDA_ENV_HISTORY" || true

  # Detect ros-noetic-desktop explicitly installed
  if grep -qE '(^|\s|\-)\s*ros-noetic-desktop(\s|$|=|<|>)' "$CONDA_ENV_HISTORY" ; then
    ROS_META_PRESENT="true"
  else
    # if not in history, still mark present if any robostack packages exist
    if [[ "$ROBOSTACK_PRESENT" == "true" ]]; then
      ROS_META_PRESENT="true"
    fi
  fi
fi

# --- build the pip section (portable, VCS/HTTP preserved, local refs dropped)
python - "$FREEZE" "$CONDA_LIST_JSON" "$PIP_BODY" <<'PY'
import json, os, re, sys
freeze_fp, conda_list_fp, out_fp = sys.argv[1:]

def is_comment_or_blank(s): s=s.strip(); return (not s) or s.startswith("#")

def is_local_line(s:str)->bool:
    t=s.strip()
    low=t.lower()
    if "git+file://" in low or "file://" in low:
        return True
    if low.startswith("/"):
        return True
    if low.startswith("-e "):
        rest=t[3:].lstrip()
        if rest.startswith(("/", "./", "../")) or rest.lower().startswith("file://"):
            return True
    if re.search(r'\s@\s*(/|\.{1,2}/|[A-Za-z]:/|file://)', t):
        return True
    if "feedstock_root/build_artifacts" in low or "/home/conda/feedstock" in t:
        return True
    return False

def is_external_url(s:str)->bool:
    if is_local_line(s): return False
    t=s.strip()
    if re.match(r'^[A-Za-z0-9_.\-]+(\[[^\]]+\])?\s*@\s*(git\+(https?|ssh)://|https?://)', t): return True
    if re.match(r'^-e\s+git\+(https?|ssh)://', t): return True
    if re.match(r'^git\+(https?|ssh)://.*#egg=[A-Za-z0-9_.\-]+', t): return True
    return False

# conda packages from robostack -> exclude from pip section
robostack_names = set()
if os.path.exists(conda_list_fp) and os.path.getsize(conda_list_fp) > 0:
    try:
        data = json.load(open(conda_list_fp))
        for rec in data:
            if str(rec.get("channel", "")).lower() == "robostack-noetic":
                nm = str(rec.get("name","")).strip()
                if nm:
                    robostack_names.add(nm.lower().replace("_","-"))
    except Exception:
        pass

def normalize_line(line:str) -> str:
    """Convert exact 'name==X.Y.Z(+local)' -> 'name~=X.Y'; keep ranges/markers as-is."""
    m=re.match(r'^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([0-9][A-Za-z0-9_.\-+]*)$', line)
    if not m:
        return line
    name, extras, ver = m.group(1), (m.group(2) or ""), m.group(3).split("+",1)[0]
    parts=re.split(r'[._-]',ver)
    if len(parts)>=2 and all(p.isdigit() for p in parts[:2]):
        return f"{name}{extras}~={parts[0]}.{parts[1]}"
    return f"{name}{extras}=={ver}"

keep = []
seen = set()

with open(freeze_fp) as f:
    for raw in f:
        line = raw.strip()
        if is_comment_or_blank(line): continue
        if is_local_line(line): continue

        low = line.lower().strip()
        # Drop tooling stubs
        if low in {"pip","wheel","setuptools"} or low.startswith(("pip==","wheel==","setuptools==")):
            continue

        # Exclude ROS python dists that come from robostack (conda), not pip
        # Match line name for forms:
        #   name==ver
        #   name[extra]==ver
        #   name @ https://...
        #   git+https://...#egg=name
        nm = None
        m = re.match(r'^([A-Za-z0-9_.\-]+)', line)
        if m:
            nm = m.group(1).lower().replace("_","-")
        else:
            m2 = re.search(r'#egg=([A-Za-z0-9_.\-]+)', line)
            if m2:
                nm = m2.group(1).lower().replace("_","-")
        if nm and nm in robostack_names:
            continue

        if is_external_url(line):
            if low not in seen:
                keep.append(line); seen.add(low)
            continue

        port = normalize_line(line)
        if port.lower() not in seen:
            keep.append(port); seen.add(port.lower())

keep.sort(key=str.lower)
with open(out_fp, "w") as g:
    g.write("\n".join(keep))
    if keep: g.write("\n")
PY

# --- derive environment name (robust) ---
if [[ -z "${ENV_NAME:-}" ]]; then
  # 1) environment vars first (works for conda/mamba/micromamba)
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    ENV_NAME="$CONDA_DEFAULT_ENV"
  elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    ENV_NAME="$(basename "$CONDA_PREFIX")"
  else
    ENV_NAME=""
  fi
fi

# 2) try conda info --json only if still empty and conda exists
if [[ -z "$ENV_NAME" ]] && command -v conda >/dev/null 2>&1; then
  ENV_NAME="$(
    { conda info --json 2>/dev/null || echo ''; } | python - <<'PY'
import sys, json
data = sys.stdin.read().strip()
if not data:
    print('')
else:
    try:
        info = json.loads(data)
        print(info.get('active_prefix_name','') or '')
    except Exception:
        print('')
PY
  )"
fi

# 3) final fallback
if [[ -z "$ENV_NAME" ]]; then
  ENV_NAME="exported"
fi

# --- write unified environment.yml
{
  echo "name: ${ENV_NAME}"
  echo "channels:"
  echo "  - conda-forge"
  if [[ "$ROBOSTACK_PRESENT" == "true" ]]; then
    echo "  - robostack-noetic"
  fi
  echo "dependencies:"
  # minimal conda deps
  if [[ -n "$PYVER" ]]; then
    echo "  - python=${PYVER}"
  else
    echo "  - python"     # fallback
  fi
  echo "  - pip"

  # add ROS meta if detected
  if [[ "$ROS_META_PRESENT" == "true" ]]; then
    echo "  - ros-noetic-desktop"
  fi

  # add other explicit conda specs from history (excluding pip & python & ros meta to avoid duplicates)
  if [[ -s "$CONDA_ENV_HISTORY" ]]; then
    # shell parse: print dependency lines; simple filter
    awk '
      $0 ~ /^dependencies:/ {deps=1; next}
      deps && $0 ~ /^ *-/ {
        gsub(/^- +/,"",$0);
        if ($0 ~ /^pip($|[ =<>~])/ || $0 ~ /^python($|[ =<>~])/) next;
        if ($0 ~ /^ros-noetic-desktop($|[ =<>~])/) next;
        print $0;
      }
    ' "$CONDA_ENV_HISTORY" | while read -r spec; do
        [[ -n "$spec" ]] && echo "  - ${spec}"
      done
  fi

  # pip block
  echo "  - pip:"
  if [[ -s "$PIP_BODY" ]]; then
    sed 's/^/    - /' "$PIP_BODY"
  fi

  # Optional: set PIP_EXTRA_INDEX_URL to prefer PyTorch CUDA wheels
  if [[ "$TORCH_INDEX" == "cu121" || "$TORCH_INDEX" == "cu124" || "$TORCH_INDEX" == "cu126" ]]; then
    echo "variables:"
    echo "  PIP_EXTRA_INDEX_URL: https://download.pytorch.org/whl/${TORCH_INDEX}"
  fi
} > "$OUT_YML"

echo "Wrote ${OUT_YML}"
echo
echo "Update on another machine with:"
echo "  mamba env update -n ${ENV_NAME} -f ${OUT_YML}    # or: conda env update -n ${ENV_NAME} -f ${OUT_YML}"
