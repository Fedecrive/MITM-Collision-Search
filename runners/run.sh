#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run.sh --user USERNAME --n N --np NP --threads T [--partitions P] [--verbose] [-- PROG_ARGS...]

Fetches challenge parameters from:
  https://ppar.tme-crypto.fr/USERNAME/N

Then runs the compiled MPI+OpenMP program with:
  --n N --C0 <hex> --C1 <hex>

Arguments:
  --user USERNAME   Challenge username (e.g., ciao)
  --n N             Key size / block size in bits (e.g., 34, 40)
  --np NP           MPI processes (mpirun -np)
  --threads T       OpenMP threads per MPI process (OMP_NUM_THREADS)
  --partitions P    Enable slicing with P partitions (power of two). Default: disabled
  --verbose         Enable verbose progress output (heartbeats, config)

Notes:
  - Extra args after "--" are passed to the program.
  - Program selection: uses $PROG if set; otherwise uses ./mitm.
  - If mpirun is not found and NP=1, runs the program directly.

Examples:
  ./run.sh --user ciao --n 34 --np 40 --threads 13
  ./run.sh --user ciao --n 40 --np 40 --threads 13 --partitions 16
  ./run.sh --user ciao --n 40 --np 40 --threads 13 --partitions 16 --verbose
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

USER=""
N=""
NP=""
THREADS=""
PARTITIONS=""
VERBOSE=0
PROG_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      USER="${2:-}"; shift 2 ;;
    --n)
      N="${2:-}"; shift 2 ;;
    --np)
      NP="${2:-}"; shift 2 ;;
    --threads)
      THREADS="${2:-}"; shift 2 ;;
    --partitions)
      PARTITIONS="${2:-}"; shift 2 ;;
    --verbose)
      VERBOSE=1; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      PROG_ARGS+=("$@");
      break
      ;;
    *)
      die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$USER" ]] || { usage; die "--user is required"; }
[[ -n "$N" ]] || { usage; die "--n is required"; }
[[ -n "$NP" ]] || { usage; die "--np is required"; }
[[ -n "$THREADS" ]] || { usage; die "--threads is required"; }

if ! [[ "$N" =~ ^[0-9]+$ ]]; then die "--n must be an integer"; fi
if ! [[ "$NP" =~ ^[0-9]+$ ]]; then die "--np must be an integer"; fi
if ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then die "--threads must be an integer"; fi
if [[ -n "$PARTITIONS" ]] && ! [[ "$PARTITIONS" =~ ^[0-9]+$ ]]; then
  die "--partitions must be an integer"
fi
if [[ -n "$PARTITIONS" ]]; then
  if [[ "$PARTITIONS" -le 1 ]]; then
    PARTITIONS=""  # treat 0/1 as disabled
  else
    # power-of-two check
    if (( (PARTITIONS & (PARTITIONS - 1)) != 0 )); then
      die "--partitions must be a power of two (got $PARTITIONS)"
    fi
  fi
fi

URL="https://ppar.tme-crypto.fr/${USER}/${N}"
echo "Fetching challenge from: $URL" >&2

if ! command -v curl >/dev/null 2>&1; then
  die "curl not found; install curl or fetch the page manually"
fi

CHALLENGE_TEXT=""
if ! CHALLENGE_TEXT="$(curl -fsSL "$URL")"; then
  # Some environments block HTTPS or have TLS issues; try HTTP as a fallback.
  URL_FALLBACK="http://ppar.tme-crypto.fr/${USER}/${N}"
  echo "Fetch failed over HTTPS; retrying: $URL_FALLBACK" >&2
  CHALLENGE_TEXT="$(curl -fsSL "$URL_FALLBACK")" || die "Failed to fetch challenge page (tried HTTPS and HTTP). Check username/n." 
fi

parse_pair() {
  local label="$1"
  local line
  line="$(printf '%s\n' "$CHALLENGE_TEXT" | grep -E "^${label}[[:space:]]*=[[:space:]]*\(" | head -n 1 || true)"
  [[ -n "$line" ]] || die "Could not find $label line in challenge page"

  # Expected format: C0 = (30197cb7, 8b964ec5)
  local a b
  a="$(printf '%s' "$line" | sed -nE 's/^.*\(([0-9a-fA-F]+),[[:space:]]*([0-9a-fA-F]+)\).*$/\1/p')"
  b="$(printf '%s' "$line" | sed -nE 's/^.*\(([0-9a-fA-F]+),[[:space:]]*([0-9a-fA-F]+)\).*$/\2/p')"
  [[ -n "$a" && -n "$b" ]] || die "Failed to parse $label from line: $line"

  # The site provides (low32, high32). The program expects high32||low32 (see example on the page).
  printf '%s%s' "${b,,}" "${a,,}"
}

C0_HEX="$(parse_pair "C0")"
C1_HEX="$(parse_pair "C1")"

echo "Parsed: --n $N --C0 $C0_HEX --C1 $C1_HEX" >&2

export OMP_NUM_THREADS="$THREADS"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

BASE_ARGS=(--n "$N" --C0 "$C0_HEX" --C1 "$C1_HEX")
if [[ -n "$PARTITIONS" ]]; then
  BASE_ARGS+=(--partitions "$PARTITIONS")
fi
if [[ "$VERBOSE" == "1" ]]; then
  BASE_ARGS+=(--verbose)
fi

PROG="${PROG:-}"
if [[ -z "$PROG" ]]; then
  if [[ -x "./mitm" ]]; then
    PROG="./mitm"
  else
    die "No executable found: ./mitm. Build it (e.g., make mitm) or set PROG=/path/to/program"
  fi
fi

if [[ ! -x "$PROG" ]]; then
  die "Program is not executable: $PROG"
fi

MPI_ARGS=()
if [[ -n "${OAR_NODEFILE:-}" ]]; then
  MPI_ARGS+=(--hostfile "$OAR_NODEFILE")
fi

# Keep the original mapping defaults, but allow them to be overridden by the user via mpirun wrapper if desired.
MPI_ARGS+=(--map-by "ppr:1:socket:pe=${OMP_NUM_THREADS}" --bind-to core)

if command -v mpirun >/dev/null 2>&1; then
  echo "Running: mpirun -np $NP ${MPI_ARGS[*]} $PROG ${BASE_ARGS[*]} ${PROG_ARGS[*]}" >&2
  exec mpirun "${MPI_ARGS[@]}" -np "$NP" "$PROG" "${BASE_ARGS[@]}" "${PROG_ARGS[@]}"
else
  if [[ "$NP" != "1" ]]; then
    die "mpirun not found, but --np is $NP. Install MPI or run with --np 1."
  fi
  echo "Running (no mpirun): $PROG ${BASE_ARGS[*]} ${PROG_ARGS[*]}" >&2
  exec "$PROG" "${BASE_ARGS[@]}" "${PROG_ARGS[@]}"
fi