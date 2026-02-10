#!/bin/bash

# 1. Verifica input
if [ -z "$1" ]; then
    echo "Errore: Devi specificare il file contenente i comandi."
    echo "Uso: ./daje.sh <file_comandi.txt>"
    exit 1
fi

COMMAND_FILE="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND_FILE_DIR="$(cd "$(dirname "$COMMAND_FILE")" && pwd)"
DEFAULT_WORKDIR="${WORKDIR:-$PWD}"

# 2. Genera Timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="logs_$TIMESTAMP"

mkdir -p "$OUTPUT_DIR"

# CSV riassuntivo (una riga per comando)
CSV_FILE="$OUTPUT_DIR/results.csv"
echo "n,np,threads,partitions,fill_time_s,probe_time_s,solutions" > "$CSV_FILE"

echo "--------------------------------------------------"
echo " Avvio esecuzione da: $COMMAND_FILE"
echo " Cartella Output: $OUTPUT_DIR"
echo "--------------------------------------------------"
echo ""

count=1

find_upwards_for_file() {
    # Cerca un file risalendo le directory (dir, dir/.., dir/../.., ...)
    local start_dir="$1"
    local filename="$2"
    local d="$start_dir"

    while true; do
        if [ -f "$d/$filename" ]; then
            printf '%s' "$d"
            return 0
        fi
        if [ "$d" = "/" ] || [ -z "$d" ]; then
            return 1
        fi
        d="$(cd "$d/.." && pwd)"
    done
}

detect_workdir_for_cmd() {
    # Heuristica: se il comando invoca ./run.sh, trova la directory dove run.sh esiste.
    local cmd_str="$1"

    if [[ "$cmd_str" != *"./run.sh"* ]]; then
        printf '%s' "$DEFAULT_WORKDIR"
        return 0
    fi

    local candidates=("$DEFAULT_WORKDIR" "$COMMAND_FILE_DIR" "$SCRIPT_DIR")
    local c

    for c in "${candidates[@]}"; do
        if [ -f "$c/run.sh" ]; then
            printf '%s' "$c"
            return 0
        fi
        if found="$(find_upwards_for_file "$c" "run.sh" 2>/dev/null)"; then
            if [ -n "$found" ]; then
                printf '%s' "$found"
                return 0
            fi
        fi
    done

    # Fallback: esegui comunque nella DEFAULT_WORKDIR
    printf '%s' "$DEFAULT_WORKDIR"
}

extract_flag_value() {
    # Estrae il valore di una flag GNU-style da una stringa comando.
    # Supporta: --flag VALUE  oppure  --flag=VALUE
    local cmd_str="$1"
    local flag="$2"
    local val=""

    # --flag=VALUE
    val=$(printf '%s\n' "$cmd_str" | sed -nE "s/.*(^|[[:space:]])${flag}=([^[:space:]]+).*/\2/p" | head -n 1)
    if [ -n "$val" ]; then
        printf '%s' "$val"
        return 0
    fi

    # --flag VALUE
    val=$(printf '%s\n' "$cmd_str" | sed -nE "s/.*(^|[[:space:]])${flag}[[:space:]]+([^[:space:]]+).*/\2/p" | head -n 1)
    printf '%s' "$val"
}

parse_recap_field() {
    # Estrae un campo numerico dalla riga: Run summary: MPI ranks=2, partitions=1, ...
    local log_file="$1"
    local key="$2"   # es: "MPI ranks" oppure "partitions"
    awk -v k="$key" '
        $0 ~ /Run summary:/ {
            if (match($0, k "=[0-9]+")) {
                s=substr($0, RSTART, RLENGTH)
                sub("^" k "=", "", s)
                print s
                exit
            }
        }
    ' "$log_file"
}

parse_timing_max_seconds() {
    # Estrae il tempo (max) dalla riga tipo:
    # fill : 46.385s / 46.391s / 46.397s
    local log_file="$1"
    local label="$2"  # "fill" o "probe"
    awk -v lbl="$label" '
        $0 ~ "^[[:space:]]*" lbl "[[:space:]]*:[[:space:]]*" {
            line=$0
            sub("^[[:space:]]*" lbl "[[:space:]]*:[[:space:]]*", "", line)
            # Ora line dovrebbe essere: min s / avg s / max s
            n=split(line, a, "/")
            if (n >= 3) {
                t=a[3]
                gsub(/[[:space:]]/, "", t)
                sub(/s$/, "", t)
                print t
                exit
            }
        }
    ' "$log_file"
}

parse_solutions_list() {
    # Raccoglie le righe sotto "Solutions (k1, k2) and origin:" del recap
    # e le unisce con "; ".
    local log_file="$1"
    awk '
        BEGIN { in_block=0; out="" }
        /^Solutions \(k1, k2\) and origin:/ { in_block=1; next }
        in_block && /^=+/ { exit }
        in_block && /^[[:space:]]*-[[:space:]]/ {
            line=$0
            sub(/^[[:space:]]*-[[:space:]]*/, "", line)
            gsub(/"/, "\"\"", line)  # escape per CSV
            if (out != "") out = out "; " line
            else out = line
        }
        END { print out }
    ' "$log_file"
}

# 3. Loop
while IFS= read -r cmd || [ -n "$cmd" ]; do
    
    # Salta commenti e righe vuote
    if [[ -z "$cmd" ]] || [[ "$cmd" =~ ^# ]]; then
        continue
    fi

    filename=$(printf "%02d_output.log" "$count")
    filepath="$OUTPUT_DIR/$filename"

    echo "========================================" | tee "$filepath"
    echo "[$count] Eseguo: $cmd" | tee -a "$filepath"
    echo "========================================" | tee -a "$filepath"

    workdir="$(detect_workdir_for_cmd "$cmd")"
    echo "[runner] workdir: $workdir" | tee -a "$filepath"

    # --- MODIFICA IMPORTANTE QUI SOTTO ---
    # Aggiunto < /dev/null per evitare che il comando "rubi" le righe successive
    (
        cd "$workdir" || exit 127
        eval "$cmd"
    ) < /dev/null 2>&1 | tee -a "$filepath"
    cmd_exit=${PIPESTATUS[0]}
    echo "[runner] exit_code: $cmd_exit" | tee -a "$filepath"

    echo "" | tee -a "$filepath"

    # --- Estrazione risultati per CSV ---
    n_val=$(extract_flag_value "$cmd" "--n")
    np_val=$(extract_flag_value "$cmd" "--np")
    threads_val=$(extract_flag_value "$cmd" "--threads")

    partitions_val=$(extract_flag_value "$cmd" "--partitions")
    if [ -z "$partitions_val" ]; then
        partitions_val=$(parse_recap_field "$filepath" "partitions")
    fi

    # Tempi: usiamo il valore MAX riportato nel recap (min/avg/max across ranks)
    fill_time=$(parse_timing_max_seconds "$filepath" "fill")
    probe_time=$(parse_timing_max_seconds "$filepath" "probe")
    solutions=$(parse_solutions_list "$filepath")

    printf '%s,%s,%s,%s,%s,%s,"%s"\n' \
        "$n_val" "$np_val" "$threads_val" "$partitions_val" \
        "$fill_time" "$probe_time" "$solutions" >> "$CSV_FILE"
    
    ((count++))

done < "$COMMAND_FILE"

# Link simbolico all'ultima esecuzione (opzionale ma comodo)
rm -f latest_logs
ln -s "$OUTPUT_DIR" latest_logs

echo "--- Finito. Log salvati in: $OUTPUT_DIR ---"