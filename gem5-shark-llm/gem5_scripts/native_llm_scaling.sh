#!/bin/bash
# Native (no gem5) SMV runs of one decode layer at the dimensions of several LLMs
# (512 cached tokens): accelerator invocations under no attack, TAtt and
# TAtt + D-TAtt, and a check of every output against the NumPy reference.
# Run inside the SMAUG container.
# Usage: bash native_llm_scaling.sh [WORK_DIR]
set -uo pipefail
WS=${WS:-/workspace}
SM=$(realpath "$(dirname "$0")/../smaug_models")
WORK=$(realpath -m "${1:-llm_scale}")
TATT="--enable-tiling-attack --tiling-attack-gemm"
#        name          d    heads kv dh  ffn
while read -r NAME DM H KV DH FFN; do
  M=$WORK/$NAME
  python "$SM/llm_decode_smaug.py" --d $DM --heads $H --kv-heads $KV --dh $DH --ffn $FFN \
      --L 512 --out "$M" | tail -1
  for c in base att def; do
    case $c in base) F="";; att) F="$TATT";; def) F="$TATT --enable-tiling-defense";; esac
    mkdir -p "$M/$c" && cp "$M/ref_output.npy" "$M/$c/" && cd "$M/$c" || exit 1
    "$WS/smaug/build/bin/smaug" "$M/dec_topo.pbtxt" "$M/dec_params.pb" \
        --print-last-output=proto --debug-level=1 $F > run.log 2>&1
    echo "$NAME $c rc=$? invocations=$(grep -c '^Input' run.log)" \
         "gemm_tiles=$(grep -c '^Input: ' run.log)" \
         "$(python "$SM/check_output.py" "$M/$c" | tail -2 | tr '\n' ' ')"
    rm -f output.pb
  done
  rm -f "$M/dec_params.pb"
done <<'LIST'
qwen2.5-0.5b   896  14  2  64  4864
qwen2.5-1.5b   1536 12  2  128 8960
qwen2.5-3b     2048 16  2  128 11008
llama3.2-1b    2048 32  8  64  8192
qwen2.5-7b     3584 28  4  128 18944
llama3.1-8b    4096 32  8  128 14336
LIST
