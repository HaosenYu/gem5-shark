#!/bin/bash
# NoC-level threat vectors and TAtt on one LLM decode step
# (smaug_models/llm_decode_smaug.py), on the system of the paper's DoS and FLAtt
# experiments (tests/mnist-lenet5/run.sh): four in-order x86 cores
# (TimingSimpleCPU) on a 2x2 Garnet mesh with XY routing. Run inside the SMAUG
# container after rebuilding smaug and gem5-aladdin.
#
#   COND = mesh_base | mesh_dos | mesh_flatt_dir | mesh_flatt_dma | mesh_flatt_cpu
#          | mesh_tatt | mesh_tatt_def
# The FLAtt victim is named by its network-interface id, which is the
# controller's Ruby NodeID (L1Cache 0-5, L2Cache 6, Directory 7, DMA 8-9, ACP
# 10-11), and by the router it attaches to (ext_links in config.ini):
#   7 = memory directory (router 1), 8 = accelerator DMA (router 3),
#   0 = CPU0 L1 cache (router 0).
#
# Usage: bash gem5_llm_noc.sh COND MODEL_DIR OUT_DIR
set -uo pipefail
COND=$1; MD=$(realpath "$2"); D=$(realpath -m "$3")
WS=${WS:-/workspace}
GEM5_DIR=${GEM5_DIR:-$WS/gem5-aladdin}
CFG=$WS/smaug/experiments/sims/smv/tests/mnist-cnn10      # gem5.cfg and smv-accel.cfg templates
TOPO=${TOPO_OVERRIDE:-$MD/dec_topo.pbtxt}; PARAMS=$MD/dec_params.pb
SMP="--sample-level=very_high --sample-num=1"
TATT="--enable-tiling-attack --tiling-attack-gemm"
F=""   # NoC attacks act inside gem5, so they use the base trace
case $COND in
  mesh_base)      N="";;
  mesh_dos)       N="--ni-dos-att --dos-num-att=30";;
  mesh_flatt_dir) N="--ni-lost-att=99 --lost-ni-att=7 --lost-router-att=1";;
  mesh_flatt_dma) N="--ni-lost-att=99 --lost-ni-att=8 --lost-router-att=3";;
  mesh_flatt_cpu) N="--ni-lost-att=99 --lost-ni-att=0 --lost-router-att=0";;
  mesh_tatt)      N=""; F="$TATT";;
  mesh_tatt_def)  N=""; F="$TATT --enable-tiling-defense";;
  *) echo "unknown condition $COND"; exit 1;;
esac

mkdir -p "$D/outputs" && cd "$D" || exit 1
if [ ! -f trace.done ]; then
  "$WS/smaug/build/bin/smaug-instrumented" $TOPO $PARAMS $SMP --debug-level=0 \
      --num-accels=1 $F > trace_gen.log 2>&1 && touch trace.done
  echo "[$COND] trace rc=$?"
fi
cp "$CFG/gem5.cfg" "$CFG/smv-accel.cfg" .
"$GEM5_DIR/build/X86/gem5.opt" --outdir="$D/outputs" --stats-db-file=stats.db \
    "$GEM5_DIR/configs/aladdin/aladdin_se.py" \
    --num-cpus=4 --mem-size=4GB --mem-type=LPDDR4_3200_2x16 --sys-clock=1.25GHz \
    --cpu-clock=2.5GHz --cpu-type=TimingSimpleCPU --ruby --network=garnet2.0 \
    --topology=Mesh_XY --mesh-rows=2 --routing-algorithm=1 $N \
    --l2_size=2097152 --l2_assoc=16 --cacheline_size=32 --access-backing-store \
    --accel_cfg_file="$D/gem5.cfg" --fast-forward=200000000000 \
    -c "$WS/smaug/build/bin/smaug" \
    -o "$TOPO $PARAMS $SMP --gem5 --debug-level=0 --num-accels=1 $F" \
    > gem5_stdout.txt 2> gem5_stderr.txt
echo "[$COND] gem5 rc=$?"
