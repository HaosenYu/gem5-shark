#!/bin/bash
# gem5 evaluation of one LLM decode step (smaug_models/llm_decode_smaug.py) on the
# SMV accelerator, in Smaug's default system: one out-of-order x86 core and a
# crossbar. Run inside the SMAUG container after rebuilding smaug and gem5-aladdin.
#
#   COND = base | tatt | tatt_def | ffs | ffs_def
#     tatt      tiling attack on the GEMM tiler (--enable-tiling-attack --tiling-attack-gemm)
#     tatt_def  the same with the tiling range check (--enable-tiling-defense)
#     ffs       false finish-flag scheduling (flag_delay_att in gem5.cfg)
#     ffs_def   the same with the finish-flag watchdog (flag_ffs_defense in gem5.cfg)
#
# Usage: bash gem5_llm.sh COND MODEL_DIR OUT_DIR
#   MODEL_DIR holds dec_topo.pbtxt and dec_params.pb written by llm_decode_smaug.py.
# Smaug's highest sampling level is used: Aladdin simulates two iterations of the
# inner-product reduction loop per invocation and extrapolates the rest.
set -uo pipefail
COND=$1; MD=$(realpath "$2"); D=$(realpath -m "$3")
WS=${WS:-/workspace}
GEM5_DIR=${GEM5_DIR:-$WS/gem5-aladdin}
CFG=$WS/smaug/experiments/sims/smv/tests/mnist-cnn10      # gem5.cfg and smv-accel.cfg templates
# TOPO_OVERRIDE: an equivalent path (e.g. with "/./") only shifts the simulated
# stack; it steps around a rare timing-dependent gem5 O3/x86 decoder crash.
TOPO=${TOPO_OVERRIDE:-$MD/dec_topo.pbtxt}; PARAMS=$MD/dec_params.pb
TATT="--enable-tiling-attack --tiling-attack-gemm"
SMP="--sample-level=very_high --sample-num=1"
case $COND in
  base)     F="";                              FD=False; FW=False;;
  tatt)     F="$TATT";                         FD=False; FW=False;;
  tatt_def) F="$TATT --enable-tiling-defense"; FD=False; FW=False;;
  ffs)      F="";                              FD=True;  FW=False;;
  ffs_def)  F="";                              FD=True;  FW=True;;
  *) echo "unknown condition $COND"; exit 1;;
esac

mkdir -p "$D/outputs" && cd "$D" || exit 1
# The trace is deterministic, so a complete one is reused.
if [ ! -f trace.done ]; then
  "$WS/smaug/build/bin/smaug-instrumented" $TOPO $PARAMS $SMP --debug-level=0 \
      --num-accels=1 $F > trace_gen.log 2>&1 && touch trace.done
  echo "[$COND] trace rc=$?"
fi
sed "s/^flag_delay_att=.*/flag_delay_att=$FD\nflag_ffs_defense=$FW/" "$CFG/gem5.cfg" > gem5.cfg
cp "$CFG/smv-accel.cfg" .
# Loading the LLM layer's parameters takes about 1e10 instructions, beyond gem5's
# default fast-forward limit, hence the larger --fast-forward.
"$GEM5_DIR/build/X86/gem5.opt" --outdir="$D/outputs" --stats-db-file=stats.db \
    "$GEM5_DIR/configs/aladdin/aladdin_se.py" \
    --num-cpus=1 --mem-size=4GB --mem-type=LPDDR4_3200_2x16 --sys-clock=1.25GHz \
    --cpu-clock=2.5GHz --cpu-type=DerivO3CPU --ruby --network=garnet2.0 \
    --access-backing-store --l2_size=2097152 --l2_assoc=16 --cacheline_size=32 \
    --accel_cfg_file="$D/gem5.cfg" --fast-forward=200000000000 \
    -c "$WS/smaug/build/bin/smaug" \
    -o "$TOPO $PARAMS $SMP --gem5 --debug-level=0 --num-accels=1 $F" \
    > gem5_stdout.txt 2> gem5_stderr.txt
echo "[$COND] gem5 rc=$?"
