#!/usr/bin/env bash

. ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
bmk_dir=`git rev-parse --show-toplevel`/../build/bin

${gem5_dir}/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=4 \
  --mem-size=4GB \
  --mem-type=LPDDR4_3200_2x16  \
  --sys-clock=1.25GHz \
  --cpu-clock=2.5GHz \
  --cpu-type=TimingSimpleCPU \
  --ruby \
  --network=garnet2.0 \
  --topology=Mesh_XY \
  --mesh-rows=2 \
  --routing-algorithm=1 \
  --access-backing-store \
  --l2_size=2097152 \
  --l2_assoc=16 \
  --cacheline_size=32 \
  --accel_cfg_file=gem5.cfg \
  --fast-forward=10000000000 \
  -c ${bmk_dir}/smaug \
  -o "${topo_file} ${params_file} --sample-level=no --gem5 --debug-level=0 --num-accels=1 -p" \
  > stdout 2> stderr
