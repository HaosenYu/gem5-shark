#!/usr/bin/env bash

. ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
bmk_dir=../../../../../build/bin

${gem5_dir}/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=1 \
  --mem-size=4GB \
  --mem-type=LPDDR4_3200_2x16  \
  --sys-clock=1.25GHz \
  --cpu-clock=2.5GHz \
  --cpu-type=TimingSimpleCPU \
  --ruby \
  --l2_size=2097152 \
  --l2_assoc=16 \
  --cacheline_size=32 \
  --access-backing-store \
  --accel_cfg_file=gem5.cfg \
  --fast-forward=10000000000 \
  -c attacked_smaug_haosen/smaug \
  -o "${topo_file} ${params_file} --sample-level=no --gem5 --debug-level=0 --num-accels=1 -p --enable-tiling-attack " \
  > stdout 2> stderr
