#!/usr/bin/env bash

source ./model_files

bmk_dir=../../../../../build/bin

${bmk_dir}/smaug \
  ${topo_file} ${params_file} --sample-level=high --debug-level=0 --num-accels=1 -p

