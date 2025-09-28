#!/usr/bin/env bash

source ./model_files

bmk_dir=../../../../../build/bin

attacked_smaug_haosen/smaug-instrumented \
   ${topo_file} ${params_file} --sample-level=no --gem5 --debug-level=0 --num-accels=1 -p --enable-tiling-attack

