# Superseded

These are the inputs and the plotting script of the first version of the
Tiling Attack figure. They are kept for reference only.

`data_extract.py` averages the "Avg Power" that gem5-Aladdin reports for every
accelerator invocation. The scratchpad part of that value is the scratchpad
energy accumulated since the start of the run divided by the cycles of the
current invocation, because the scratchpad statistics are not reset between
invocations, so the average grows with the number of invocations, which the
tiling attack multiplies. The VGG-16 attack summary here
(`nnet_fwd_summary_att_vgg_2003.txt`) cannot be corrected: its accumulated
scratchpad energy forms two interleaved series, and unlike its baseline it
contains no loop-sampled (upsampled) cycles.

The figure in the paper is produced by `../tiling_att_fig4_corrected/`, which
charges each invocation its own scratchpad energy and uses three runs per model
from one gem5 configuration.
