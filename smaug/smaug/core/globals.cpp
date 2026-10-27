#include "smaug/core/globals.h"

namespace smaug {
bool runningInSimulation;
bool fastForwardMode = true;
int numAcceleratorsAvailable;
ThreadPool* threadPool = nullptr;
bool useSystolicArrayWhenAvailable;
//TODO attack simulation only
bool enable_swap_att;
bool enable_tiling_att;
// Extends the tiling attack to the inner-product (GEMM) tiler.
bool tiling_att_gemm;
// Defense knob (mitigation of the tiling attack).
bool enable_tiling_defense;
}  // namespace smaug
