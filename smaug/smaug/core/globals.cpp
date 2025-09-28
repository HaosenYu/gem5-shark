#include "smaug/core/globals.h"

namespace smaug {
bool runningInSimulation;
bool fastForwardMode = true;
int numAcceleratorsAvailable;
ThreadPool* threadPool = nullptr;
bool useSystolicArrayWhenAvailable;
//TODO attack simulation only
bool enable_swap_att;
}  // namespace smaug
