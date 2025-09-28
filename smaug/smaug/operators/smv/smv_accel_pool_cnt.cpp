#include <string>
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_accel_pool.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {

SmvAcceleratorPool::SmvAcceleratorPool(int _size)
        : size(_size), finishFlags(_size), finishFlagZeroCycles(0), finishFlagOneCycles(0) {}

void SmvAcceleratorPool::addFinishFlag(
        int accelIdx, std::unique_ptr<volatile int> finishFlag) {
    if (runningInSimulation) {
        finishFlags[accelIdx].push_back(std::move(finishFlag));
    }
}

void SmvAcceleratorPool::join(int accelIdx) {
    if (finishFlags[accelIdx].empty())
        return;

    while (!finishFlags[accelIdx].empty()) {
        std::unique_ptr<volatile int> finishFlag =
                std::move(finishFlags[accelIdx].front());
        waitForAcceleratorWithCycleCount(finishFlag.get()); // Updated to use the cycle-counting version.
        finishFlags[accelIdx].pop_front();
    }
    dout(1) << "Accelerator " << accelIdx << " finished.\n";
}

void SmvAcceleratorPool::joinAll() {
    std::cout<< "Waiting for all accelerators to finish.\n";
    for (int i = 0; i < size; i++)
        join(i);
    std::cout << "All accelerators finished.\n";

    // Print the results for the cycle counts.
    std::cout << "FinishFlag=0 cycles: " << finishFlagZeroCycles << "\n";
    std::cout << "FinishFlag=1 cycles: " << finishFlagOneCycles << "\n";
}

int SmvAcceleratorPool::getNextAvailableAccelerator(int currAccelIdx) {
    // Round-robin policy.
    int pickedAccel = currAccelIdx + 1;
    if (pickedAccel == size)
        pickedAccel = 0;
    // If the picked accelerator has not finished, wait until it returns.
    join(pickedAccel);
    if (size > 1)
        dout(1) << "Switched to accelerator " << pickedAccel << ".\n";
    return pickedAccel;
}

// New method to track cycles when waiting for the accelerator to finish.
void SmvAcceleratorPool::waitForAcceleratorWithCycleCount(volatile int* finishFlag) {
    while (*finishFlag == 0) {
        finishFlagZeroCycles++;
        // std::cout << "FinishFlag=0; Incrementing zero cycles.\n";
    }
    finishFlagOneCycles++;
    // std::cout << "FinishFlag=1; Incrementing one cycles.\n";
}

}  // namespace smaug
