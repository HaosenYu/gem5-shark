#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <iostream>

#include "aladdin_sys_connection.h"
#include "aladdin_sys_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

aladdin_params_t* getParams(volatile int* finish_flag,
                            int finish_flag_val,
                            void* accel_params_ptr,
                            int size) {
    aladdin_params_t* params =
        (aladdin_params_t*)malloc(sizeof(aladdin_params_t));
    params->finish_flag = finish_flag;
    if (params->finish_flag == NULL)
        params->finish_flag = (volatile int*)malloc(sizeof(int));
    *(params->finish_flag) = finish_flag_val;
    params->accel_params_ptr = accel_params_ptr;
    params->size = size;
    return params;
}

void suspendCPUUntilFlagChanges(volatile int* finish_flag) {
    while (1) {
        ioctl(ALADDIN_FD, WAIT_FINISH_SIGNAL, finish_flag);
        if (*finish_flag != NOT_COMPLETED)
            break;
    }
}

// New function to count cycles when the finish flag is 0 and 1.
void waitForAcceleratorWithCounters(volatile int* finish_flag) {
    uint64_t flagZeroTicks = 0;  // Total ticks where finish_flag == 0
    uint64_t flagOneTicks = 0;   // Total ticks where finish_flag == 1
    uint64_t prevTick = curTick();  // Start tracking from the current tick
    int prevState = *finish_flag;  // Initial state of the finish flag

    while (true) {
        ioctl(ALADDIN_FD, WAIT_FINISH_SIGNAL, finish_flag);  // Suspend the CPU thread

        uint64_t currentTick = curTick();  // Get the current tick
        if (prevState == 0) {
            flagZeroTicks += (currentTick - prevTick);
        } else if (prevState == 1) {
            flagOneTicks += (currentTick - prevTick);
        }

        // Update for the next iteration
        prevTick = currentTick;
        prevState = *finish_flag;

        // Break if the finish flag is no longer "NOT_COMPLETED" (assumed `0`).
        if (*finish_flag != NOT_COMPLETED)
            break;
    }

    // Final update for the last state
    uint64_t finalTick = curTick();
    if (prevState == 0) {
        flagZeroTicks += (finalTick - prevTick);
    } else if (prevState == 1) {
        flagOneTicks += (finalTick - prevTick);
    }

    // Print the results
    std::cout << "Finish flag was 0 for " << flagZeroTicks << " ticks.\n";
    std::cout << "Finish flag was 1 for " << flagOneTicks << " ticks.\n";
}

void invokeAcceleratorAndBlock(unsigned req_code) {
    aladdin_params_t* params = getParams(NULL, NOT_COMPLETED, NULL, 0);
    ioctl(ALADDIN_FD, req_code, params);
    std::cout << "Finish flag now is: " << params->finish_flag << ".\n";
    waitForAcceleratorWithCounters(params->finish_flag);  // Use the updated function here
    free((void*)(params->finish_flag));
    free(params);
}

volatile int* invokeAcceleratorAndReturn(unsigned req_code) {
    aladdin_params_t* params = getParams(NULL, NOT_COMPLETED, NULL, 0);
    ioctl(ALADDIN_FD, req_code, params);
    std::cout << "Finish flag now is: " << params->finish_flag << ".\n";
    volatile int* finish_flag = params->finish_flag;
    free(params);
    return finish_flag;
}

void invokeAcceleratorAndReturn2(unsigned req_code, volatile int* finish_flag) {
    aladdin_params_t* params = getParams(finish_flag, NOT_COMPLETED, NULL, 0);
    ioctl(ALADDIN_FD, req_code, params);
    std::cout << "Finish flag now is: " << params->finish_flag << ".\n";
    free(params);
}

// Modified to use the new cycle-counting wait function
void waitForAccelerator(volatile int* finish_flag) {
    waitForAcceleratorWithCounters(finish_flag);
}

void dumpGem5Stats(const char* stats_desc) {
    ioctl(ALADDIN_FD, DUMP_STATS, stats_desc);
}

void resetTrace(unsigned req_code) {
    ioctl(ALADDIN_FD, RESET_TRACE, &req_code);
}

void resetGem5Stats() { ioctl(ALADDIN_FD, RESET_STATS, NULL); }

void mapArrayToAccelerator(unsigned req_code,
                           const char* array_name,
                           void* addr,
                           size_t size) {
    aladdin_map_t mapping;
    mapping.array_name = array_name;
    mapping.addr = addr;
    mapping.request_code = req_code;
    mapping.size = size;

    syscall(SYS_fcntl, ALADDIN_FD, ALADDIN_MAP_ARRAY, &mapping);
}

void setArrayMemoryType(unsigned req_code,
                        const char* array_name,
                        MemoryType mem_type) {
    aladdin_mem_desc_t desc;
    desc.array_name = array_name;
    desc.request_code = req_code;
    desc.mem_type = mem_type;

    syscall(SYS_fcntl, ALADDIN_FD, ALADDIN_MEM_DESC, &desc);
}

const char* memoryTypeToString(MemoryType mem_type) {
    switch (mem_type) {
        case spad:
            return "spad";
        case reg:
            return "reg";
        case dma:
            return "dma";
        case acp:
            return "acp";
        case cache:
            return "cache";
        default:
            return "invalid";
    }
}

#ifdef __cplusplus
}  // extern "C"
#endif
