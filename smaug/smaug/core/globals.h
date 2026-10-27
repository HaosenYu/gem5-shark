/**
 * \file globals.h
 * \brief SMAUG Global variables.
 */

#ifndef _CORE_GLOBALS_H_
#define _CORE_GLOBALS_H_

namespace smaug {

class ThreadPool;

/**
 * This is true if the user chooses to run the network in gem5 simulation.
 */
extern bool runningInSimulation;

/** True if we are simulating in fast-forward mode. */
extern bool fastForwardMode;

/**
 * The maximum number of accelerators an operator's work can be split across.
 * This limit exists to keep Aladdin simulation time and resources in check.
 */
constexpr const int maxNumAccelerators = 8;

/**
 * The actual number of accelerator complexes currently in use.
 */
extern int numAcceleratorsAvailable;

/**
 * The user-space thread pool used by SMAUG to run multithreaded tasks.
 */
extern ThreadPool* threadPool;

/**
 * If true, uses the systolic array for applicable operators when backend
 * support exists.
 */
extern bool useSystolicArrayWhenAvailable;

extern bool enable_swap_att;

extern bool enable_tiling_att;

/**
 * Scope of the tiling attack: when set, the attack also shrinks the tile
 * budget of the SMV inner-product (GEMM) tiler, not only the convolution
 * tiler. Off by default so earlier convolution-only results stay reproducible.
 */
extern bool tiling_att_gemm;

/** Defense: tiling-parameter range check that mitigates the tiling attack. */
extern bool enable_tiling_defense;


}  // namespace smaug

#endif
