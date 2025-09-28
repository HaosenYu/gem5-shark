#include "systolic_array.h"
#include "dataflow.h"

namespace systolic {

Dataflow::Dataflow(SystolicArray& _accel, const SystolicArrayParams& params)
    : Ticked(_accel, &(_accel.numCycles)), accel(_accel), state(Idle),
      peArray(params.peArrayRows * params.peArrayCols),
      inputFetchUnits(params.peArrayRows), weightFetchUnits(params.peArrayCols), 
      pe_destory_att(params.pe_destory_att), commitUnits(params.peArrayRows), weightFoldBarrier(0), doneCount(0) {
  // Create PEs.

  if (pe_destory_att) {
    peArrayRows_re = std::rand() % params.peArrayRows;
    peArrayCols_re = std::rand() % params.peArrayCols;

    std::cout << "The safe PEs are " << peArrayCols_re * peArrayRows_re << " due to dirtying PEs.\n";
  } else {
    peArrayRows_re = params.peArrayRows;
    peArrayCols_re = params.peArrayCols;

    std::cout << "No attack triggered, safe PEs: " << peArrayCols_re * peArrayRows_re << " \n";
  }

  for (int r = 0; r < params.peArrayRows; r++) {
        for (int c = 0; c < params.peArrayCols; c++) {
            if (r < peArrayRows_re && c < peArrayCols_re) {
 
            } else {
                // Outside the perfect region
                std::cout<<"The dirty PE id "<< r * accel.peArrayCols + c<<" \n";
            }
        }
    }


  for (int i = 0; i < peArray.size(); i++) {
    peArray[i] = new ProcElem(params.acceleratorName + ".pe" + std::to_string(i), accel);
  }

  // Form the pipeline by chaining the PEs for the active subset.
  for (int r = 0; r < params.peArrayRows; r++) {
    for (int c = 0; c < params.peArrayCols; c++) {
      // Connect the input register to the one in the next PE down the row.
      if (c < params.peArrayCols - 1) {
        peArray[peIndex(r, c)]->output0 =
            peArray[peIndex(r, c + 1)]->inputReg.input();
      }
      // Connect the weight register to the one in the next PE down the column.
      if (r < params.peArrayRows - 1) {
        peArray[peIndex(r, c)]->output1 =
            peArray[peIndex(r + 1, c)]->weightReg.input();
      }
    }
  }

  // Create input fetch units.
  for (int i = 0; i < inputFetchUnits.size(); i++) {
    inputFetchUnits[i] = new InputFetch(
        i, accel, params, peArray[peIndex(i, 0)]->inputReg.input());
  }

  // Create weight fetch units.
  for (int i = 0; i < weightFetchUnits.size(); i++) {
    weightFetchUnits[i] = new WeightFetch(
        i, accel, params, peArray[peIndex(0, i)]->weightReg.input());
  }

  // Create output commit units for the active rows.
  for (int i = 0; i < params.peArrayRows; i++) {
    commitUnits[i] = new Commit(i, accel, params);
    // Connect output registers of this PE row to the commit unit.
    for (int j = 0; j < params.peArrayCols; j++) {
      commitUnits[i]->inputs[j] = peArray[peIndex(i, j)]->outputReg.output();
    }
  }
}

int Dataflow::peIndex(int r, int c) const {
  assert(r < accel.peArrayRows && c < accel.peArrayCols && "Out of bounds of the PE array.");
  return r * accel.peArrayCols + c;
}

void Dataflow::scheduleStreamingEvents() {
  for (int i = 0; i < inputFetchUnits.size(); i++)
    accel.schedule(inputFetchUnits[i]->startStreamingEvent,
                   accel.clockEdge(Cycles(i + 1)));
  for (int i = 0; i < weightFetchUnits.size(); i++)
    accel.schedule(weightFetchUnits[i]->startStreamingEvent,
                   accel.clockEdge(Cycles(i + 1)));
}

void Dataflow::notifyDone() {
  if (++doneCount == commitUnits.size()) {
    DPRINTF(SystolicDataflow, "Done :)\n");
    state = Idle;
    accel.notifyDone();
  }
}

void Dataflow::evaluate() {
  // DPRINTF(SystolicDataflow, "%s\n", __func__);
  // Fetch unit operations.
// Loop over active input fetch units based on peArrayRows_re
// for (int i = 0; i < peArrayRows_re; i++) {
//      inputFetchUnits[i]->evaluate();
// }

// // Loop over active weight fetch units based on peArrayCols_re
// for (int i = 0; i < peArrayCols_re; i++) {
//        weightFetchUnits[i]->evaluate();
// }

// // Loop over active commit units based on peArrayRows_re
// for (int i = 0; i < peArrayRows_re; i++) {
//        commitUnits[i]->evaluate();
// }

  for (auto fetch : inputFetchUnits)
    fetch->evaluate();
  for (auto fetch : weightFetchUnits)
    fetch->evaluate();
  for (auto commit : commitUnits)
    commit->evaluate();


  if (state == Prefill) {
    bool prefillDone = true;
//     for (int i = 0; i < peArrayRows_re; i++) {
//         prefillDone &= inputFetchUnits[i]->isUnused() || inputFetchUnits[i]->filled();
//     }
// // Check active weight fetch units based on peArrayCols_re
//     for (int i = 0; i < peArrayCols_re; i++) {
//         prefillDone &= weightFetchUnits[i]->isUnused() || weightFetchUnits[i]->filled();
//     }

    for (const auto& fetch : inputFetchUnits)
      prefillDone &= fetch->isUnused() || fetch->filled();
    for (const auto& fetch : weightFetchUnits)
      prefillDone &= fetch->isUnused() || fetch->filled();


    if (prefillDone) {
     // DPRINTF(SystolicDataflow, "Prefilling done.\n");
      scheduleStreamingEvents();
      state = Compute;
    }
  } else if (state == Compute) {
    // DPRINTF(SystolicDataflow, "PE Computing start :: )\n");
    for (int r = 0; r < accel.peArrayRows; r++) {
        for (int c = 0; c < accel.peArrayCols; c++) {
            if (r < peArrayRows_re && c < peArrayCols_re) {
                // Inside the perfect region
                peArray[peIndex(r, c)]->evaluate();
            } else {
                // Outside the perfect region (dirty PEs)
                peArray[peIndex(r, c)]->evaluate_att();
            }
        }
    }


    // for (auto pe : peArray)
    //   pe->evaluate();

    // DPRINTF(SystolicDataflow, "PE Computing done :: )\n");
    // for (int r = 0; r < peArrayRows_re; r++) {
    //   for (int c = 0; c < peArrayCols_re; c++) {
    //     peArray[peIndex(r, c)]->inputReg.evaluate();
    //     peArray[peIndex(r, c)]->weightReg.evaluate();
    //     peArray[peIndex(r, c)]->outputReg.evaluate();
    //   }
    // }
    for (auto pe : peArray) {
      pe->inputReg.evaluate();
      pe->weightReg.evaluate();
      pe->outputReg.evaluate();
    }
  }
}

}  // namespace systolic
