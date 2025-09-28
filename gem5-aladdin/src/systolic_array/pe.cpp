#include "systolic_array.h"
#include "pe.h"
#include <iostream>
#include <random>


namespace systolic {

void MulAccUnit::checkEndOfWindow() {
  // Check if this is the end of a convolutional window.
  const std::vector<int>& weightIndices = input1->indices;
  if (weightIndices[1] == accel.weightRows - 1 &&
      weightIndices[2] == accel.weightCols - 1 &&
      weightIndices[3] == accel.weightChans - 1) {
    output->windowEnd = true;
    output->bubble = false;
  }
}

// We can't directly do float16 operations, here we use a FP16 library for that.
template <>
void MulAccUnit::doMulAcc<float16>() {
  float16 input0Data = *(input0->getDataPtr<float16>());
  float16 input1Data = *(input1->getDataPtr<float16>());
  float16 input2Data = (input2->isWindowEnd() || input2->size() == 0)
                           ? 0
                           : *(input2->getDataPtr<float16>());
  output->resize(input0->size());
  float16* outputData = output->getDataPtr<float16>();
  *outputData = fp16(fp32(input0Data) * fp32(input1Data) + fp32(input2Data));
  //std::cout << "PE Output Before DMA Write: " << fp32(*outputData) << std::endl;
  //std::cout<<"Output data is::: "<<fp16(fp32(input0Data) * fp32(input1Data) + fp32(input2Data))<<"\n";

  std::vector<int>& inputIndices = input0->indices;
  std::vector<int>& weightIndices = input1->indices;
  DPRINTF(SystolicPE,
          "IReg (%d, %d, %d, %d): %f, WReg (%d, %d, %d, %d): %f, OReg: %f.\n",
          inputIndices[0], inputIndices[1], inputIndices[2], inputIndices[3],
          fp32(input0Data), weightIndices[0], weightIndices[1],
          weightIndices[2], weightIndices[3], fp32(input1Data),
          fp32(input2Data));
}

template <>
void MulAccUnit::doMulAcc_att<float16>(float randomization_prob) {

      // Create random engine and distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());  // Mersenne Twister random number generator
    static std::uniform_real_distribution<float> prob_dist(0.0, 1.0);  // Probability distribution
    static std::uniform_int_distribution<int> int_dist(1, 100);  // Example integer range
    static std::uniform_real_distribution<float> float_dist(0, 100000);  // Example float range

  float16 input0Data = *(input0->getDataPtr<float16>());
  float16 input1Data;
  float16 input2Data;

    // Apply randomization based on probability
    //std::cout<<"prob test::"<<prob_dist(gen)<<"\n";
    if (prob_dist(gen) < randomization_prob) {
        // if(std::is_integral<float16>::value) {
        //     input1Data = int_dist(gen);  // Generate random integer
        // } else if(std::is_floating_point<float16>::value) {
            input1Data =  static_cast<float>(float_dist(gen));  // Generate random float
        //    std::cout << "Random input 1 is: " << input1Data << "\n";
        //}
    }

    if (prob_dist(gen) < randomization_prob) {
        // if(std::is_integral<float16>::value) {
        //     input2Data = int_dist(gen);  // Generate random integer
        // } else if(std::is_floating_point<float16>::value) {
            input2Data =  static_cast<float>(float_dist(gen));   // Generate random float
        //    std::cout << "Random input 2 is: " << input2Data << "\n";
        //}
    }

  output->resize(input0->size());
  float16* outputData = output->getDataPtr<float16>();
  *outputData = fp16(fp32(input0Data) * fp32(input1Data) + fp32(input2Data));
  std::vector<int>& inputIndices = input0->indices;
  std::vector<int>& weightIndices = input1->indices;
  DPRINTF(SystolicPE,
          "IReg (%d, %d, %d, %d): %f, WReg (%d, %d, %d, %d): %f, OReg: %f.\n",
          inputIndices[0], inputIndices[1], inputIndices[2], inputIndices[3],
          fp32(input0Data), weightIndices[0], weightIndices[1],
          weightIndices[2], weightIndices[3], fp32(input1Data),
          fp32(input2Data));
}

void MulAccUnit::evaluate() {
  // Only perform the MACC operation if the input and weight registers do not
  // contain bubbles.
  if (!input0->isBubble() && !input1->isBubble()) {
    if (accel.dataType == Int32)
      doMulAcc<int>();
    else if (accel.dataType == Int64)
      doMulAcc<int64_t>();
    else if (accel.dataType == Float16)
      doMulAcc<float16>();
    else if (accel.dataType == Float32)
      doMulAcc<float>();
    else if (accel.dataType == Float64)
      doMulAcc<double>();
    checkEndOfWindow();
  }
}

  //for attack simulation only TODO
void MulAccUnit::evaluate_att() {
  // Only perform the MACC operation if the input and weight registers do not
  // contain bubbles.
  float ram_prob = 0.8;
  if (!input0->isBubble() && !input1->isBubble()) {
    if (accel.dataType == Int32)
      doMulAcc_att<int>(ram_prob);
    else if (accel.dataType == Int64)
      doMulAcc_att<int64_t>(ram_prob);
    else if (accel.dataType == Float16)
      doMulAcc_att<float16>(ram_prob);
    else if (accel.dataType == Float32)
      doMulAcc_att<float>(ram_prob);
    else if (accel.dataType == Float64)
      doMulAcc_att<double>(ram_prob);
    checkEndOfWindow();
  }
}



}  // namespace systolic
