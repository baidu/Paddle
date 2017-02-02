/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#ifndef HL_CPU_GRU_CUH_
#define HL_CPU_GRU_CUH_

#ifndef __NVCC__

#include "paddle/math/MathFunctions.h"

#ifndef PADDLE_TYPE_DOUBLE
#define     CBLAS_GEMM     paddle::gemm<float>
#else
#define     CBLAS_GEMM     paddle::gemm<double>
#endif

/**
 * dot(h_{t-1}, r_t) = h_{t-1} * r_t
 * @param[in,out]   gateValue           (update gate(z_t), reset gate(r_t), h_t)
 * @param[out]      resetOutputValue    intermediate value for frame state
 * @param[in]       prevOutputValue     previous output (h_{t-1})
 * @param[in]       frameSize           size of frame at t
 * @param[in]       active_gate         forward gate function
 */
template<class OpResetOutput>
void hl_naive_gru_forward_reset_output(OpResetOutput opResetOutput,
                                       real *gateValue,
                                       real *resetOutputValue,
                                       real *prevOutputValue,
                                       int frameSize,
                                       hl_activation_mode_t active_gate) {
  real rValueUpdateGate;
  real rValueResetGate;
  real rValueResetOutput;
  real rPrevOut = 0;
  real *updateGate = gateValue;
  real *resetGate = gateValue + frameSize;

  for (int i = 0; i < frameSize; i++) {
    rValueUpdateGate = updateGate[i];
    rValueResetGate = resetGate[i];
    if (prevOutputValue) {
      rPrevOut = prevOutputValue[i];
    }

    opResetOutput(rValueUpdateGate,
                  rValueResetGate,
                  rPrevOut,
                  rValueResetOutput,
                  hppl::cpu::forward[active_gate]);

    updateGate[i] = rValueUpdateGate;
    resetGate[i] = rValueResetGate;
    resetOutputValue[i] = rValueResetOutput;
  }
}

/**
 * output hidden state value at time t
 * h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
 * \tilde{h}_t = f(\tilde{h}_t)
 * @param[in,out]   gateValue           (update gate(z_t), reset gate(r_t), h_t)
 * @param[in]       prevOutputValue     previous output (h_{t-1})
 * @param[in]       outputValue         output state value (h_t)
 * @param[in]       frameSize           size of frame at t
 * @param[in]       active_gate         forward gate function
 */
template<class OpFinalOutput>
void hl_naive_gru_forward_final_output(OpFinalOutput opFinalOutput,
                                       real *gateValue,
                                       real *prevOutputValue,
                                       real *outputValue,
                                       int frameSize,
                                       hl_activation_mode_t active_node) {
  real rValueUpdateGate;
  real rValueFrameState;
  real rPrevOut = 0;
  real rOutput;
  real *updateGate = gateValue;
  real *frameState = gateValue + frameSize * 2;

  for (int i = 0; i < frameSize; i++) {
    rValueUpdateGate = updateGate[i];
    rValueFrameState = frameState[i];
    if (prevOutputValue) {
      rPrevOut = prevOutputValue[i];
    }

    opFinalOutput(rValueUpdateGate,
                  rValueFrameState,
                  rPrevOut,
                  rOutput,
                  hppl::cpu::forward[active_node]);

    frameState[i] = rValueFrameState;
    outputValue[i] = rOutput;
  }
}

template<class OpResetOutput>
void hl_avx_gru_forward_reset_output(OpResetOutput opResetOutput,
                                     real *gateValue,
                                     real *resetOutputValue,
                                     real *prevOutputValue,
                                     int frameSize,
                                     hl_activation_mode_t active_gate) {
#ifdef __AVX__
  __m256 rValueUpdateGate;
  __m256 rValueResetGate;
  __m256 rValueResetOutput;
  __m256 rPrevOut = _mm256_set1_ps(0.0f);
  __m256 *updateGate = (__m256*)gateValue;
  __m256 *resetGate = (__m256*)(gateValue + frameSize);

  for (int i = 0; i < frameSize / 8; i++) {
    rValueUpdateGate = updateGate[i];
    rValueResetGate = resetGate[i];
    if (prevOutputValue) {
      rPrevOut = ((__m256*)prevOutputValue)[i];
    }

    opResetOutput(rValueUpdateGate,
                  rValueResetGate,
                  rPrevOut,
                  rValueResetOutput,
                  hppl::avx::forward[active_gate]);

    updateGate[i] = rValueUpdateGate;
    resetGate[i] = rValueResetGate;
    ((__m256*)resetOutputValue)[i] = rValueResetOutput;
  }
#endif
}

template<class OpFinalOutput>
void hl_avx_gru_forward_final_output(OpFinalOutput opFinalOutput,
                                     real *gateValue,
                                     real *prevOutputValue,
                                     real *outputValue,
                                     int frameSize,
                                     hl_activation_mode_t active_node) {
#ifdef __AVX__
  __m256 rValueUpdateGate;
  __m256 rValueFrameState;
  __m256 rPrevOut = _mm256_set1_ps(0.0f);
  __m256 rOutput;
  __m256 *updateGate = (__m256*)gateValue;
  __m256 *frameState = (__m256*)(gateValue + frameSize * 2);

  for (int i = 0; i < frameSize / 8; i++) {
    rValueUpdateGate = updateGate[i];
    rValueFrameState = frameState[i];
    if (prevOutputValue) {
      rPrevOut = ((__m256*)prevOutputValue)[i];
    }

    opFinalOutput(rValueUpdateGate,
                  rValueFrameState,
                  rPrevOut,
                  rOutput,
                  hppl::avx::forward[active_node]);

    frameState[i] = rValueFrameState;
    ((__m256*)outputValue)[i] = rOutput;
  }
#endif
}

/// dot(h_{t-1}, r_t) = h_{t-1} * r_t, r_t = f(r_t), z_t = f(z_t)
template<class OpResetOutput>
inline void forward_reset_output(OpResetOutput opResetOutput,
                                 hl_gru_value value,
                                 int frameSize,
                                 int batchSize,
                                 hl_activation_mode_t active_gate) {
  for (int b = 0; b < batchSize; b++) {
    if (OpResetOutput::avx && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
      hl_avx_gru_forward_reset_output(opResetOutput,
        value.gateValue, value.resetOutputValue, value.prevOutValue,
        frameSize, active_gate);
    } else {
      hl_naive_gru_forward_reset_output(opResetOutput,
        value.gateValue, value.resetOutputValue, value.prevOutValue,
        frameSize, active_gate);
    }

    value.gateValue += frameSize * 3;
    value.resetOutputValue += frameSize;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }
  }
}

template<class OpFinalOutput>
inline void forward_final_output(OpFinalOutput opFinalOutput,
                                 hl_gru_value value,
                                 int frameSize,
                                 int batchSize,
                                 hl_activation_mode_t active_node) {
  for (int b = 0; b < batchSize; b++) {
    if (OpFinalOutput::avx && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
      hl_avx_gru_forward_final_output(opFinalOutput,
        value.gateValue, value.prevOutValue, value.outputValue,
        frameSize, active_node);
    } else {
      hl_naive_gru_forward_final_output(opFinalOutput,
        value.gateValue, value.prevOutValue, value.outputValue,
        frameSize, active_node);
    }

    value.gateValue += frameSize * 3;
    value.outputValue += frameSize;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }
  }
}

template<class OpResetOutput, class OpFinalOutput>
void hl_cpu_gru_forward(OpResetOutput opResetOutput,
                        OpFinalOutput opFinalOutput,
                        hl_gru_value value,
                        int frameSize,
                        int batchSize,
                        hl_activation_mode_t active_node,
                        hl_activation_mode_t active_gate) {
  /// z_t = x_{z_t} + h_{t-1} * Uz, r_t = x_{r_t} + h_{t-1} * Ur
  if (value.prevOutValue) {
    CBLAS_GEMM(CblasNoTrans,
               CblasNoTrans,
               batchSize,
               2 * frameSize,
               frameSize,
               1,
               value.prevOutValue,
               frameSize,
               value.gateWeight,
               frameSize * 2,
               1,
               value.gateValue,
               frameSize * 3);
  }

  /// dot(h_{t-1}, r_t) = h_{t-1} * r_t, r_t = f(r_t), z_t = f(z_t)
  forward_reset_output(opResetOutput, value, frameSize, batchSize, active_gate);

  /// \tilde{h}_t = x_{i_t} + U * dot(h_{t-1}, r_t)
  if (value.prevOutValue) {
    CBLAS_GEMM(CblasNoTrans,
               CblasNoTrans,
               batchSize,
               frameSize,
               frameSize,
               1,
               value.resetOutputValue,
               frameSize,
               value.stateWeight,
               frameSize,
               1,
               value.gateValue + frameSize * 2,
               frameSize * 3);
  }

  /// h_t = (1 - z_t) * h_{t-1} + z_t * f(\tilde{h}_t)
  forward_final_output(opFinalOutput, value, frameSize, batchSize, active_node);
}

/**
 * dLdz_t = (\tilde{h}_t - h_{t-1}) * dLdh_t
 * dLdh_{t-1} += (1 - z_t) * dLdh_t
 * dLd\tilde{h}_t = dLdh_t * z_t * f'({h}_t)
 *
 * @param[in]       gateValue           (update gate(z_t), reset gate(r_t), h_t)
 * @param[out]      gateGrad            (dLd{z_t}, dLd{r_t}, dLd{h}_t)
 * @param[in]       prevOutValue        previous output (h_{t-1})
 * @param[out]      prevOutGrad         dLdh_{t-1}
 * @param[in]       outputGrad          dLdh_t
 * @param[in]       frameSize           size of frame at t
 * @param[in]       active_gate         backward gate function
 */
template<class OpStateGrad>
void hl_naive_gru_backward_state_grad(OpStateGrad opStateGrad,
                                      real *gateValue,
                                      real *gateGrad,
                                      real *prevOutValue,
                                      real *prevOutGrad,
                                      real *outputGrad,
                                      int frameSize,
                                      hl_activation_mode_t active_node) {
  real rUpdateGateValue;
  real rUpdateGateGrad;
  real rFrameStateValue;
  real rFrameStateGrad;
  real rOutGrad;
  real rPrevOutValue = 0;
  real rPrevOutGrad  = 0;
  real *updateGateValue = gateValue;
  real *updateGateGrad = gateGrad;
  real *frameStateValue = gateValue + frameSize * 2;
  real *frameStateGrad = gateGrad + frameSize * 2;

  for (int i = 0; i < frameSize; i++) {
    rUpdateGateValue = updateGateValue[i];
    rFrameStateValue = frameStateValue[i];
    rOutGrad  = outputGrad[i];
    if (prevOutValue) {
      rPrevOutValue = prevOutValue[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad  = prevOutGrad[i];
    }

    opStateGrad(rUpdateGateValue,
                rUpdateGateGrad,
                rFrameStateValue,
                rFrameStateGrad,
                rPrevOutValue,
                rPrevOutGrad,
                rOutGrad,
                hppl::cpu::backward[active_node]);

    updateGateGrad[i] = rUpdateGateGrad;
    frameStateGrad[i] = rFrameStateGrad;
    if (prevOutGrad) {
      prevOutGrad[i] = rPrevOutGrad;
    }
  }
}

/**
 * dLd{r}_t = dLd\tilde{h}_t * h_{t-1} * f'(r_t)
 * dLdz_t = dLdz_t * f'(z_t)
 * dLdh_{t-1} += dLd\tilde{h}_t * r_t
 *
 * @param[in]       gateValue           (update gate(z_t), reset gate(r_t), h_t)
 * @param[out]      gateGrad            (dLd{z_t}, dLd{r_t}, dLd{h}_t)
 * @param[in]       prevOutValue        previous output (h_{t-1})
 * @param[out]      prevOutGrad         dLdh_{t-1}
 * @param[in]       resetOutputGrad     dLd{dot(r_t, h_{t-1})}
 * @param[in]       frameSize           size of frame at t
 * @param[in]       active_gate         backward gate function
 */
template<class OpResetGrad>
void hl_naive_gru_backward_reset_grad(OpResetGrad opResetGrad,
                                      real *gateValue,
                                      real *gateGrad,
                                      real *prevOutValue,
                                      real *prevOutGrad,
                                      real *resetOutputGrad,
                                      int frameSize,
                                      hl_activation_mode_t active_gate) {
  real rUpdateGateValue;
  real rUpdateGateGrad;
  real rResetGateValue;
  real rResetGateGrad;
  real rResetOutputGrad = 0;
  real rPrevOutValue = 0;
  real rPrevOutGrad  = 0;
  real *updateGateValue = gateValue;
  real *updateGateGrad = gateGrad;
  real *resetGateValue = gateValue + frameSize;
  real *resetGateGrad = gateGrad + frameSize;

  for (int i = 0; i < frameSize; i++) {
    rUpdateGateValue = updateGateValue[i];
    rUpdateGateGrad = updateGateGrad[i];
    rResetGateValue = resetGateValue[i];

    if (prevOutValue && prevOutGrad) {
      rResetOutputGrad = resetOutputGrad[i];
    }
    if (prevOutValue) {
      rPrevOutValue = prevOutValue[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad  = prevOutGrad[i];
    }

    opResetGrad(rUpdateGateValue,
                rUpdateGateGrad,
                rResetGateValue,
                rResetGateGrad,
                rPrevOutValue,
                rPrevOutGrad,
                rResetOutputGrad,
                hppl::cpu::backward[active_gate]);

    updateGateGrad[i] = rUpdateGateGrad;
    resetGateGrad[i] = rResetGateGrad;
    if (prevOutGrad) {
      prevOutGrad[i] = rPrevOutGrad;
    }
  }
}

template<class OpStateGrad>
void hl_avx_gru_backward_state_grad(OpStateGrad opStateGrad,
                                    real *gateValue,
                                    real *gateGrad,
                                    real *prevOutValue,
                                    real *prevOutGrad,
                                    real *outputGrad,
                                    int frameSize,
                                    hl_activation_mode_t active_node) {
#ifdef __AVX__
  __m256 rUpdateGateValue;
  __m256 rUpdateGateGrad;
  __m256 rFrameStateValue;
  __m256 rFrameStateGrad;
  __m256 rOutGrad;
  __m256 rPrevOutValue = _mm256_set1_ps(0.0f);
  __m256 rPrevOutGrad  = _mm256_set1_ps(0.0f);
  __m256 *updateGateValue = (__m256*)gateValue;
  __m256 *updateGateGrad = (__m256*)gateGrad;
  __m256 *frameStateValue = (__m256*)(gateValue + frameSize * 2);
  __m256 *frameStateGrad = (__m256*)(gateGrad + frameSize * 2);

  for (int i = 0; i < frameSize / 8; i++) {
    rUpdateGateValue = updateGateValue[i];
    rFrameStateValue = frameStateValue[i];
    rOutGrad  = ((__m256*)outputGrad)[i];
    if (prevOutValue) {
      rPrevOutValue = ((__m256*)prevOutValue)[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad  = ((__m256*)prevOutGrad)[i];
    }

    opStateGrad(rUpdateGateValue,
                rUpdateGateGrad,
                rFrameStateValue,
                rFrameStateGrad,
                rPrevOutValue,
                rPrevOutGrad,
                rOutGrad,
                hppl::avx::backward[active_node]);

    updateGateGrad[i] = rUpdateGateGrad;
    frameStateGrad[i] = rFrameStateGrad;
    if (prevOutGrad) {
      ((__m256*)prevOutGrad)[i] = rPrevOutGrad;
    }
  }
#endif
}

template<class OpResetGrad>
void hl_avx_gru_backward_reset_grad(OpResetGrad opResetGrad,
                                    real *gateValue,
                                    real *gateGrad,
                                    real *prevOutValue,
                                    real *prevOutGrad,
                                    real *resetOutputGrad,
                                    int frameSize,
                                    hl_activation_mode_t active_gate) {
#ifdef __AVX__
  __m256 rUpdateGateValue;
  __m256 rUpdateGateGrad;
  __m256 rResetGateValue;
  __m256 rResetGateGrad;
  __m256 rResetOutputGrad = _mm256_set1_ps(0.0f);
  __m256 rPrevOutValue = _mm256_set1_ps(0.0f);
  __m256 rPrevOutGrad  = _mm256_set1_ps(0.0f);
  __m256 *updateGateValue = (__m256*)gateValue;
  __m256 *updateGateGrad = (__m256*)gateGrad;
  __m256 *resetGateValue = (__m256*)(gateValue + frameSize);
  __m256 *resetGateGrad = (__m256*)(gateGrad + frameSize);

  for (int i = 0; i < frameSize / 8; i++) {
    rUpdateGateValue = updateGateValue[i];
    rUpdateGateGrad = updateGateGrad[i];
    rResetGateValue = resetGateValue[i];

    if (prevOutValue && prevOutGrad) {
      rResetOutputGrad = ((__m256*)resetOutputGrad)[i];
    }
    if (prevOutValue) {
      rPrevOutValue = ((__m256*)prevOutValue)[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad  = ((__m256*)prevOutGrad)[i];
    }

    opResetGrad(rUpdateGateValue,
                rUpdateGateGrad,
                rResetGateValue,
                rResetGateGrad,
                rPrevOutValue,
                rPrevOutGrad,
                rResetOutputGrad,
                hppl::avx::backward[active_gate]);

    updateGateGrad[i] = rUpdateGateGrad;
    resetGateGrad[i] = rResetGateGrad;
    if (prevOutGrad) {
      ((__m256*)prevOutGrad)[i] = rPrevOutGrad;
    }
  }
#endif
}

template<class OpStateGrad>
inline void backward_state_grad(OpStateGrad opStateGrad,
                                hl_gru_value value,
                                hl_gru_grad  grad,
                                int frameSize,
                                int batchSize,
                                hl_activation_mode_t active_node) {
  for (int b = 0; b < batchSize; b++) {
    if (OpStateGrad::avx && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
      hl_avx_gru_backward_state_grad(opStateGrad,
        value.gateValue, grad.gateGrad, value.prevOutValue, grad.prevOutGrad,
        grad.outputGrad, frameSize, active_node);
    } else {
      hl_naive_gru_backward_state_grad(opStateGrad,
        value.gateValue, grad.gateGrad, value.prevOutValue, grad.prevOutGrad,
        grad.outputGrad, frameSize, active_node);
    }

    value.gateValue += frameSize * 3;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }

    grad.gateGrad += frameSize * 3;
    grad.outputGrad += frameSize;
    if (grad.prevOutGrad) {
      grad.prevOutGrad += frameSize;
    }
  }
}

template<class OpResetGrad>
inline void backward_reset_grad(OpResetGrad opResetGrad,
                                hl_gru_value value,
                                hl_gru_grad  grad,
                                int frameSize,
                                int batchSize,
                                hl_activation_mode_t active_gate) {
  for (int b = 0; b < batchSize; b++) {
    if (OpResetGrad::avx && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
      hl_avx_gru_backward_reset_grad(opResetGrad,
        value.gateValue, grad.gateGrad, value.prevOutValue, grad.prevOutGrad,
        grad.resetOutputGrad, frameSize, active_gate);
    } else {
      hl_naive_gru_backward_reset_grad(opResetGrad,
        value.gateValue, grad.gateGrad, value.prevOutValue, grad.prevOutGrad,
        grad.resetOutputGrad, frameSize, active_gate);
    }

    value.gateValue += frameSize * 3;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }

    grad.gateGrad += frameSize * 3;
    grad.resetOutputGrad += frameSize;
    if (grad.prevOutGrad) {
      grad.prevOutGrad += frameSize;
    }
  }
}

/**
 * GRU backward
 * 1. prevOutGrad : dLdh_{t-1} = dLdz_t * (U_z)' + dLdr_t * (U_r)'
 *                 + dLd{dot(r_t, h_{t-1})} * r_t + (1 - z_t) * dLdh_t
 * 2. stateWeightGrad : dLd\tilde{h}_t = dLdh_t * z_t * f'(\tilde{h}_t)
 * 3. resetOutputGrad dLd{dot(r_t,h_{t-1})} = U' * dLd\tilde{h}_t
 * 4. resetGateGrad dLd{r}_t = dLd\tilde{h}_t * h_{t-1} * f'(r_t)
 * 5. updateGateGrad dLdz_t = (\tilde{h}_t - h_{t-1}) * dLdh_t * f'(z_t)
 * 6. updateGateWeightGrad : dLdU_z = (h_{t-1})' * dLdz_t
 * 7. resetGateWeightGrad : dLdU_r = (h_{t-1})' * dLdr_t
 * 8. stateWeightGrad : dLdU = (dLd{dot})' * dLd\tilde{h}_t
 *
 * some parameter notation:
 * gateValue        (update gate(z_t), reset gate(r_t), \tidle{h}_t)
 * gateGrad         (dLd{z_t}, dLd{r_t}, dLd\tilde{h}_t)
 * gateWeightGrad   (dLdU_z, dLdU_r)
 */
template<class OpStateGrad, class OpResetGrad>
void hl_cpu_gru_backward(OpStateGrad opStateGrad,
                         OpResetGrad opResetGrad,
                         hl_gru_value value,
                         hl_gru_grad  grad,
                         int frameSize,
                         int batchSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate) {
  /// calculate dLdz_t, dLd\tilde{h}_t, dLdh_{t-1} (partial sum)
  backward_state_grad(opStateGrad, value, grad,
    frameSize, batchSize, active_node);

  /// dLd{dot(r_t,h_{t-1})} = U' * dLd\tilde{h}_t
  if (value.prevOutValue && grad.prevOutGrad) {
    CBLAS_GEMM(CblasNoTrans,
               CblasTrans,
               batchSize,
               frameSize,
               frameSize,
               1,
               grad.gateGrad + frameSize * 2,
               frameSize * 3,
               value.stateWeight,
               frameSize,
               0,
               grad.resetOutputGrad,
               frameSize);

    /// dLdU = (dLd{dot})' * dLd\tilde{h}_t
    if (grad.stateWeightGrad) {
      CBLAS_GEMM(CblasTrans,
                 CblasNoTrans,
                 frameSize,
                 frameSize,
                 batchSize,
                 1,
                 value.resetOutputValue,
                 frameSize,
                 grad.gateGrad + frameSize * 2,
                 frameSize * 3,
                 1,
                 grad.stateWeightGrad,
                 frameSize);
    }
  }

  /// calculate dLdr_t, complete dLdz_t, increment dLdh_{t-1}
  backward_reset_grad(opResetGrad, value, grad,
    frameSize, batchSize, active_gate);

  /// complete dLdh_{t-1} += dLdz_t * (U_z)' + dLdr_t * (U_r)'
  if (grad.prevOutGrad && value.prevOutValue) {
    CBLAS_GEMM(CblasNoTrans,
               CblasTrans,
               batchSize,
               frameSize,
               frameSize * 2,
               1,
               grad.gateGrad,
               frameSize * 3,
               value.gateWeight,
               frameSize * 2,
               1,
               grad.prevOutGrad,
               frameSize);

    /// [dLdU_z, dLdU_r] = (h_{t-1})' * [dLdz_t, dLdr_t]
    if (grad.gateWeightGrad) {
      CBLAS_GEMM(CblasTrans,
                 CblasNoTrans,
                 frameSize,
                 frameSize * 2,
                 batchSize,
                 1,
                 value.prevOutValue,
                 frameSize,
                 grad.gateGrad,
                 frameSize * 3,
                 1,
                 grad.gateWeightGrad,
                 frameSize * 2);
    }
  }
}

#endif

#endif  // HL_CPU_GRU_CUH_
