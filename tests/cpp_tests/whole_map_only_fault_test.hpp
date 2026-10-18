// Copyright (c) 2026 Falcata contributors. MIT license; see LICENSE.
// Compile-time-only diagnostic contract. Normal builds include/define NONE of this interface.
#ifndef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST_CONTRACT_HPP_
#define FALCATA_WHOLE_MAP_ONLY_FAULT_TEST_CONTRACT_HPP_
#ifndef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
#error "Diagnostic fault harness requires a SEPARATE Release test build with the explicit compile definition"
#endif
#if !defined(USE_CUDA) || CUDART_VERSION < 12040
#error "Diagnostic whole-DL test requires supported CUDA>=12.4 headers"
#endif
#ifdef DEBUG
#error "Use Release plus the dedicated test macro; DEBUG disables the canonical cache/map eligibility"
#endif
namespace Falcata {
class CUDASingleGPUTreeLearner;
enum class WholeMapOnlyFaultPoint {
  BeforePublish, AfterPublish, BeforeRecoveryRepair, AfterRecoveryRepair,
  RecoveryCleared, BeforeRootReuse, AfterRootInit, AfterTrainFlagsReset
};
using WholeMapOnlyFaultCallback = void (*)(CUDASingleGPUTreeLearner*, WholeMapOnlyFaultPoint);
// Definitions live in the diagnostic library's learner.cpp, NOT an executable-local TLS copy.
void SetWholeMapOnlyFaultCallback(WholeMapOnlyFaultCallback callback);
void WholeMapOnlyFaultCheckpoint(CUDASingleGPUTreeLearner* learner, WholeMapOnlyFaultPoint point);
}  // namespace Falcata
#endif
