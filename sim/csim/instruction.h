#pragma once

#include <stdint.h>
#include "types.h"
#include "decodemask.h"
#include <vector>
#include "operandTypeAssigner.h"

namespace gpgpu {
class Operand;
class PredOperand;
class Core;
class Instruction {
  friend class DebugDisplay;

 public:
  Instruction(Core& sm, const uint64_t& pc, int warp_id,
              const SassCodeType& sass);
  virtual ~Instruction();
  virtual void decode() = 0;
  void readOperand();
  virtual void execute() = 0;
  void writeBack();

 private:
  //读@p设置thread_active_mask_
  void readPrevPredicate();

 protected:
  uint64_t pc_;
  Core& sm_;
  ThreadMask thread_active_mask_;
  SassCodeType sass_bin_;
  DecodeInfo decode_info_;
  uint32_t warp_id_;

  //为了方便基类统一处理read和write
  std::vector<Operand*> src_operands_;
  std::vector<Operand*> dst_operands_;

  std::vector<PredOperand*> src_preds_;
  std::vector<PredOperand*> dst_preds_;
};

}  // namespace gpgpu