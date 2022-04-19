#pragma once

#include "instruction.h"
#include "types.h"

namespace gpgpu {
class PLOP3Instr : public Instruction {
 public:
  PLOP3Instr(Core& sm, const uint64_t& pc, int warp_id,
             const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t immLUT;  // 5
};
}  // namespace gpgpu