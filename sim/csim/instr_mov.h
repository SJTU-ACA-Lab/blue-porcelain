#pragma once

#include "instruction.h"
#include "types.h"

namespace gpgpu {
class MovInstr : public Instruction {
 public:
  MovInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];
};

class SELInstr : public Instruction {
 public:
  SELInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];
};
}  // namespace gpgpu