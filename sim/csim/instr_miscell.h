#pragma once

#include "instruction.h"
#include "types.h"

namespace gpgpu {
class SHFLInstr : public Instruction {
 public:
  SHFLInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t shfl_type = 0;  // 2
  // sass_reg_t ra_data_[WARP_SIZE];
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t sc_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];
};

class S2RInstr : public Instruction {
 public:
  S2RInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t sreg_ = 0;  // 9
  // sass_reg_t rd_data_[WARP_SIZE];
};

class BARInstr : public Instruction {
 public:
  BARInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
};
}  // namespace gpgpu