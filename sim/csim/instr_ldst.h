#pragma once

#include "instruction.h"
#include "types.h"
#include "util.h"

namespace gpgpu {
class LDGInstr : public Instruction {
 public:
  LDGInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t E_ = 0;          // 1
  uint32_t data_type_ = 0;  // 3
  uint32_t scope_ = 0;      // 2
  uint32_t strong_ = 0;     // 2
  uint32_t cache_ = 0;      // 2

  // sass_reg_t ra_data_[WARP_SIZE];
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];
};

class STGInstr : public Instruction {
 public:
  STGInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t E_ = 0;          // 1
  uint32_t data_type_ = 0;  // 3
  uint32_t scope_ = 0;      // 2
  uint32_t strong_ = 0;     // 2
  uint32_t cache_ = 0;      // 2

  // sass_reg_t ra_data_[WARP_SIZE];
  // sass_reg_t sb_data_[WARP_SIZE];
};

class LDSInstr : public Instruction {
 public:
  LDSInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t data_type_ = 0;  // 3
  uint32_t U = 0;           // 1
};

class STSInstr : public Instruction {
 public:
  STSInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t data_type_ = 0;  // 3
  uint32_t U = 0;           // 1
};

class LDCInstr : public Instruction {
 public:
  LDCInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t data_type_ = 0;  // 3
};
}  // namespace gpgpu