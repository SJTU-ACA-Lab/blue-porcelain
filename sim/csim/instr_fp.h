#pragma once

#include "instruction.h"
#include "types.h"

namespace gpgpu {
class FADDInstr : public Instruction {
 public:
  FADDInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t ftz_ = 0;
  uint32_t rnd_ = 0;

  // sass_reg_t ra_data_[WARP_SIZE];
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t sc_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];

  // sass_reg_t ps0_data_[WARP_SIZE];
  // sass_reg_t pd0_data_[WARP_SIZE];
};

class MUFUInstr : public Instruction {
 public:
  MUFUInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t mufu_func_;

  // sass_reg_t ra_data_[WARP_SIZE];
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t sc_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];

  // sass_reg_t ps0_data_[WARP_SIZE];
  // // sass_reg_t ps1_data_[WARP_SIZE];
  // sass_reg_t pd0_data_[WARP_SIZE];
};

class FFMAInstr : public Instruction {
 public:
  FFMAInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();

 private:
  bool ftz_ = 0;
  uint32_t rnd_ = 0;
};

class FMULInstr : public Instruction {
 public:
  FMULInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();

 private:
  bool ftz_ = 0;
  uint32_t rnd_ = 0;
};

class FCHKInstr : public Instruction {
 public:
  FCHKInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();

 private:
};

class FSETPInstr : public Instruction {
 public:
  FSETPInstr(Core& sm, const uint64_t& pc, int warp_id,
             const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t boolop_;  // 3
  uint32_t cmpop_;   // 4
  bool ftz_ = 0;
  bool ra_abs_ = 0;
  bool sb_abs_ = 0;
};

}  // namespace gpgpu