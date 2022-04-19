#pragma once

#include "instruction.h"
#include "types.h"

namespace gpgpu {
class IMADInstr : public Instruction {
 public:
  IMADInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t X_ = 0;          // 1
  uint32_t data_type_ = 0;  // 3

  // sass_reg_t ra_data_[WARP_SIZE];
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t sc_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];

  // sass_reg_t ps0_data_[WARP_SIZE];
  // sass_reg_t pd0_data_[WARP_SIZE];
};

class IMADWIDEInstr : public Instruction {
 public:
  IMADWIDEInstr(Core& sm, const uint64_t& pc, int warp_id,
                const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t X_ = 0;          // 1
  uint32_t data_type_ = 0;  // 3

  // sass_reg_t ra_data_[WARP_SIZE];
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t sc_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];

  // sass_reg_t ps0_data_[WARP_SIZE];
  // sass_reg_t pd0_data_[WARP_SIZE];
};

class IADD3Instr : public Instruction {
 public:
  IADD3Instr(Core& sm, const uint64_t& pc, int warp_id,
             const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t X_ = 0;          // 1
  uint32_t data_type_ = 0;  // 3

  // sass_reg_t ra_data_[WARP_SIZE];
  // sass_reg_t sb_data_[WARP_SIZE];
  // sass_reg_t sc_data_[WARP_SIZE];
  // sass_reg_t rd_data_[WARP_SIZE];

  // sass_reg_t ps0_data_[WARP_SIZE];
  // // sass_reg_t ps1_data_[WARP_SIZE];
  // sass_reg_t pd0_data_[WARP_SIZE];
};

class ISETPInstr : public Instruction {
 public:
  ISETPInstr(Core& sm, const uint64_t& pc, int warp_id,
             const SassCodeType& sass);

  void decode();

  void execute();

 private:
  bool setpex_;
  bool data_type_;
  uint32_t boolop_;  // 3
  uint32_t cmpop_;   // 4
};

class SHFInstr : public Instruction {
 public:
  SHFInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  bool pos_;
  bool hi_;
  uint32_t data_type_;  // 4
};

class LEAInstr : public Instruction {
 public:
  LEAInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  bool hi_;
  uint32_t imm_;  // 5
  bool sx32_;
  bool X_;
};

class LOP3Instr : public Instruction {
 public:
  LOP3Instr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t immLUT_;  // 8
};

}  // namespace gpgpu