#pragma once

#include "instruction.h"
#include "types.h"

namespace gpgpu {
class EXITInstr : public Instruction {
 public:
  EXITInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();
};

class BRAInstr : public Instruction {
 public:
  BRAInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();
};

class BREAKInstr : public Instruction {
 public:
  BREAKInstr(Core& sm, const uint64_t& pc, int warp_id,
             const SassCodeType& sass);

  void decode();

  void execute();
};

class BSSYInstr : public Instruction {
 public:
  BSSYInstr(Core& sm, const uint64_t& pc, int warp_id,
            const SassCodeType& sass);

  void decode();

  void execute();
};

class BSYNCInstr : public Instruction {
 public:
  BSYNCInstr(Core& sm, const uint64_t& pc, int warp_id,
             const SassCodeType& sass);

  void decode();

  void execute();
};

class RETInstr : public Instruction {
 public:
  RETInstr(Core& sm, const uint64_t& pc, int warp_id, const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t ret_type_ = 0;  // ret方式
  uint32_t nodec_ = 0;     // 1
};

class CALABSInstr : public Instruction {
 public:
  CALABSInstr(Core& sm, const uint64_t& pc, int warp_id,
              const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t noinc = 0;  // 1
};

class CALRELInstr : public Instruction {
 public:
  CALRELInstr(Core& sm, const uint64_t& pc, int warp_id,
              const SassCodeType& sass);

  void decode();

  void execute();

 private:
  uint32_t noinc = 0;  // 1
};

}  // namespace gpgpu