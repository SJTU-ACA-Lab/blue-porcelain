#pragma once
#include "decodemask.h"
#include "types.h"
namespace gpgpu {

struct DecodeInfo {
  uint32_t opcode;            // 9
  uint32_t opcode_type;       // 3
  uint32_t local = 0;         // 20
  uint32_t control_code = 0;  // 24

  uint32_t predicate = 7;  // 3
  uint32_t p_neg = 0;
  uint32_t rd = 255;                   // 8
  oprand_data_type_t rd_type = REG32;  // 1
  uint32_t ra = 255;                   // 8
  oprand_data_type_t ra_type = REG32;  // 1
  int32_t sb = 255;                    // 8
  oprand_data_type_t sb_type = REG32;  // 2
  uint32_t cm_addr = 0;                // 20
  int32_t sc = 255;                    // 8
  oprand_data_type_t sc_type = REG32;
  uint32_t ra_neg = 0;   // 1 decode缺少
  uint32_t sb_neg = 0;   // 1
  uint32_t sc_neg = 0;   // 1
  uint32_t ps0 = 7;      // 3
  uint32_t ps1 = 7;      // 3
  uint32_t ps2 = 7;      // 3
  uint32_t ps2_neg = 0;  // 1
  uint32_t ps0_neg = 0;  // 1
  uint32_t ps1_neg = 0;  // 1
  uint32_t pd0 = 7;      // 3
  uint32_t pd1 = 7;      // 3
  DecodeInfo(const SassCodeType& sass);
  void basicDecode();
  //一般指令（除ldst）的sb sc处理
  void operandSbScAssign();
  // ldst指令的sb sc处理
  void ldstSbScAssign();

 private:
  SassCodeType sass_bin_;
};

};  // namespace gpgpu
