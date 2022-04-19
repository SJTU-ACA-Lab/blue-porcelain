#include "operandTypeAssigner.h"

using namespace gpgpu;
DecodeInfo::DecodeInfo(const SassCodeType &sass) : sass_bin_(sass) {}
void DecodeInfo::basicDecode() {
  const uint64_t &L = sass_bin_.b64.L, &H = sass_bin_.b64.H;
  opcode = L & opcode_mask;
  opcode_type = L >> 9 & 0x7;

  predicate = L >> pred_shf & predicate_mask;
  p_neg = L >> 15 & 0x1;
  rd = L >> rd_shf & reg_mask;
  ra = L >> rs0_shf & reg_mask;
  local = H >> local_shf & local_mask;
  control_code = H >> ctrl_code_shf & control_code_mask;
}
void DecodeInfo::operandSbScAssign() {
  const uint64_t &L = sass_bin_.b64.L, &H = sass_bin_.b64.H;
  switch (opcode_type) {
    case 0x1:
      if (sb_type != INVALID) {
        sb = L >> rs1_shf & reg_mask;
        sb_type = REG32;
      }
      if (sc_type != INVALID) {
        sc = H & reg_mask;
        if (opcode == FADD) sc = L >> rs1_shf & reg_mask;
        sc_type = REG32;
      }
      break;
    case 0x2:
      if (sb_type != INVALID) {
        sb = H & reg_mask;
        sb_type = REG32;
      }
      if (sc_type != INVALID) {
        sc = L >> immed_shf & immed_mask;
        sc_type = IMM32;
      }
      break;
    case 0x3:
      if (sb_type != INVALID) {
        sb = H & reg_mask;
        sb_type = REG32;
      }
      if (sc_type != INVALID) {
        sc = L >> cm_shf & const_mem_addr_mask;
        sc_type = CMADDR;
      }
      break;
    case 0x4:
      if (sb_type != INVALID) {
        sb = L >> immed_shf & immed_mask;
        sb_type = IMM32;
      }
      if (sc_type != INVALID) {
        sc = H & reg_mask;
        sc_type = REG32;
      }
      break;
    case 0x5:
      if (sb_type != INVALID) {
        sb = L >> cm_shf & const_mem_addr_mask;
        sb_type = CMADDR;
      }
      if (sc_type != INVALID) {
        sc = H & reg_mask;
        sc_type = REG32;
      }
      break;
    default:
      break;
  }
}

void DecodeInfo::ldstSbScAssign() {
  const uint64_t &L = sass_bin_.b64.L, &H = sass_bin_.b64.H;
  uint32_t data_type = H >> dtype_shf & 0x7;
  switch (opcode) {
    case LDG:
    case LD:
    case LDS:
    case LDL:
      if (data_type == 5)
        rd_type = REG64;
      else if (data_type == 6)
        rd_type = REG128;
      sb = L >> 40 & 0xffffff;
      sb_type = IMM32;
      sc_type = INVALID;
      break;
    case ST:
    case STG:
    case STS:
    case STL:
      rd_type = INVALID;
      sb = L >> 32 & reg_mask;
      sb_type = REG32;
      if (data_type == 5)
        sb_type = REG64;
      else if (data_type == 6)
        sb_type = REG128;
      sc = L >> 40 & 0xffffff;
      sc_type = IMM32;
      break;
    case LDC:
      sb = L >> cm_shf & const_mem_addr_mask;
      sb_type = CMADDR;
      sc_type = INVALID;
      break;
    default:
      break;
  }
}