#include "instr_fp.h"

#include <fenv.h>

#include <cmath>

#include "operand.h"
using namespace gpgpu;

FADDInstr::FADDInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void FADDInstr::decode() {
  // basic decode：指令中一些通用字段的解析
  Instruction::decode();

  // FADD中特有的一些字段解码
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.sb_type = INVALID;
  decode_info_.operandSbScAssign();
  decode_info_.ra_neg = H >> rs0neg_shf & 0x1;
  decode_info_.sc_neg = L >> 63 & 0x1;
  ftz_ = H >> ftz_shf & 0x1;
  rnd_ = H >> rnd_shf & 0x3;
  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sc = new Operand(warp_id_, decode_info_.sc, thread_active_mask_,
                            decode_info_.sc_type);

  src_operands_.push_back(ra);
  src_operands_.push_back(sc);
}

void FADDInstr::execute() {
  //   sass_reg_t ra_data, sb_data, data;
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sc_data_ = src_operands_[1]->getData();
  sass_reg_t rd_data_[WARP_SIZE];

  for (int tid = 0; tid < WARP_SIZE; tid++) {
    if (thread_active_mask_.test(tid)) {
      sass_reg_t ra_data, sc_data, data;

      // int overflow = 0;
      int carry = 0;

      // 先拓宽长度（同一符号），再符号转换
      ra_data.f32 = ra_data_[tid].f32;
      sc_data.f32 = sc_data_[tid].f32;
      if (ftz_) {
        if (!std::isnormal(ra_data.f32) && !std::isnan(ra_data.f32) &&
            !std::isinf(ra_data.f32))
          ra_data.f32 = 0.0f;
        if (!std::isnormal(sc_data.f32) && !std::isnan(sc_data.f32) &&
            !std::isinf(sc_data.f32))
          sc_data.f32 = 0.0f;
      }
      if (decode_info_.ra_neg) ra_data.f32 = -ra_data.f32;
      if (decode_info_.sc_neg && decode_info_.sc_type != IMM32)
        sc_data.f32 = -sc_data.f32;

      unsigned rounding_mode = rnd_;
      int orig_rm = fegetround();
      // \.rni	TC; return RNI_OPTION;
      // \.rzi	TC; return RZI_OPTION;
      // \.rmi	TC; return RMI_OPTION;
      // \.rpi	TC; return RPI_OPTION;

      // \.rn	TC; return RN_OPTION;
      // \.rz	TC; return RZ_OPTION;
      // \.rm	TC; return RM_OPTION;
      // \.rp	TC; return RP_OPTION;
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }

      data.f32 = ra_data.f32 + sc_data.f32;
      if (ftz_) {
        if (!std::isnormal(data.f32) && !std::isnan(data.f32) &&
            !std::isinf(data.f32))
          data.f32 = 0.0f;
      }
      fesetround(orig_rm);
      rd_data_[tid].f32 = data.f32;
    }
  }

  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

/// MUFU ///
MUFUInstr::MUFUInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void MUFUInstr::decode() {
  // basic decode：指令中一些通用字段的解析
  Instruction::decode();

  // MUFU中特有的一些字段解码
  const uint64_t& H = sass_bin_.b64.H;

  decode_info_.ra_type = INVALID;
  decode_info_.rd_type = REG32;
  decode_info_.operandSbScAssign();
  decode_info_.sc_type = INVALID;
  mufu_func_ = (H >> 10) & 0xf;

  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  src_operands_.push_back(sb);
}

void MUFUInstr::execute() {
  sass_reg_t* sb_data_ = src_operands_[0]->getData();
  sass_reg_t rd_data_[WARP_SIZE];

  for (int tid = 0; tid < WARP_SIZE; tid++) {
    if (thread_active_mask_.test(tid)) {
      sass_reg_t sb_data = sb_data_[tid], data;

      switch (mufu_func_) {
        case 0x0:
          data.f32 = cos(sb_data.f32);
          break;
        case 0x1:
          data.f32 = sin(sb_data.f32);
          break;
        case 0x2:
          data.f32 = pow(2, sb_data.f32);
          break;
        case 0x3:
          data.f32 = log2(sb_data.f32);
          break;
        case 0x4:
          data.f32 = 1 / sb_data.f32;
          break;
        case 0x5:
          data.f32 = sqrt(1 / sb_data.f32);
          break;
        case 0x6:
          data.u64 = sb_data.u64 << 32;
          data.f64 = 1 / data.f64;
          data.u64 = data.u64 >> 32;
          break;
        case 0x7:
          data.u64 = sb_data.u64 << 32;
          data.f64 = sqrt(1 / data.f64);
          data.u64 = data.u64 >> 32;
          break;
        case 0x8:
          data.f32 = sqrt(sb_data.f32);
          break;
        default:
          break;
      }
      rd_data_[tid].f32 = data.f32;
    }
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

FFMAInstr::FFMAInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void FFMAInstr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  ftz_ = H >> ftz_shf & 0x1;
  rnd_ = H >> rnd_shf & 0x3;
  decode_info_.operandSbScAssign();
  decode_info_.ra_neg = H >> rs0neg_shf & 0x1;
  switch (decode_info_.opcode_type) {
    case 0x1:
    case 0x4:
    case 0x5:
      decode_info_.sb_neg = L >> rs1neg_shf & 0x1;
      decode_info_.sc_neg = H >> rs2neg_shf & 0x1;
      break;
    default:
      decode_info_.sb_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sc_neg = L >> rs1neg_shf & 0x1;
      break;
  }
  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  Operand* sc = new Operand(warp_id_, decode_info_.sc, thread_active_mask_,
                            decode_info_.sc_type);

  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
  src_operands_.push_back(sc);
}

void FFMAInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* sc_data_ = src_operands_[2]->getData();
  sass_reg_t rd_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid],
               sc_data = sc_data_[tid];
    if (ftz_) {
      if (!std::isnormal(ra_data.f32) && !std::isnan(ra_data.f32) &&
          !std::isinf(ra_data.f32))
        ra_data.f32 = 0.0f;
      if (!std::isnormal(sb_data.f32) && !std::isnan(sb_data.f32) &&
          !std::isinf(sb_data.f32))
        sb_data.f32 = 0.0f;
      if (!std::isnormal(sc_data.f32) && !std::isnan(sc_data.f32) &&
          !std::isinf(sc_data.f32))
        sc_data.f32 = 0.0f;
    }
    if (decode_info_.ra_neg) ra_data.f32 = -ra_data.f32;
    if (decode_info_.sb_neg && decode_info_.sb_type != IMM32)
      sb_data.f32 = -sb_data.f32;
    if (decode_info_.sc_neg && decode_info_.sc_type != IMM32)
      sc_data.f32 = -sc_data.f32;
    int orig_rm = fegetround();
    unsigned rounding_mode = rnd_;
    switch (rounding_mode) {
      case RN_OPTION:
        // fesetround(FE_TONEAREST);
        break;
      case RM_OPTION:
        fesetround(FE_DOWNWARD);
        break;
      case RP_OPTION:
        fesetround(FE_UPWARD);
        break;
      case RZ_OPTION:
        fesetround(FE_TOWARDZERO);
        break;
      default:
        assert(0);
        break;
    }
    float temp = ra_data.f32 * sb_data.f32 + sc_data.f32;
    fesetround(orig_rm);
    if (ftz_) {
      if (!std::isnormal(temp) && !std::isnan(temp) && !std::isinf(temp))
        temp = 0.0f;
    }
    rd_data_[tid].f32 = temp;
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

FMULInstr::FMULInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void FMULInstr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  ftz_ = H >> ftz_shf & 0x1;
  rnd_ = H >> rnd_shf & 0x3;
  decode_info_.sc_type = INVALID;
  decode_info_.operandSbScAssign();
  decode_info_.ra_neg = H >> rs0neg_shf & 0x1;
  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
}

void FMULInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t rd_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid];
    if (ftz_) {
      if (!std::isnormal(ra_data.f32) && !std::isnan(ra_data.f32) &&
          !std::isinf(ra_data.f32))
        ra_data.f32 = 0.0f;
      if (!std::isnormal(sb_data.f32) && !std::isnan(sb_data.f32) &&
          !std::isinf(sb_data.f32))
        sb_data.f32 = 0.0f;
    }
    if (decode_info_.ra_neg) ra_data.f32 = -ra_data.f32;
    if (decode_info_.sb_neg && decode_info_.sb_type != 0x1)
      sb_data.f32 = -sb_data.f32;
    int orig_rm = fegetround();
    unsigned rounding_mode = rnd_;
    switch (rounding_mode) {
      case RN_OPTION:
        // fesetround(FE_TONEAREST);
        break;
      case RM_OPTION:
        fesetround(FE_DOWNWARD);
        break;
      case RP_OPTION:
        fesetround(FE_UPWARD);
        break;
      case RZ_OPTION:
        fesetround(FE_TOWARDZERO);
        break;
      default:
        assert(0);
        break;
    }
    float temp = ra_data.f32 * sb_data.f32;
    fesetround(orig_rm);
    if (ftz_) {
      if (!std::isnormal(temp) && !std::isnan(temp) && !std::isinf(temp))
        temp = 0.0f;
    }
    rd_data_[tid].f32 = temp;
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

FCHKInstr::FCHKInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void FCHKInstr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.rd_type = INVALID;
  decode_info_.sc_type = INVALID;
  decode_info_.operandSbScAssign();
  decode_info_.ra_neg = H >> rs0neg_shf & 0x1;
  decode_info_.sb_neg = L >> rs1neg_shf & 0x1;
  decode_info_.pd0 = H >> pd0_shf & 0x7;
  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
}

void FCHKInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t pd0_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid];
    int exp1, exp2;
    int emin = -126, emax = 127, n = 24;
    if (decode_info_.ra_neg) ra_data.f32 = -ra_data.f32;
    if (decode_info_.sb_neg && decode_info_.sb_type != IMM32)
      sb_data.f32 = -sb_data.f32;
    frexp(ra_data.f32, &exp1);
    frexp(sb_data.f32, &exp2);
    if (exp1 <= emin + n - 1 || exp1 >= emax + 1 || exp2 <= emin ||
        exp2 >= emax - 2 || exp1 - exp2 <= emin + 1 || exp1 - exp2 > emax)
      pd0_data_[tid].pred = 0x1;
    else
      pd0_data_[tid].pred = 0x0;
  }
  PredOperand* pd0 =
      new PredOperand(warp_id_, decode_info_.pd0, thread_active_mask_);
  pd0->setData(pd0_data_);
  dst_preds_.push_back(pd0);
}

FSETPInstr::FSETPInstr(Core& sm, const uint64_t& pc, int warp_id,
                       const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void FSETPInstr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  boolop_ = H >> 10 & 0x3;
  cmpop_ = H >> 12 & 0xf;
  decode_info_.pd0 = H >> 17 & 0x7;
  decode_info_.pd1 = H >> 20 & 0x7;
  decode_info_.ps0 = H >> 23 & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;
  ftz_ = H >> ftz_shf & 0x1;
  ra_abs_ = H >> rs0abs_shf & 0x1;
  sb_abs_ = L >> rs1abs_shf & 0x1;
  decode_info_.rd_type = INVALID;
  decode_info_.sc_type = INVALID;
  decode_info_.operandSbScAssign();
  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  src_operands_.push_back(ra);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  src_operands_.push_back(sb);
  PredOperand* ps0 =
      new PredOperand(warp_id_, decode_info_.ps0, thread_active_mask_);
  src_preds_.push_back(ps0);
}

void FSETPInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* ps0_data_ = src_preds_[0]->getData();
  sass_reg_t* ps1_data_ = src_preds_[1]->getData();
  sass_reg_t pd0_data_[WARP_SIZE], pd1_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid];
    sass_reg_t ps0_data = ps0_data_[tid];
    if (ftz_) {
      if (!std::isnormal(ra_data.f32) && !std::isnan(ra_data.f32) &&
          !std::isinf(ra_data.f32))
        ra_data.f32 = 0.0f;
      if (!std::isnormal(sb_data.f32) && !std::isnan(sb_data.f32) &&
          !std::isinf(sb_data.f32))
        sb_data.f32 = 0.0f;
    }
    if (decode_info_.ra_neg) ra_data.f32 = -ra_data.f32;
    if (decode_info_.sb_neg && decode_info_.sb_type != IMM32)
      sb_data.f32 = -sb_data.f32;
    if (decode_info_.ps0_neg) ps0_data.pred = !ps0_data.pred;
    if (ra_abs_) ra_data.f32 = fabs(ra_data.f32);
    if (sb_abs_ && decode_info_.sb_type == REG32)
      sb_data.f32 = fabs(sb_data.f32);

    bool temp;
    switch (cmpop_) {
      case 0x1:
        temp = (ra_data.f32 < sb_data.f32);
        break;
      case 0x2:
        temp = (ra_data.f32 == sb_data.f32);
        break;
      case 0x3:
        temp = (ra_data.f32 <= sb_data.f32);
        break;
      case 0x4:
        temp = (ra_data.f32 > sb_data.f32);
        break;
      case 0x5:
        temp = (ra_data.f32 != sb_data.f32);
        break;
      case 0x6:
        temp = (ra_data.f32 >= sb_data.f32);
        break;
      case 0x9:
        temp = (ra_data.f32 < sb_data.f32);
        break;
      case 0xa:
        temp = (ra_data.f32 == sb_data.f32);
        break;
      case 0xb:
        temp = (ra_data.f32 <= sb_data.f32);
        break;
      case 0xc:
        temp = (ra_data.f32 > sb_data.f32);
        break;
      case 0xd:
        temp = (ra_data.f32 != sb_data.f32);
        break;
      case 0xe:
        temp = (ra_data.f32 >= sb_data.f32);
        break;
      default:
        break;
    }

    switch (boolop_) {
      case 0x0:
        pd0_data_[tid].pred = temp & ps0_data.pred;
        ;  // && ps0_data.u32;
        pd1_data_[tid].pred = !temp & ps0_data.pred;
        ;  //&& ps0_data.u32;
        break;
      case 0x1:
        pd0_data_[tid].pred = temp || ps0_data.pred;
        pd1_data_[tid].pred = !temp || ps0_data.pred;
        break;
      case 0x2:
        // TODO: check
        pd0_data_[tid].pred = temp ^ ps0_data.pred;
        pd1_data_[tid].pred = !temp ^ ps0_data.pred;
        break;
      default:
        break;
    }
  }
  PredOperand* pd0 =
      new PredOperand(warp_id_, decode_info_.pd0, thread_active_mask_);
  pd0->setData(pd0_data_);
  PredOperand* pd1 =
      new PredOperand(warp_id_, decode_info_.pd1, thread_active_mask_);
  pd1->setData(pd1_data_);
  dst_preds_.push_back(pd0);
  dst_preds_.push_back(pd1);
}