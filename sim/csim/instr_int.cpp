#include "instr_int.h"
#include "operand.h"
using namespace gpgpu;

IMADInstr::IMADInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void IMADInstr::decode() {
  // basic decode：指令中一些通用字段的解析
  Instruction::decode();

  // IMAD中特有的一些字段解码
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  X_ = H >> X_shf & 0x1;

  //设置操作数
  data_type_ = H >> dtype_shf & 0x1;
  decode_info_.pd0 = H >> pd0_shf & 0x7;
  decode_info_.ps0 = H >> ps0_shf & 0x7;  //目前pd0与ps0无作用，恒为全1
  // sb sc不为invalid要根据相关字段解析
  decode_info_.operandSbScAssign();
  //设置操作数正负号
  decode_info_.ps0_neg = H >> 26 & 0x1;
  switch (decode_info_.opcode_type) {
    case 0x1:
    case 0x4:
    case 0x5:
      decode_info_.sc_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sb_neg = L >> 63 & 0x1;
      break;
    default:
      decode_info_.sb_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sc_neg = L >> 63 & 0x1;
      if (decode_info_.sc_type == CMADDR) decode_info_.sc_type = CMADDR64;
      break;
  }

  //将源操作数封装成operandRW并push进vector，方便后续readOperand
  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  Operand* sc = new Operand(warp_id_, decode_info_.sc, thread_active_mask_,
                            decode_info_.sc_type);
  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
  src_operands_.push_back(sc);

  PredOperand* ps0 =
      new PredOperand(warp_id_, decode_info_.ps0, thread_active_mask_);
  src_preds_.push_back(ps0);
}

void IMADInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* sc_data_ = src_operands_[2]->getData();
  sass_reg_t rd_data_[WARP_SIZE];

  sass_reg_t* ps0_data_ = src_preds_[0]->getData();
  // sass_reg_t pd0_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    if (thread_active_mask_.test(tid)) {
      sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid],
                 sc_data = sc_data_[tid], pred_data = ps0_data_[tid], data;

      if (decode_info_.ra_neg)
        ra_data.u64 =
            X_ == 1 ? unsigned(~(ra_data.u32)) : unsigned(-ra_data.u32);
      if (decode_info_.sb_neg && decode_info_.sb_type != IMM32)  // 1=IMM32
        sb_data.u64 =
            X_ == 1 ? unsigned(~(sb_data.u32)) : unsigned(-sb_data.u32);
      if (decode_info_.sc_neg && decode_info_.sc_type != IMM32)  //
        sc_data.u64 =
            X_ == 1 ? unsigned(~(sc_data.u32)) : unsigned(-sc_data.u32);

      // if(X) data.pred += pred_data.pred;
      switch (data_type_) {
        case 1:  // s32
          data.s32 = ra_data.s32 * sb_data.s32;
          data.s32 += sc_data.s32;  // + data.pred;
          if (X_) data.s32 += pred_data.pred;
          rd_data_[tid].s32 = data.s32;
          break;
        case 0:  // u32
          data.u32 = ra_data.u32 * sb_data.u32;
          data.u32 += sc_data.u32;  //+ data.pred;
          if (X_) data.u32 += pred_data.pred;
          rd_data_[tid].u32 = data.u32;
          break;
        default:
          assert(0);
          break;
      }
    }
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

//// IMADWIDE///

IMADWIDEInstr::IMADWIDEInstr(Core& sm, const uint64_t& pc, int warp_id,
                             const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void IMADWIDEInstr::decode() {
  Instruction::decode();

  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  data_type_ = H >> dtype_shf & 0x1;
  X_ = H >> X_shf & 0x1;
  switch (decode_info_.opcode_type) {
    case 0x1:
    case 0x4:
    case 0x5:
      decode_info_.sc_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sb_neg = L >> 63 & 0x1;
      break;
    default:
      decode_info_.sb_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sc_neg = L >> 63 & 0x1;
      if (decode_info_.sc_type == CMADDR) decode_info_.sc_type = CMADDR64;
      break;
  }

  decode_info_.operandSbScAssign();
  decode_info_.rd_type = REG64;
  if (decode_info_.sc_type == REG32 && decode_info_.sc != 255)
    decode_info_.sc_type = REG64;

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

void IMADWIDEInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* sc_data_ = src_operands_[2]->getData();
  sass_reg_t rd_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    if (thread_active_mask_.test(tid)) {
      sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid],
                 sc_data = sc_data_[tid], data;

      if (decode_info_.ra_neg) ra_data.u64 = -ra_data.u32;
      if (decode_info_.sb_neg && decode_info_.sb_type != IMM32)
        sb_data.u64 = unsigned(-sb_data.u32);
      if (decode_info_.sc_neg && decode_info_.sc_type != IMM32)
        sc_data.u64 = unsigned(-sc_data.u64);

      switch (data_type_) {
        case 1:  //此处可能两种实现，可以乘加分开算或合一起算
          if (ra_data.u64 >> 31 == 1) ra_data.u64 += 0xffffffff00000000;
          if (sb_data.u64 >> 31 == 1) sb_data.u64 += 0xffffffff00000000;
          data.s64 = ra_data.s64 * sb_data.s64;
          data.s64 += sc_data.s64;
          break;
        case 0:
          data.u64 = ra_data.u64 * sb_data.u64;
          data.u64 += sc_data.u64;
          break;
        default:
          assert(0);
          break;
      }
      //低位放在rd，高位放在rd相邻的寄存器
      rd_data_[tid].u64 = data.u64;
    }
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

/// IADD3 ///
IADD3Instr::IADD3Instr(Core& sm, const uint64_t& pc, int warp_id,
                       const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void IADD3Instr::decode() {
  Instruction::decode();

  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  X_ = H >> X_shf & 0x1;
  decode_info_.ps1 = H >> ps1_shf & 0x7;
  decode_info_.pd0 = H >> pd0_shf & 0x7;
  decode_info_.pd1 = H >> pd1_shf & 0x7;
  decode_info_.ps0 = H >> ps0_shf & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;
  decode_info_.ps1_neg = H >> 16 & 0x1;
  decode_info_.ra_neg = H >> 8 & 0x1;
  switch (decode_info_.opcode_type) {
    case 0x1:
    case 0x4:
    case 0x5:
      decode_info_.sc_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sb_neg = L >> 63 & 0x1;
      break;
    default:
      decode_info_.sb_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sc_neg = L >> 63 & 0x1;
      if (decode_info_.sc_type == CMADDR) decode_info_.sc_type = CMADDR64;
      break;
  }

  decode_info_.operandSbScAssign();

  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  Operand* sc = new Operand(warp_id_, decode_info_.sc, thread_active_mask_,
                            decode_info_.sc_type);
  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
  src_operands_.push_back(sc);

  PredOperand* ps0 =
      new PredOperand(warp_id_, decode_info_.ps0, thread_active_mask_);
  src_preds_.push_back(ps0);
}

void IADD3Instr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* sc_data_ = src_operands_[2]->getData();
  sass_reg_t* ps0_data_ = src_preds_[0]->getData();

  sass_reg_t rd_data_[WARP_SIZE];
  sass_reg_t pd0_data_[WARP_SIZE];

  for (int tid = 0; tid < WARP_SIZE; tid++) {
    if (thread_active_mask_.test(tid)) {
      sass_reg_t ra_data, sb_data, sc_data, pred_data, data;
      int carry = 0;

      //先拓宽长度（同一符号），再符号转换
      ra_data.u32 = ra_data_[tid].u32;
      if (decode_info_.ra_neg)
        ra_data.u64 =
            (X_ == 1 ? unsigned(~ra_data.u32) : unsigned(-ra_data.u32));
      sb_data.u32 = sb_data_[tid].u32;
      if (decode_info_.sb_neg && decode_info_.sb_type != IMM32)
        sb_data.u64 =
            (X_ == 1 ? unsigned(~sb_data.u32) : unsigned(-sb_data.u32));
      sc_data.u32 = sc_data_[tid].u32;
      if (decode_info_.sc_neg && decode_info_.sc_type != IMM32)
        sc_data.u64 =
            (X_ == 1 ? unsigned(~sc_data.u32) : unsigned(-sc_data.u32));
      pred_data.pred = ps0_data_[tid].pred;

      //目前sass指令解析无data type，也没考虑overflow；只有进位在此处需要考虑
      // gpusim里面只有低于64位的数据类型才会考虑进位，这里是不是也应该低于32位才考虑进位
      data.u64 = (ra_data.u64 & 0x0FFFFFFFF) + (sb_data.u64 & 0x0FFFFFFFF) +
                 (sc_data.u64 & 0x0FFFFFFFF);
      if (X_) data.u64 += (pred_data.pred);
      carry = (data.u64 & 0x100000000) >> 32;
      if ((ra_data.u64 == 0 && decode_info_.ra_neg == 1) ||
          (sb_data.u64 == 0 && decode_info_.sb_neg == 1) ||
          (sc_data.u64 == 0 && decode_info_.sc_neg == 1))
        carry = 1;
      // set dst register
      rd_data_[tid].u32 = (data.u64 & 0x0FFFFFFFF);
      if (decode_info_.pd0 != 0x7) {  // pred register is not PT
        pd0_data_[tid].pred = carry;
      }
    }
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);

  PredOperand* pd =
      new PredOperand(warp_id_, decode_info_.pd0, thread_active_mask_);
  pd->setData(pd0_data_);
  dst_preds_.push_back(pd);
}

ISETPInstr::ISETPInstr(Core& sm, const uint64_t& pc, int warp_id,
                       const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void ISETPInstr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.ps1 = H >> 4 & 0x7;
  decode_info_.ps1_neg = H >> 7 & 0x1;
  setpex_ = H >> 8 & 0x1;
  data_type_ = H >> dtype_shf & 0x1;
  boolop_ = H >> 10 & 0x3;
  cmpop_ = H >> 12 & 0xf;
  decode_info_.pd0 = H >> 17 & 0x7;
  decode_info_.pd1 = H >> 20 & 0x7;
  decode_info_.ps0 = H >> 23 & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;
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
  PredOperand* ps1 =
      new PredOperand(warp_id_, decode_info_.ps1, thread_active_mask_);
  src_preds_.push_back(ps1);
}

void ISETPInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* ps0_data_ = src_preds_[0]->getData();
  sass_reg_t* ps1_data_ = src_preds_[1]->getData();
  sass_reg_t pd0_data_[WARP_SIZE], pd1_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid];
    sass_reg_t ps0_data = ps0_data_[tid], ps1_data = ps1_data_[tid];
    if (decode_info_.ps0_neg) ps0_data.pred = !ps0_data.pred;
    if (setpex_)
      if (decode_info_.ps1_neg) ps1_data.pred = !ps1_data.pred;
    bool temp;
    if (data_type_ == 1) {
      switch (cmpop_) {
        case 0x1:
          temp = (ra_data.s32 < sb_data.s32);
          if (setpex_) {
            temp |= (ra_data.s32 == sb_data.s32 && ps1_data.pred);
          }
          break;
        case 0x2:
          temp = (ra_data.s32 == sb_data.s32);
          if (setpex_) {
            temp = (ra_data.s32 == sb_data.s32) && (ps1_data.pred);
          }
          break;
        case 0x3:
          temp = (ra_data.s32 <= sb_data.s32);
          if (setpex_) {
            temp = (ra_data.s32 < sb_data.s32) ||
                   (ra_data.s32 == sb_data.s32 && ps1_data.pred);
          }
          break;
        case 0x4:
          temp = (ra_data.s32 > sb_data.s32);
          if (setpex_) {
            temp |= (ra_data.s32 == sb_data.s32 && ps1_data.pred);
          }
          break;
        case 0x5:
          temp = (ra_data.s32 != sb_data.s32);
          if (setpex_) {
            temp |= (ra_data.s32 == sb_data.s32) && (ps1_data.pred);
          }
          break;
        case 0x6:
          temp = (ra_data.s32 >= sb_data.s32);
          if (setpex_) {
            temp = (ra_data.s32 > sb_data.s32) ||
                   (ra_data.s32 == sb_data.s32 && ps1_data.pred);
          }
          break;
        default:
          break;
      }
    } else {
      switch (cmpop_) {
        case 0x1:
          temp = (ra_data.u32 < sb_data.u32);
          if (setpex_) {
            temp |= (ra_data.u32 == sb_data.u32 && ps1_data.pred);
          }
          break;
        case 0x2:
          temp = (ra_data.u32 == sb_data.u32);
          if (setpex_) {
            temp = (ra_data.u32 == sb_data.u32) && (ps1_data.pred);
          }
          break;
        case 0x3:
          temp = (ra_data.u32 <= sb_data.u32);
          if (setpex_) {
            temp = (ra_data.u32 < sb_data.u32) ||
                   (ra_data.u32 == sb_data.u32 && ps1_data.pred);
          }
          break;
        case 0x4:
          temp = (ra_data.u32 > sb_data.u32);
          if (setpex_) {
            temp |= (ra_data.u32 == sb_data.u32 && ps1_data.pred);
          }
          break;
        case 0x5:
          temp = (ra_data.u32 != sb_data.u32);
          if (setpex_) {
            temp |= (ra_data.u32 == sb_data.u32) && (ps1_data.pred);
          }
          break;
        case 0x6:
          temp = (ra_data.u32 >= sb_data.u32);
          if (setpex_) {
            temp = (ra_data.u32 > sb_data.u32) ||
                   (ra_data.u32 == sb_data.u32 && ps1_data.pred);
          }
          break;
        default:
          break;
      }
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

SHFInstr::SHFInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void SHFInstr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  data_type_ = H >> shf_dytpe_shf & 0xf;
  pos_ = H >> shfpos_shf & 0x1;
  hi_ = H >> shfhi_shf & 0x1;
  decode_info_.operandSbScAssign();
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

void SHFInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* sc_data_ = src_operands_[2]->getData();
  sass_reg_t rd_data_[WARP_SIZE];
  for (int tid = 0; tid < 32; tid++) {
    sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid],
               sc_data = sc_data_[tid];
    uint64_t val = ((sc_data.u64 << 32) | ra_data.u32);
    if (hi_) sb_data.u32 += 32;
    if (pos_) {
      if (data_type_ == 4) {
        rd_data_[tid].s32 = (((int64_t)val >> sb_data.u32) & 0xffffffff);
      } else {
        rd_data_[tid].u32 = (((uint64_t)val >> sb_data.u32) & 0xffffffff);
      }
    } else {
      val <<= sb_data.u32;
      rd_data_[tid].u32 = (val & 0xffffffff);
    }
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

LEAInstr::LEAInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void LEAInstr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  hi_ = H >> leahi_shf & 0x1;
  imm_ = H >> leaimm_shf & 0x1f;
  sx32_ = H >> leasx32_shf & 0x1;
  X_ = H >> X_shf & 0x1;
  decode_info_.ra_neg = H >> rs0neg_shf & 0x1;
  decode_info_.operandSbScAssign();
  switch (decode_info_.opcode_type) {
    case 0x1:
    case 0x4:
    case 0x5:
      decode_info_.sc_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sb_neg = L >> 63 & 0x1;
      break;
    default:
      decode_info_.sb_neg = H >> rs2neg_shf & 0x1;
      decode_info_.sc_neg = L >> 63 & 0x1;
      if (decode_info_.sc_type == CMADDR) decode_info_.sc_type = CMADDR64;
      break;
  }
  decode_info_.pd0 = H >> pd0_shf & 0x7;
  decode_info_.ps0 = H >> ps0_shf & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;
  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  Operand* sc = new Operand(warp_id_, decode_info_.sc, thread_active_mask_,
                            decode_info_.sc_type);
  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
  src_operands_.push_back(sc);
  PredOperand* ps0 =
      new PredOperand(warp_id_, decode_info_.ps0, thread_active_mask_);
  src_preds_.push_back(ps0);
}

void LEAInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* sc_data_ = src_operands_[2]->getData();
  sass_reg_t* ps0_data_ = src_preds_[0]->getData();
  sass_reg_t rd_data_[WARP_SIZE], pd0_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid],
               sc_data = sc_data_[tid], ps0_data = ps0_data_[tid];
    if (decode_info_.sb_neg && decode_info_.sb_type != IMM32)
      sb_data.u64 = (X_ ? unsigned(~sb_data.u32) : unsigned(-sb_data.u32));
    if (decode_info_.sc_neg && decode_info_.sc_type != IMM32)
      sc_data.u64 = (X_ ? unsigned(~sc_data.u32) : unsigned(-sc_data.u32));
    uint64_t val;
    if (decode_info_.sc != 255)
      val = ((sc_data.u64 << 32) | ra_data.u32);
    else {
      if (sx32_)
        val = ra_data.u32 +
              (((ra_data.u32 >> 31) == 1) ? 0xffffffff00000000 : 0x0);
      else
        val = ra_data.u32;
    }
    if (decode_info_.ra_neg) val = X_ ? ~val : -val;
    if (hi_) {
      if (sx32_)
        val = (int64_t)val >> (32 - imm_);
      else
        val = val >> (32 - imm_);
    } else
      val = val << imm_;
    val += sb_data.u32;
    if (X_) val += ps0_data.pred;
    if (decode_info_.pd0 != 7) {
      pd0_data_[tid].pred = ((val >> 32) & 0x1);
    }
    val &= 0xffffffff;
    rd_data_[tid].u32 = val;
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
  PredOperand* pd =
      new PredOperand(warp_id_, decode_info_.pd0, thread_active_mask_);
  pd->setData(pd0_data_);
  dst_preds_.push_back(pd);
}

LOP3Instr::LOP3Instr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void LOP3Instr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.operandSbScAssign();
  immLUT_ = H >> 8 & 0xff;
  decode_info_.ps0 = H >> ps0_shf & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;
  decode_info_.pd0 = H >> pd0_shf & 0x7;
  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  Operand* sc = new Operand(warp_id_, decode_info_.sc, thread_active_mask_,
                            decode_info_.sc_type);
  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
  src_operands_.push_back(sc);
  PredOperand* ps0 =
      new PredOperand(warp_id_, decode_info_.ps0, thread_active_mask_);
  src_preds_.push_back(ps0);
}

void LOP3Instr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* sc_data_ = src_operands_[2]->getData();
  sass_reg_t* ps0_data_ = src_preds_[0]->getData();
  sass_reg_t rd_data_[WARP_SIZE], pd0_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    sass_reg_t ra_data = ra_data_[tid], sb_data = sb_data_[tid],
               sc_data = sc_data_[tid];
    unsigned temp = 0;
    if (immLUT_ & 0x01)
      temp |= (~ra_data.u32) & (~sb_data.u32) & (~sc_data.u32);
    if (immLUT_ & 0x02) temp |= (~ra_data.u32) & (~sb_data.u32) & (sc_data.u32);
    if (immLUT_ & 0x04) temp |= (~ra_data.u32) & (sb_data.u32) & (~sc_data.u32);
    if (immLUT_ & 0x08) temp |= (~ra_data.u32) & (sb_data.u32) & (sc_data.u32);
    if (immLUT_ & 0x10) temp |= (ra_data.u32) & (~sb_data.u32) & (~sc_data.u32);
    if (immLUT_ & 0x20) temp |= (ra_data.u32) & (~sb_data.u32) & (sc_data.u32);
    if (immLUT_ & 0x40) temp |= (ra_data.u32) & (sb_data.u32) & (~sc_data.u32);
    if (immLUT_ & 0x80) temp |= (ra_data.u32) & (sb_data.u32) & (sc_data.u32);
    rd_data_[tid].u32 = temp;
    pd0_data_[tid].pred = temp != 0x0;
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
  PredOperand* pd =
      new PredOperand(warp_id_, decode_info_.pd0, thread_active_mask_);
  pd->setData(pd0_data_);
  dst_preds_.push_back(pd);
}