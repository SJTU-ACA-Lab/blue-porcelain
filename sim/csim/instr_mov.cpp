#include "instr_mov.h"
#include "operand.h"
#include "operandTypeAssigner.h"

using namespace gpgpu;

MovInstr::MovInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void MovInstr::decode() {
  Instruction::decode();

  decode_info_.ra_type = INVALID;
  decode_info_.sc_type = INVALID;
  decode_info_.operandSbScAssign();

  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  src_operands_.push_back(sb);
}

void MovInstr::execute() {
  sass_reg_t* sb_data_ = src_operands_[0]->getData();
  sass_reg_t rd_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; ++tid) {
    if (thread_active_mask_.test(tid)) rd_data_[tid].u32 = sb_data_[tid].u32;
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

SELInstr::SELInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void SELInstr::decode() {
  Instruction::decode();

  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;

  decode_info_.operandSbScAssign();
  decode_info_.sc_type = INVALID;

  decode_info_.ps0 = H >> ps0_shf & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;

  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);

  src_operands_.push_back(ra);
  src_operands_.push_back(sb);

  PredOperand* ps0 =
      new PredOperand(warp_id_, decode_info_.ps0, thread_active_mask_);
  src_preds_.push_back(ps0);
}

void SELInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t rd_data_[WARP_SIZE];

  sass_reg_t* ps0_data_ = src_preds_[0]->getData();

  for (int tid = 0; tid < WARP_SIZE; ++tid) {
    if (thread_active_mask_.test(tid)) {
      sass_reg_t ra_data, sb_data, ps0_data;

      ra_data.u32 = ra_data_[tid].u32;
      sb_data.u32 = sb_data_[tid].u32;
      ps0_data.pred = ps0_data_[tid].pred;
      if (decode_info_.ps0_neg == 1) {
        ps0_data.pred = 1 - ps0_data.pred;
      }

      if (ps0_data.pred == 1) {
        rd_data_[tid].u32 = ra_data.u32;
      } else {
        rd_data_[tid].u32 = sb_data.u32;
      }
    }
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}
