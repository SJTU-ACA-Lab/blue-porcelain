#include "instr_pred.h"
#include "operand.h"
#include "operandTypeAssigner.h"

using namespace gpgpu;

PLOP3Instr::PLOP3Instr(Core& sm, const uint64_t& pc, int warp_id,
                       const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void PLOP3Instr::decode() {
  Instruction::decode();
  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.pd0 = H >> pd0_shf & 0x7;
  decode_info_.pd1 = H >> pd1_shf & 0x7;
  decode_info_.ps0 = H >> ps0_shf & 0x7;
  decode_info_.ps1 = H >> ps1_shf & 0x7;
  decode_info_.ps2 = H >> 4 & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;
  decode_info_.ps1_neg = H >> 16 & 0x1;
  immLUT = (H >> 8 & 0x1f) << 3;
  immLUT += H & 0x7;
  PredOperand* ps0 =
      new PredOperand(warp_id_, decode_info_.ps0, thread_active_mask_);
  src_preds_.push_back(ps0);
  PredOperand* ps1 =
      new PredOperand(warp_id_, decode_info_.ps1, thread_active_mask_);
  src_preds_.push_back(ps1);
  PredOperand* ps2 =
      new PredOperand(warp_id_, decode_info_.ps2, thread_active_mask_);
  src_preds_.push_back(ps2);
}

void PLOP3Instr::execute() {
  sass_reg_t* ps0_data_ = src_preds_[0]->getData();
  sass_reg_t* ps1_data_ = src_preds_[0]->getData();
  sass_reg_t* ps2_data_ = src_preds_[0]->getData();
  sass_reg_t pd0_data_[WARP_SIZE], pd1_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    if (thread_active_mask_.test(tid)) {
      sass_reg_t ps0_data = ps0_data_[tid], ps1_data = ps1_data_[tid],
                 ps2_data = ps2_data_[tid];
      bool temp = false;
      if (decode_info_.ps0_neg) ps0_data.pred = !ps0_data.pred;
      if (decode_info_.ps1_neg) ps1_data.pred = !ps1_data.pred;
      if (immLUT & 0x01)
        temp |= (~ps0_data.pred) & (~ps1_data.pred) & (~ps2_data.pred);
      if (immLUT & 0x02)
        temp |= (~ps0_data.pred) & (~ps1_data.pred) & (ps2_data.pred);
      if (immLUT & 0x04)
        temp |= (~ps0_data.pred) & (ps1_data.pred) & (~ps2_data.pred);
      if (immLUT & 0x08)
        temp |= (~ps0_data.pred) & (ps1_data.pred) & (ps2_data.pred);
      if (immLUT & 0x10)
        temp |= (ps0_data.pred) & (~ps1_data.pred) & (~ps2_data.pred);
      if (immLUT & 0x20)
        temp |= (ps0_data.pred) & (~ps1_data.pred) & (ps2_data.pred);
      if (immLUT & 0x40)
        temp |= (ps0_data.pred) & (ps1_data.pred) & (~ps2_data.pred);
      if (immLUT & 0x80)
        temp |= (ps0_data.pred) & (ps1_data.pred) & (ps2_data.pred);
      pd0_data_[tid].pred = temp;
      pd1_data_[tid].pred = !temp;
    }
    PredOperand* pd0 =
        new PredOperand(warp_id_, decode_info_.pd0, thread_active_mask_);
    pd0->setData(pd0_data_);
    dst_preds_.push_back(pd0);
    PredOperand* pd1 =
        new PredOperand(warp_id_, decode_info_.pd1, thread_active_mask_);
    pd1->setData(pd1_data_);
    dst_preds_.push_back(pd1);
  }
}