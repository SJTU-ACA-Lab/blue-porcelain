#include "instr_ctrl.h"

#include "core.h"
#include "simt_stack.h"
using namespace gpgpu;

EXITInstr::EXITInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void EXITInstr::decode() {
  Instruction::decode();

  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.rd_type = INVALID;
  decode_info_.ra_type = INVALID;
  decode_info_.sb_type = INVALID;
  decode_info_.sc_type = INVALID;
  decode_info_.ps0 = H >> ps0_shf & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;
}

void EXITInstr::execute() {
  sm_.warp_thread_left[warp_id_] -= thread_active_mask_.count();
  sm_.warp_info[warp_id_].thread_active &= (~thread_active_mask_);
  //第warp_id_ / (sm_.warp_num_in_cta)个cta中剩余活跃的线程数, for barrier
  sm_.cta_threads[warp_id_ / (sm_.warp_num_in_cta)] -=
      thread_active_mask_.count();
  if (sm_.warp_thread_left[warp_id_] == 0) {
    sm_.warp_exit.set(warp_id_);
    return;
  }
  if (sm_.warp_info[warp_id_].thread_active.none()) {
    auto pcandmask = sm_.simt_stack_unit[warp_id_].pop();
    sm_.warp_info[warp_id_].pc = pcandmask.pc;
    sm_.warp_info[warp_id_].thread_active = pcandmask.mask;
  }
}

BRAInstr::BRAInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void BRAInstr::decode() {
  Instruction::decode();

  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.rd_type = INVALID;
  decode_info_.ra_type = INVALID;
  decode_info_.sb_type = IMM32;
  decode_info_.sb = L >> 32 & 0xffffffff;
  decode_info_.sc_type = INVALID;
  decode_info_.ps0 = H >> ps0_shf & 0x7;
  decode_info_.ps0_neg = H >> 26 & 0x1;
  // 暂时不考虑.U扩展
  // decode_info_.brau = L >> 32 & 0x1;
  PredOperand* ps0 =
      new PredOperand(warp_id_, decode_info_.ps0, thread_active_mask_);
  src_preds_.push_back(ps0);
}

void BRAInstr::execute() {
  sass_reg_t* ps0_data = src_preds_[0]->getData();
  ThreadMask ps0_mask(0);
  for (int i = 0; i < WARP_SIZE; ++i) {
    if ((ps0_data[i].pred == 1 && !decode_info_.ps0_neg) ||
        (ps0_data[i].pred == 0 && decode_info_.ps0_neg)) {
      ps0_mask.set(i);
    }
  }

  auto taken_mask = thread_active_mask_ & ps0_mask;
  auto untaken_mask = sm_.warp_info[warp_id_].thread_active & ~taken_mask;
  auto taken_pc = (sm_.warp_info[warp_id_].pc + decode_info_.sb) & 0xffffffff;
  if (taken_mask.none()) return;
  if (untaken_mask.none()) {
    sm_.warp_info[warp_id_].pc = taken_pc;
    return;
  }

  if (taken_mask.count() < untaken_mask.count()) {
    sm_.simt_stack_unit[warp_id_].push(sm_.warp_info[warp_id_].pc,
                                       untaken_mask);
    sm_.warp_info[warp_id_].pc = taken_pc;
    sm_.warp_info[warp_id_].thread_active = taken_mask;
  } else {
    sm_.simt_stack_unit[warp_id_].push(taken_pc, taken_mask);
    sm_.warp_info[warp_id_].thread_active = untaken_mask;
  }
}

BSSYInstr::BSSYInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void BSSYInstr::decode() {
  Instruction::decode();

  const uint64_t& L = sass_bin_.b64.L;
  decode_info_.ra_type = INVALID;
  decode_info_.sb_type = IMM32;
  decode_info_.sb = L >> 32 & 0xffffffff;
  decode_info_.sc_type = INVALID;
}

void BSSYInstr::execute() {
  sm_.barrier_reg_file_.setParticipationMask(
      warp_id_, decode_info_.rd, sm_.warp_info[warp_id_].thread_active);
  sm_.barrier_reg_file_.setJoinedMask(warp_id_, decode_info_.rd);
  sm_.barrier_reg_file_.setRpc(warp_id_, decode_info_.rd,
                               sm_.warp_info[warp_id_].pc + decode_info_.sb);
}

BREAKInstr::BREAKInstr(Core& sm, const uint64_t& pc, int warp_id,
                       const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void BREAKInstr::decode() { Instruction::decode(); }

void BREAKInstr::execute() {
  auto new_participation_mask =
      sm_.barrier_reg_file_.getParticipationMask(warp_id_, decode_info_.rd) &
      ~thread_active_mask_;
  sm_.barrier_reg_file_.setParticipationMask(warp_id_, decode_info_.rd,
                                             new_participation_mask);
  auto joined_mask =
      sm_.barrier_reg_file_.getJoinedMask(warp_id_, decode_info_.rd);
  auto rpc = sm_.barrier_reg_file_.getRpc(warp_id_, decode_info_.rd);
  if (new_participation_mask == joined_mask) {
    sm_.simt_stack_unit[warp_id_].push(rpc, joined_mask);
  }
}

BSYNCInstr::BSYNCInstr(Core& sm, const uint64_t& pc, int warp_id,
                       const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void BSYNCInstr::decode() { Instruction::decode(); }

void BSYNCInstr::execute() {
  auto new_joined_mask =
      sm_.barrier_reg_file_.getJoinedMask(warp_id_, decode_info_.rd) |
      thread_active_mask_;
  sm_.barrier_reg_file_.setJoinedMask(warp_id_, decode_info_.rd,
                                      new_joined_mask);
  auto participation_mask =
      sm_.barrier_reg_file_.getParticipationMask(warp_id_, decode_info_.rd);
  if (new_joined_mask == participation_mask) {
    sm_.barrier_reg_file_.setJoinedMask(warp_id_, decode_info_.rd);
    sm_.warp_info[warp_id_].pc =
        sm_.barrier_reg_file_.getRpc(warp_id_, decode_info_.rd);
    sm_.warp_info[warp_id_].thread_active = new_joined_mask;
  } else {
    auto pcandmask = sm_.simt_stack_unit[warp_id_].pop();
    sm_.warp_info[warp_id_].pc = pcandmask.pc;
    sm_.warp_info[warp_id_].thread_active = pcandmask.mask;
  }
}

RETInstr::RETInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void RETInstr::decode() {
  Instruction::decode();

  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  ret_type_ = H >> 21 & 0x1;
  nodec_ = H >> 22 & 0x1;

  decode_info_.rd_type = INVALID;
  decode_info_.ra_type = ret_type_ == 1 ? REG64 : REG32;
  decode_info_.sb_type = IMM32;
  decode_info_.sc_type = INVALID;
  decode_info_.sb = L >> 32 & 0xffffffff;
  decode_info_.ps0 = H >> ps0_shf & 0x7;

  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);

  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
}

void RETInstr::execute() {
  uint64_t sch_o_ret_pc;
  sass_reg_t* ra_data = src_operands_[0]->getData();
  sass_reg_t* sb_data = src_operands_[1]->getData();
  for (int i = 0; i < WARP_SIZE; ++i) {
    if (thread_active_mask_.test(i)) {
      if (ret_type_ == 0)
        sch_o_ret_pc =
            (ra_data[i].u32 + sb_data[i].u32 + pc_ + 0x10) & 0xffffffff;
      else
        sch_o_ret_pc = ra_data[i].u64 + sb_data[i].u32;
      sm_.warp_info[warp_id_].pc = sch_o_ret_pc;
      break;
    }
  }
}

CALABSInstr::CALABSInstr(Core& sm, const uint64_t& pc, int warp_id,
                         const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void CALABSInstr::decode() {
  Instruction::decode();

  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.ra_type = INVALID;
  decode_info_.rd_type = INVALID;
  decode_info_.sb_type = IMM32;
  decode_info_.sb = L >> 32 & 0xffffffff;
  decode_info_.sc_type = INVALID;
  noinc = H >> 22 & 0x1;
}

void CALABSInstr::execute() {}

CALRELInstr::CALRELInstr(Core& sm, const uint64_t& pc, int warp_id,
                         const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void CALRELInstr::decode() {
  Instruction::decode();

  const uint64_t &H = sass_bin_.b64.H, &L = sass_bin_.b64.L;
  decode_info_.ra_type = INVALID;
  decode_info_.rd_type = INVALID;
  decode_info_.sb_type = IMM32;
  decode_info_.sb = L >> 32 & 0xffffffff;
  decode_info_.sc_type = INVALID;
  noinc = H >> 22 & 0x1;
}

void CALRELInstr::execute() {
  for (int i = 0; i < WARP_SIZE; ++i) {
    if (thread_active_mask_.test(i)) {
      sm_.warp_info[warp_id_].pc = (pc_ + 0x10 + decode_info_.sb) & 0xffffffff;
      break;
    }
  }
}