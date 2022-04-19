#include "instruction.h"
#include "core.h"
#include "operand.h"
#include "debug.h"
using namespace gpgpu;
Instruction::Instruction(Core& sm, const uint64_t& pc, int warp_id,
                         const SassCodeType& sass)
    : sm_(sm),
      pc_(pc),
      warp_id_(warp_id),
      sass_bin_(sass),
      decode_info_(sass) {}

Instruction::~Instruction() {
  for (auto& op : src_operands_) {
    delete op;
  }
  for (auto& op : dst_operands_) {
    delete op;
  }
  for (auto& op : src_preds_) {
    delete op;
  }
  for (auto& op : dst_preds_) {
    delete op;
  }
}

void Instruction::decode() {
  decode_info_.basicDecode();
  readPrevPredicate();
}

void Instruction::readOperand() {
  //读取所有的源操作数
  for (auto& op : src_operands_) {
    op->readFrom(sm_.registerfile, sm_.mem_);
  }
  //读取所有的源pred
  for (auto& op : src_preds_) {
    op->readFrom(sm_.pred_reg_file_);
  }
}
void Instruction::writeBack() {
  // dst写回
  for (auto& op : dst_operands_) {
    op->writeTo(sm_.registerfile);
  }
  // pred写回
  for (auto& op : dst_preds_) {
    op->writeTo(sm_.pred_reg_file_);
  }
#if (DEBUG_MODE)
  DebugDisplay ds(*this, sm_.id());
  ds.printBasicInfo();
  ds.printRegInfo();
#endif
}

void Instruction::readPrevPredicate() {
  //读@p寄存器设置thread_active_mask_
  int pred_id = decode_info_.predicate;
  bool neg = decode_info_.p_neg;
  ThreadMask pred_value;

  if (pred_id == 7) {
    pred_value.set();
  } else {
    sass_reg_t pred_vec[WARP_SIZE];
    sm_.pred_reg_file_.read(warp_id_, pred_id,
                            sm_.warp_info[warp_id_].thread_active, pred_vec);
    for (int j = 0; j < WARP_SIZE; ++j) {
      if (pred_vec[j].u64 == 1) pred_value.set(j);
    }
    pred_value = neg ? ~pred_value : pred_value;
  }
  thread_active_mask_ = pred_value & sm_.warp_info[warp_id_].thread_active;
}
