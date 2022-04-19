#include "instr_miscell.h"
#include "operand.h"
#include "core.h"
using namespace gpgpu;

SHFLInstr::SHFLInstr(Core& sm, const uint64_t& pc, int warp_id,
                     const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void SHFLInstr::decode() {
  Instruction::decode();
  shfl_type = sass_bin_.b64.L >> 58 & 0x3;
  // shfl不满足当opcode_type=2时不满足operandAssign
  if (decode_info_.opcode_type == 2) {
    decode_info_.sb = sass_bin_.b64.L >> rs1_shf & reg_mask;
    decode_info_.sb_type = REG32;
    decode_info_.sc = sass_bin_.b64.L >> 40 & 0x1fff;
    decode_info_.sc_type = IMM32;
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
}

void SHFLInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();
  sass_reg_t* sc_data_ = src_operands_[2]->getData();
  sass_reg_t rd_data_[WARP_SIZE];
  for (int tid = 0; tid < WARP_SIZE; tid++) {
    if (thread_active_mask_.test(tid)) {
      sass_reg_t ra_data;
      sass_reg_t sb_data = sb_data_[tid];
      sass_reg_t sc_data = sc_data_[tid];
      switch (shfl_type) {
        case 2: {
          unsigned temp = tid + sb_data.u32;
          if (temp > sc_data.u32) temp = sc_data.u32;
          ra_data = ra_data_[temp];
          break;
        }
        default:
          break;
      }
      rd_data_[tid] = ra_data;
    }
  }

  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

//////////////////// S2R////////////////////
S2RInstr::S2RInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void S2RInstr::decode() {
  Instruction::decode();
  sreg_ = sass_bin_.b64.H >> csr_shf & csr_mask;
  decode_info_.ra_type = INVALID;
  decode_info_.sb_type = INVALID;
  decode_info_.sc_type = INVALID;
}

void S2RInstr::execute() {
  sass_reg_t rd_data_[WARP_SIZE];

  for (int tid = 0; tid < WARP_SIZE; tid++) {
    uint32_t hw_cta_idx = warp_id_ / sm_.warp_num_in_cta;
    dim3 cta_id = sm_.cta_id_list[hw_cta_idx];
    uint32_t tid_idx = (warp_id_ % sm_.warp_num_in_cta) * WARP_SIZE + tid;
    uint32_t tid_x = tid_idx % (sm_.cta_dim).x;
    uint32_t tid_y = (tid_idx / (sm_.cta_dim).x) % (sm_.cta_dim).y;
    uint32_t tid_z =
        (tid_idx / ((sm_.cta_dim).x * (sm_.cta_dim).y)) % (sm_.cta_dim).z;

    switch (sreg_) {
      case (SR_TID_X):
        rd_data_[tid].u32 = tid_x;
        // assert(m_warp_info[warp_id].tid[tid].x == tid_x);
        break;
      case (SR_TID_Y):
        rd_data_[tid].u32 = tid_y;
        // assert(m_warp_info[warp_id].tid[tid].y == tid_y);
        break;
      case (SR_TID_Z):
        rd_data_[tid].u32 = tid_z;
        // assert(m_warp_info[warp_id].tid[tid].z == tid_z);
        break;
      case (SR_CTID_X):
        rd_data_[tid].u32 = cta_id.x;
        // assert(m_warp_info[warp_id].ctaid.x == rd_data_[tid].u32);
        break;
      case (SR_CTID_Y):
        rd_data_[tid].u32 = cta_id.y;
        // assert(m_warp_info[warp_id].ctaid.y == rd_data_[tid].u32);
        break;
      case (SR_CTID_Z):
        rd_data_[tid].u32 = cta_id.z;
        // assert(m_warp_info[warp_id].ctaid.z == rd_data_[tid].u32);
        break;
      default:
        assert(0);
        break;
    }
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

BARInstr::BARInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void BARInstr::decode() {
  decode_info_.rd_type = INVALID;
  decode_info_.ra_type = INVALID;
  decode_info_.sb_type = INVALID;
  decode_info_.sc_type = INVALID;
}

void BARInstr::execute() {
  sm_.bar_stall.set(warp_id_);
  sm_.cta_threads[warp_id_ / sm_.warp_num_in_cta] -=
      sm_.warp_info[warp_id_].thread_active.count();
}
