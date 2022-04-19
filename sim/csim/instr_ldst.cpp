#include "instr_ldst.h"
#include "operand.h"
#include "core.h"

using namespace gpgpu;

LDGInstr::LDGInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void LDGInstr::decode() {
  Instruction::decode();

  const uint64_t& H = sass_bin_.b64.H;
  E_ = H >> E_shf & 0x1;
  data_type_ = H >> dtype_shf & 0x7;
  scope_ = H >> scope_shf & 0x3;
  strong_ = H >> strong_shf & 0x3;
  cache_ = H >> cache_shf & 0x3;

  decode_info_.ldstSbScAssign();
  if (E_) decode_info_.ra_type = REG64;

  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);

  src_operands_.push_back(ra);
}

void LDGInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t rd_data_[WARP_SIZE];
  unsigned word_size;  // .U8, .S8, .U16, .S16, .32, .64, .128
  switch (data_type_) {
    case 0:  // .U8
    case 1:  // .S8
      word_size = 1;
      break;
    case 2:  // .U16
    case 3:  // .S16
      word_size = 2;
      break;
    case 4:  // .32
      word_size = 4;
      break;
    case 5:  // .64
      word_size = 8;
      break;
    case 6:  // .128
    case 7:  // .U.128
      word_size = 16;
      break;
    default:
      break;
  }

  for (int tid = 0; tid < WARP_SIZE; ++tid) {
    if (thread_active_mask_.test(tid)) {
      unsigned memreqaddr;
      memreqaddr = ra_data_[tid].u64 +
                   int(decode_info_.sb +
                       ((decode_info_.sb >> 23) == 1 ? 0xff000000 : 0x0));
      sm_.mem_->read(&rd_data_[tid], memreqaddr, word_size);
    }
  }

  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

/// STG ///
STGInstr::STGInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void STGInstr::decode() {
  Instruction::decode();

  const uint64_t& H = sass_bin_.b64.H;
  E_ = H >> E_shf & 0x1;
  data_type_ = H >> dtype_shf & 0x7;
  scope_ = H >> scope_shf & 0x3;
  strong_ = H >> strong_shf & 0x3;
  cache_ = H >> cache_shf & 0x3;

  decode_info_.ldstSbScAssign();
  if (E_) decode_info_.ra_type = REG64;
  decode_info_.rd_type = INVALID;

  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);
  // OperandRW* sc = new OperandRW(sc_data_, warp_id_, decode_info_.sc,
  //                               thread_active_mask_, decode_info_.sc_type);

  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
  // src_operands_.push_back(sc);
}

void STGInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();

  unsigned word_size;  // .U8, .S8, .U16, .S16, .32, .64, .128
  switch (data_type_) {
    case 0:  // .U8
    case 1:  // .S8
      word_size = 1;
      break;
    case 2:  // .U16
    case 3:  // .S16
      word_size = 2;
      break;
    case 4:  // .32
      word_size = 4;
      break;
    case 5:  // .64
      word_size = 8;
      break;
    case 6:  // .128
    case 7:  // .U.128
      word_size = 16;
      break;
    default:
      break;
  }

  for (int tid = 0; tid < WARP_SIZE; ++tid) {
    if (thread_active_mask_.test(tid)) {
      unsigned memreqaddr;
      memreqaddr = ra_data_[tid].u64 +
                   int((unsigned)decode_info_.sc +
                       ((decode_info_.sc >> 23) == 1 ? 0xff000000 : 0x0));
      sm_.mem_->write((&sb_data_[tid]), memreqaddr, word_size);
    }
  }
}

LDSInstr::LDSInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void LDSInstr::decode() {
  Instruction::decode();

  const uint64_t& H = sass_bin_.b64.H;
  data_type_ = H >> dtype_shf & 0x7;
  U = H >> 12 & 0x1;

  decode_info_.ldstSbScAssign();

  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);

  src_operands_.push_back(ra);
}

void LDSInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t rd_data_[WARP_SIZE];
  unsigned word_size;  // .U8, .S8, .U16, .S16, .32, .64, .128
  switch (data_type_) {
    case 0:  // .U8
    case 1:  // .S8
      word_size = 1;
      break;
    case 2:  // .U16
    case 3:  // .S16
      word_size = 2;
      break;
    case 4:  // .32
      word_size = 4;
      break;
    case 5:  // .64
      word_size = 8;
      break;
    case 6:  // .128
    case 7:  // .U.128
      word_size = 16;
      break;
    default:
      break;
  }

  uint32_t cta_id = warp_id_ / sm_.warp_num_in_cta;

  for (unsigned t = 0; t < WARP_SIZE; t++) {
    if (thread_active_mask_.test(t)) {
      unsigned memreqaddr =
          SHARED_GENERIC_START + sm_.sid_ * GPGPU_SHMEM_SIZE +
          cta_id * (sm_.kernel_uid->smem_size) + ra_data_[t].s32 +
          int(decode_info_.sb +
              ((decode_info_.sb >> 23) == 1 ? 0xff000000 : 0x0));
      for (int word = 0; word < (word_size / 4); word++) {
        unsigned reqaddr = line_size_based_tag_func(memreqaddr + word * 4, 4);
        sm_.mem_->read((((uint8_t*)(&rd_data_[t])) + word * 4), reqaddr, 4);
      }
    }
  }

  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data_);
  dst_operands_.push_back(rd);
}

STSInstr::STSInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void STSInstr::decode() {
  Instruction::decode();

  const uint64_t& H = sass_bin_.b64.H;
  data_type_ = H >> dtype_shf & 0x7;
  U = H >> 12 & 0x1;

  decode_info_.ldstSbScAssign();

  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
                            decode_info_.sb_type);

  src_operands_.push_back(ra);
  src_operands_.push_back(sb);
}

void STSInstr::execute() {
  sass_reg_t* ra_data_ = src_operands_[0]->getData();
  sass_reg_t* sb_data_ = src_operands_[1]->getData();

  unsigned word_size;  // .U8, .S8, .U16, .S16, .32, .64, .128
  switch (data_type_) {
    case 0:  // .U8
    case 1:  // .S8
      word_size = 1;
      break;
    case 2:  // .U16
    case 3:  // .S16
      word_size = 2;
      break;
    case 4:  // .32
      word_size = 4;
      break;
    case 5:  // .64
      word_size = 8;
      break;
    case 6:  // .128
    case 7:  // .U.128
      word_size = 16;
      break;
    default:
      break;
  }

  uint32_t cta_id = warp_id_ / sm_.warp_num_in_cta;

  for (int t = 0; t < WARP_SIZE; ++t) {
    if (thread_active_mask_.test(t)) {
      unsigned memreqaddr =
          SHARED_GENERIC_START + sm_.sid_ * GPGPU_SHMEM_SIZE +
          cta_id * sm_.kernel_uid->smem_size + ra_data_[t].s32 +
          int((unsigned)decode_info_.sc +
              ((decode_info_.sc >> 23) == 1 ? 0xff000000 : 0x0));
      for (int word = 0; word < (word_size / 4); word++) {
        unsigned reqaddr = line_size_based_tag_func(memreqaddr + word * 4, 4);
        sm_.mem_->write((((uint8_t*)(&sb_data_[t])) + word * 4), reqaddr, 4);
      }
    }
  }
}

LDCInstr::LDCInstr(Core& sm, const uint64_t& pc, int warp_id,
                   const SassCodeType& sass)
    : Instruction(sm, pc, warp_id, sass) {}

void LDCInstr::decode() {
  Instruction::decode();

  const uint64_t& H = sass_bin_.b64.H;
  data_type_ = H >> dtype_shf & 0x7;

  decode_info_.sc_type = INVALID;
  decode_info_.ldstSbScAssign();
  switch (data_type_) {
    case 4:
      decode_info_.rd_type = REG32;
      break;
    case 5:
      decode_info_.rd_type = REG64;
      break;
    case 6:
      decode_info_.rd_type = REG128;
      break;
    default:
      break;
  }

  Operand* ra = new Operand(warp_id_, decode_info_.ra, thread_active_mask_,
                            decode_info_.ra_type);
  // Operand* sb = new Operand(warp_id_, decode_info_.sb, thread_active_mask_,
  //                           decode_info_.sb_type);

  src_operands_.push_back(ra);
  // src_operands_.push_back(sb);
}

void LDCInstr::execute() {
  // ldc需要重新计算一次sb的地址，因为它的地址是要在ra的基础上算
  uint32_t cm_addr_high = (decode_info_.sb >> 16 & 0xf);
  uint32_t cm_addr_low = (decode_info_.sb & 0xffff);

  sass_reg_t* ra_data = src_operands_[0]->getData();

  //从const mem里拿数据放到data
  sass_reg_t rd_data[WARP_SIZE];
  for (unsigned tid = 0; tid < WARP_SIZE; ++tid) {
    if (thread_active_mask_.test(tid)) {
      unsigned addr = CONSTANT_MEM_START + CONSTANT_BANK_SIZE * cm_addr_high +
                      cm_addr_low + ra_data[tid].u32;
      unsigned size = 4;
      if (data_type_ == 5)
        size = 8;
      else if (data_type_ == 6)
        size = 16;
      sass_reg_t rs_const_mem;
      sm_.mem_->read(&rs_const_mem, addr, size);
      rd_data[tid] = rs_const_mem;
    }
  }
  Operand* rd = new Operand(warp_id_, decode_info_.rd, thread_active_mask_,
                            decode_info_.rd_type);
  rd->setData(rd_data);
  dst_operands_.push_back(rd);
}