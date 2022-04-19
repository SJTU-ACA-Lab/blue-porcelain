#include "operand.h"
#include "regfile.h"
#include "mem.h"
using namespace gpgpu;

Operand::Operand(unsigned wid, unsigned regnum,
                 const ThreadMask& thread_active_mask, oprand_data_type_t type)
    : wid_(wid),
      regnum_(regnum),
      thread_active_mask_(thread_active_mask),
      type_(type) {}

void Operand::readFrom(RegFile& reg_file, RAM*& mmu) {
  switch (type_) {
    case REG32:  // REG
    {
      reg_file.read(wid_, regnum_, thread_active_mask_, data_, 0);
      break;
    }
    case IMM32: {
      for (unsigned i = 0; i < WARP_SIZE; ++i) {
        if (thread_active_mask_.test(i)) {
          data_[i] = regnum_;
        }
      }
      break;
    }
    case CMADDR:
    case CMADDR64: {
      uint32_t cm_addr_high = (regnum_ >> 16 & 0xf);
      uint32_t cm_addr_low = (regnum_ & 0xffff);
      unsigned addr =
          CONSTANT_MEM_START + CONSTANT_BANK_SIZE * cm_addr_high + cm_addr_low;
      sass_reg_t rs_const_mem;
      unsigned size = (type_ == CMADDR64) ? 8 : 4;
      mmu->read(&rs_const_mem, addr, size);

      for (unsigned i = 0; i < WARP_SIZE; ++i) {
        if (thread_active_mask_.test(i)) {
          data_[i] = rs_const_mem;
        }
      }
      break;
    }
    case REG64: {
      for (int i = 0; i < 2; ++i) {
        reg_file.read(wid_, regnum_ + i, thread_active_mask_, data_, i);
      }
      break;
    }
    case REG128: {
      for (int i = 0; i < 4; ++i) {
        reg_file.read(wid_, regnum_ + i, thread_active_mask_, data_, i);
      }
      break;
    }
    default:
      assert(0);
      break;
  }
}

void Operand::writeTo(RegFile& reg_file) {
  if (type_ == INVALID) return;
  int rd_reg_num = 0;
  switch (type_) {
    case REG32:  // REG
      rd_reg_num = 1;
      break;
    case REG64:
      rd_reg_num = 2;
      break;
    case REG128:
      rd_reg_num = 4;
      break;
    default:
      assert(0);
      break;
  }
  for (int i = 0; i < rd_reg_num; ++i) {
    if (regnum_ + i == 255) {
      break;
    }
    reg_file.write(wid_, regnum_ + i, thread_active_mask_, data_, i);
  }
}

sass_reg_t* Operand::getData() { return data_; }

void Operand::setData(sass_reg_t* data) {
  for (int tid = 0; tid < WARP_SIZE; ++tid) {
    data_[tid] = data[tid];
  }
}

std::fstream& gpgpu::operator<<(std::fstream& fs, const Operand& op) {
  for (int tid = 0; tid < WARP_SIZE; ++tid) {
    if (op.thread_active_mask_.test(tid)) {
      fs << "R" << op.regnum_ << std::dec << "_T" << tid << ": ";
      if (op.type_ == REG128) {
        fs << std::hex << "[0x" << op.data_[tid].u128.highest
           << op.data_[tid].u128.high << op.data_[tid].u128.low
           << op.data_[tid].u128.lowest << "] ";
      } else {
        fs << std::hex << "[0x" << std::setw(8) << std::setfill('0')
           << op.data_[tid].u64 << "], ";
      }
    }
  }
  fs << std::endl;
  return fs;
}

/// PredOpeerandRW///
PredOperand::PredOperand(unsigned wid, unsigned regnum,
                         const ThreadMask& thread_active_mask)
    : wid_(wid), regnum_(regnum), thread_active_mask_(thread_active_mask) {}
void PredOperand::readFrom(PredRegfile& pred_reg_file) {
  pred_reg_file.read(wid_, regnum_, thread_active_mask_, data_);
}
void PredOperand::writeTo(PredRegfile& pred_reg_file) {
  pred_reg_file.write(wid_, regnum_, thread_active_mask_, data_);
}

sass_reg_t* PredOperand::getData() { return data_; }

void PredOperand::setData(sass_reg_t* data) {
  for (int tid = 0; tid < WARP_SIZE; ++tid) {
    data_[tid] = data[tid];
  }
}

std::fstream& gpgpu::operator<<(std::fstream& fs, const PredOperand& op) {
  for (int tid = 0; tid < WARP_SIZE; ++tid) {
    if (op.thread_active_mask_.test(tid)) {
      fs << "P" << op.regnum_ << std::dec << "_T" << tid << ":";
      fs << std::hex << " : [" << op.data_[tid].pred << "] ";
    }
  }
  fs << std::endl;
  return fs;
}
