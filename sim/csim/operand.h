#pragma once
#include "types.h"
#include <fstream>
namespace gpgpu {
class RegFile;
class RAM;
class PredRegfile;

class Operand {
  friend std::fstream& operator<<(std::fstream& fs, const Operand& op);

 public:
  Operand(unsigned wid, unsigned regnum, const ThreadMask& thread_active_mask,
          oprand_data_type_t type);
  void readFrom(RegFile& reg_file, RAM*& mmu);
  void writeTo(RegFile& reg_file);
  sass_reg_t* getData();
  void setData(sass_reg_t* data);

 private:
  sass_reg_t data_[WARP_SIZE];
  unsigned wid_;
  unsigned regnum_;
  const ThreadMask& thread_active_mask_;
  oprand_data_type_t type_;
};

class PredOperand {
  friend std::fstream& operator<<(std::fstream& fs, const PredOperand& op);

 public:
  PredOperand(unsigned wid, unsigned regnum,
              const ThreadMask& thread_active_mask);
  void readFrom(PredRegfile& pred_reg_file);
  void writeTo(PredRegfile& pred_reg_file);
  sass_reg_t* getData();
  void setData(sass_reg_t* data);

 private:
  sass_reg_t data_[WARP_SIZE];
  unsigned wid_;
  unsigned regnum_;
  const ThreadMask& thread_active_mask_;
};

}  // namespace gpgpu