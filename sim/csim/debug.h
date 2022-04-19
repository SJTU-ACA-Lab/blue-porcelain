#pragma once
#include <algorithm>
#include <climits>
#include <map>
#include <vector>
#include "operand.h"
#include "instruction.h"
#include "types.h"

#include <iomanip>
#include <iostream>

// 文件debug信息开关
#define DEBUG_MODE 0

namespace gpgpu {
class DebugDisplay {
 public:
  DebugDisplay(const Instruction& wp, int sid) : warp_(wp), sid_(sid) {
    fs_.open(filename_, std::ios::app);
  }
  void printBasicInfo() {
    if (warpSelected()) {
      fs_ << std::dec << "SM: [" << sid_ << "], WARP ID: [" << std::setw(2)
          << std::setfill('0') << warp_.warp_id_;
      fs_ << "], PC: [0x" << std::hex << std::setw(4) << std::setfill('0')
          << warp_.pc_ - STARTUP_ADDR << "], SASS: [" << warp_.sass_bin_
          << "], ACTIVE_MASK: [0x" << std::hex
          << warp_.thread_active_mask_.to_ulong() << "] ";
      fs_ << std::endl;
    }
  }

  void printRegInfo() {
    if (warpSelected()) {
      for (auto& op : warp_.dst_preds_) {
        fs_ << *op;
      }
      for (auto& op : warp_.dst_operands_) {
        fs_ << *op;
      }
      for (auto& op : warp_.src_operands_) {
        fs_ << *op;
      }
      for (auto& op : warp_.src_preds_) {
        fs_ << *op;
      }
      fs_ << std::endl;
    }
  }

 private:
  std::string filename_ = "gpgpu_trace.log";
  std::fstream fs_;
  const Instruction& warp_;
  int sid_;
  std::vector<int> warps_ = {1};

  bool warpSelected() const {
    auto it = std::find(warps_.begin(), warps_.end(), warp_.warp_id_);
    return warps_.empty() || (it != warps_.end());
  }
};
}  // namespace gpgpu