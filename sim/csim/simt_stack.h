#pragma once

#include <stack>
#include "types.h"
namespace gpgpu {

class SimtStack {
 public:
  struct PcAndMask {
    uint64_t pc;
    ThreadMask mask;
  };

  void push(const uint64_t &pc, const ThreadMask &mask);
  PcAndMask pop();
  // output
 private:
  std::stack<PcAndMask> simt_stack_;
};

}  // namespace gpgpu