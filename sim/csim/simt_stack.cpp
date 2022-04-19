#include "simt_stack.h"

#include "debug.h"
using gpgpu::SimtStack;

void SimtStack::push(const uint64_t &pc, const ThreadMask &mask) {
  PcAndMask data;
  data.mask = mask;
  data.pc = pc;
  simt_stack_.push(data);
}

SimtStack::PcAndMask SimtStack::pop() {
  if (!simt_stack_.empty()) {
    PcAndMask result = simt_stack_.top();
    simt_stack_.pop();
    return result;
  }
  assert(0);
}