#pragma once

#include <stdint.h>

#include <string>
#include <vector>

#include "vector_types.h"
// #include "core.h"

class KernelInfoType {
 private:
  /* data */
  dim3 tid;    // have allocated the tid id of BlockDim
  dim3 ctaid;  // have allocated the cta id of GridDim

  std::vector<uint32_t> ctaid_in_sm;  // [0, 1, ..., SM_NUM, 0, 1, ..., SM_NUM]

  // dim3 next_ctaid;
  // dim3 next_tid;
  void increment_xyz(dim3 &next, dim3 bound) {
    next.x++;
    if (next.x >= bound.x) {
      next.x = 0;
      next.y++;
      if (next.y >= bound.y) {
        next.y = 0;
        if (next.z < bound.z) {
          next.z++;
        }
      }
    }
  }

 public:
  KernelInfoType(uint64_t code_addr, dim3 blockDim, dim3 gridDim,
                 size_t sharedMem, u_int32_t code_size, u_int8_t reg_num,
                 std::vector<std::string> *sass_code_text);
  ~KernelInfoType();
  void init();
  // void issue_block(SM *sm); // allocate thread and cta to SM
  uint64_t code_address;
  uint32_t code_size;
  dim3 grid_dim;
  dim3 block_dim;
  std::vector<std::string> *sass_code_text;

  unsigned int smem_size;
  unsigned int regs_size;

  bool more_threads_in_cta();
  bool more_cta_in_kernel();
  unsigned int threads_per_cta() const {
    return block_dim.x * block_dim.y * block_dim.z;
  }
  unsigned int num_blocks() const {
    return grid_dim.x * grid_dim.y * grid_dim.z;
  }
  void dim_reset(dim3 &id) {
    id.x = 0;
    id.y = 0;
    id.z = 0;
  }

  void increment_ctaid() {
    increment_xyz(ctaid, grid_dim);
    dim_reset(tid);
  }
  void increment_tid() { increment_xyz(tid, block_dim); }

  dim3 get_cta_id() { return ctaid; }
  dim3 get_tid() { return tid; }
};