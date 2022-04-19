#include "kernel.h"

#include <stdint.h>

#include <vector>

#include "vector_types.h"

void KernelInfoType::init() {
  // CTA ID in Grid
  dim_reset(ctaid);
  // thread id in CTA
  dim_reset(tid);
  // dim_reset(next_ctaid);
  // dim_reset(next_tid);
};
KernelInfoType::KernelInfoType(uint64_t code_addr, dim3 blockDim, dim3 gridDim,
                               size_t sharedMem, u_int32_t code_size,
                               u_int8_t reg_num,
                               std::vector<std::string> *sass_code_text)
    : code_address(code_addr),
      grid_dim(gridDim),
      block_dim(blockDim),
      sass_code_text(sass_code_text),
      code_size(code_size),
      smem_size(sharedMem),
      regs_size(reg_num) {
  this->init();
}

KernelInfoType::~KernelInfoType() {}
bool KernelInfoType::more_cta_in_kernel() {
  return ctaid.z < grid_dim.z && ctaid.y < grid_dim.y && ctaid.x < grid_dim.x;
}
bool KernelInfoType::more_threads_in_cta() {
  return tid.z < block_dim.z && tid.y < block_dim.y && tid.x < block_dim.x;
};