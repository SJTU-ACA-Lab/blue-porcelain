#pragma once

#include <string>
#include <sstream>

#include <cstdlib>
#include <stdio.h>
#include "types.h"

namespace gpgpu {

class ArchConfig {
 private:
  uint16_t num_cores_;
  uint16_t num_warps_;
  uint16_t num_threads_;
  uint16_t wsize_;
  // uint16_t num_regs_;
  // uint16_t num_barriers_;

 public:
  ArchConfig(uint16_t num_cores, uint16_t num_warps, uint16_t num_threads)
      : num_cores_(num_cores),
        num_warps_(num_warps),
        num_threads_(num_threads) {}

  uint16_t num_threads() const { return num_threads_; }

  uint16_t num_warps() const { return num_warps_; }

  uint16_t num_cores() const { return num_cores_; }
};

}  // namespace gpgpu