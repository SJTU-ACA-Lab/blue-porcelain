#pragma once

#include <vector>

#include "core.h"
#include "kernel.h"

#define gs_min2(a, b) (((a) < (b)) ? (a) : (b))

namespace gpgpu {
class ThreadEngine {
 private:
  /* data */
  std::list<KernelInfoType *> kernel_list_;
  std::vector<Core *> *sm_cores_;
  ArchConfig arch_;
  bool running_;

 public:
  void issue_block(Core *select_sm, KernelInfoType *kernel);
  void attach_sm(std::vector<Core *> *sm_cores);
  void push_command(void *cmd_q, int size);
  void run();
  void reset();
  bool more_kernel_in_gpu();
  KernelInfoType *current_kernel();
  unsigned int max_cta(KernelInfoType *kernel_info);
  bool running();
  ThreadEngine(const ArchConfig &arch);
  ~ThreadEngine();
};

}  // namespace gpgpu
