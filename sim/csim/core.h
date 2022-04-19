#pragma once

#include <bitset>
#include <list>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "archdef.h"
#include "debug.h"
#include "mem.h"
#include "types.h"
#include "kernel.h"
#include "regfile.h"
#include "simt_stack.h"
#include "vector_types.h"
namespace gpgpu {
class Instruction;

class Core {
  friend class Instruction;
  friend class S2RInstr;
  friend class LDGInstr;
  friend class STGInstr;
  friend class EXITInstr;
  friend class BRAInstr;
  friend class BREAKInstr;
  friend class BSSYInstr;
  friend class BSYNCInstr;
  friend class LDSInstr;
  friend class STSInstr;
  friend class BARInstr;
  friend class RETInstr;
  friend class CALRELInstr;
  friend class LDCInstr;

 public:
  Core(const ArchConfig &arch, uint32_t id);
  ~Core();

  void init();
  void connect_mem(RAM *ram);

  bool running() const;

  void reset();

  void cycle();

  uint32_t id() const { return sid_; }

  const ArchConfig &arch() const { return arch_; }

  WarpInfoType *get_warp_infos() { return warp_info; }

  bool issue_cta_ready();
  void issue_cta_in_sm(dim3 cta_id, uint64_t code_address);
  int issue_free_cta_id();
  bool issue_kernel_ready() { return kernel_uid == nullptr; }
  KernelInfoType *allocated_kernel_uid() { return kernel_uid; }
  void issue_init_in_sm(uint32_t cta_size, uint32_t cta_num_in_sm,
                        uint32_t warp_num_in_cta, dim3 grid_dim, dim3 cta_dim,
                        KernelInfoType *kernel_uid);

 private:
  uint32_t sid_;
  const ArchConfig arch_;
  uint32_t cta_size;
  uint32_t cta_num_in_sm;
  uint32_t warp_num_in_cta;

  WarpInfoMask warp_active;
  WarpInfoType warp_info[WARP_NUM];
  WarpInfoMask warp_exit;
  WarpInfoMask bar_stall{0x0};

  dim3 cta_id_list[MAX_CTA_PER_SM];
  dim3 grid_dim;
  dim3 cta_dim;

  std::bitset<MAX_CTA_PER_SM> cta_vld;
  int cta_warp_left[MAX_CTA_PER_SM];
  int warp_thread_left[WARP_NUM];
  std::vector<int> cta_threads;

  int last_warp_fetch{-1};

  SimtStack simt_stack_unit[WARP_NUM];
  RAM *mem_;
  RegFile registerfile;
  PredRegfile pred_reg_file_;
  BarrierRegfile barrier_reg_file_;

  KernelInfoType *kernel_uid;  // kernel allocated;

  bool running_;

  Instruction *FetchNextWarpInstruction();
  void thread_cycle();
};

}  // namespace gpgpu
