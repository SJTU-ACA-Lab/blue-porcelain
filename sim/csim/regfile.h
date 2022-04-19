#pragma once
#include <map>
#include <queue>

#include "GPU_config.h"
#include "types.h"
namespace gpgpu {

class RegFile {
 public:
  struct regIndex {
    unsigned bank;
    unsigned row_index;
    regIndex(unsigned b, unsigned r) : bank(b), row_index(r) {}
  };
  regIndex register_bank(int wid, int reg_num);
  void init(KernelInfoType** kernel_uid);

  void read(unsigned wid, unsigned regnum, const ThreadMask& thread_active_mask,
            sass_reg_t* data, int index);
  void write(unsigned wid, unsigned regnum,
             const ThreadMask& thread_active_mask, const sass_reg_t* data,
             int index);

 private:
  unsigned int m_register[REGFILE_BANK]
                         [GPGPU_SM_REGISTERS / (REGFILE_BANK * WARP_SIZE)]
                         [WARP_SIZE] = {{{0}}};
  KernelInfoType** m_kernel_uid;
};

class PredRegfile {
 public:
  void read(unsigned wid, unsigned regnum, const ThreadMask& thread_active_mask,
            sass_reg_t* data);
  void write(unsigned wid, unsigned regnum,
             const ThreadMask& thread_active_mask, const sass_reg_t* data);

 private:
  std::map<unsigned, std::vector<sass_reg_t>> predicate_map_[WARP_NUM];
};

class BarrierRegfile {
 public:
  u_int64_t getRpc(unsigned wid, unsigned regnum);
  ThreadMask getParticipationMask(unsigned wid, unsigned regnum);
  ThreadMask getJoinedMask(unsigned wid, unsigned regnum);
  void setRpc(unsigned wid, unsigned regnum, u_int64_t rpc);
  void setParticipationMask(unsigned wid, unsigned regnum,
                            ThreadMask mask = ThreadMask(0));
  void setJoinedMask(unsigned wid, unsigned regnum,
                     ThreadMask mask = ThreadMask(0));

 private:
  struct ConvergenceBarrierReg {
    u_int64_t rpc;
    ThreadMask participation_mask;
    ThreadMask joined_mask;
  };
  std::map<unsigned, ConvergenceBarrierReg> barrier_map_[WARP_NUM];
};
}  // namespace gpgpu