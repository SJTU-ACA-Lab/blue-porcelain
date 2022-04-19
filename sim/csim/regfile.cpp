#include "regfile.h"

#include "debug.h"
#include "types.h"
using namespace gpgpu;

//(warpid, regnum)->(bank, row)
void RegFile::init(KernelInfoType** kernel_uid) { m_kernel_uid = kernel_uid; }

RegFile::regIndex RegFile::register_bank(int wid, int reg_num) {
  unsigned regSize =
      (((*m_kernel_uid)->regs_size) / REGFILE_BANK) * REGFILE_BANK;
  if (((*m_kernel_uid)->regs_size) % REGFILE_BANK != 0)
    regSize = (((*m_kernel_uid)->regs_size) / REGFILE_BANK + 1) * REGFILE_BANK;
  if (reg_num > regSize && reg_num != 255)
    std::cout << "reg_num: " << reg_num << std::endl;
  unsigned bank = reg_num % 4;
  unsigned row = (wid * regSize + reg_num) / REGFILE_BANK;
  return regIndex(bank, row);
}

void RegFile::read(unsigned wid, unsigned regnum,
                   const ThreadMask& thread_active_mask, sass_reg_t* data,
                   int index) {
  if (regnum == 255) {
    for (unsigned i = 0; i < WARP_SIZE; ++i) {
      if (thread_active_mask.test(i)) {
        data[i] = 0;
      }
    }
    return;
  }
  // regIndex reg = m_regTable[wid][regnum];
  regIndex reg = register_bank(wid, regnum);
  for (unsigned i = 0; i < WARP_SIZE; ++i) {
    if (thread_active_mask.test(i)) {
      if (index == 0)
        data[i].u128.lowest = m_register[reg.bank][reg.row_index][i];
      else if (index == 1)
        data[i].u128.low = m_register[reg.bank][reg.row_index][i];
      else if (index == 2)
        data[i].u128.high = m_register[reg.bank][reg.row_index][i];
      else
        data[i].u128.highest = m_register[reg.bank][reg.row_index][i];
    }
  }
}

void RegFile::write(unsigned wid, unsigned regnum,
                    const ThreadMask& thread_active_mask,
                    const sass_reg_t* data, int index) {
  if (regnum == 255) return;
  // regIndex reg = m_regTable[wid][regnum];
  regIndex reg = register_bank(wid, regnum);
  for (unsigned i = 0; i < WARP_SIZE; ++i) {
    if (thread_active_mask.test(i)) {
      if (0 == index)
        m_register[reg.bank][reg.row_index][i] = data[i].u128.lowest;
      else if (1 == index)
        m_register[reg.bank][reg.row_index][i] = data[i].u128.low;
      else if (2 == index)
        m_register[reg.bank][reg.row_index][i] = data[i].u128.high;
      else
        m_register[reg.bank][reg.row_index][i] = data[i].u128.highest;
    }
  }
}

/// pred regiset file///
void PredRegfile::read(unsigned wid, unsigned regnum,
                       const ThreadMask& thread_active_mask, sass_reg_t* data) {
  assert(regnum <= 7);
  if (regnum == 7) {
    for (int i = 0; i < WARP_SIZE; i++) {
      data[i] = sass_reg_t(1);
    }
  } else {
    for (int i = 0; i < WARP_SIZE; ++i)
      if (thread_active_mask.test(i)) data[i] = predicate_map_[wid][regnum][i];
  }
}
void PredRegfile::write(unsigned wid, unsigned regnum,
                        const ThreadMask& thread_active_mask,
                        const sass_reg_t* data) {
  assert(regnum <= 7);
  if (regnum == 7)
    return;
  else {
    if (predicate_map_[wid].find(regnum) == predicate_map_[wid].end()) {
      predicate_map_[wid][regnum] = std::vector<sass_reg_t>(WARP_SIZE, 0);
    }
    for (int i = 0; i < WARP_SIZE; i++) {
      if (thread_active_mask.test(i)) predicate_map_[wid][regnum][i] = data[i];
    }
  }
}

u_int64_t BarrierRegfile::getRpc(unsigned wid, unsigned regnum) {
  return barrier_map_[wid][regnum].rpc;
}
ThreadMask BarrierRegfile::getParticipationMask(unsigned wid, unsigned regnum) {
  return barrier_map_[wid][regnum].participation_mask;
}
ThreadMask BarrierRegfile::getJoinedMask(unsigned wid, unsigned regnum) {
  return barrier_map_[wid][regnum].joined_mask;
}
void BarrierRegfile::setRpc(unsigned wid, unsigned regnum, u_int64_t rpc) {
  barrier_map_[wid][regnum].rpc = rpc;
}
void BarrierRegfile::setParticipationMask(unsigned wid, unsigned regnum,
                                          ThreadMask mask) {
  if (barrier_map_[wid].find(regnum) == barrier_map_[wid].end()) {
    barrier_map_[wid][regnum] = ConvergenceBarrierReg();
  }
  barrier_map_[wid][regnum].participation_mask = mask;
}
void BarrierRegfile::setJoinedMask(unsigned wid, unsigned regnum,
                                   ThreadMask mask) {
  barrier_map_[wid][regnum].joined_mask = mask;
}