#include <assert.h>
#include <string.h>
#include <util.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA Include
#include "archdef.h"
#include "bitmanip.h"
#include "core.h"
#include "kernel.h"
#include "mem.h"
#include "stdio.h"
#include "types.h"
#include "vector_types.h"

#include "instr_ctrl.h"
#include "instr_fp.h"
#include "instr_int.h"
#include "instr_ldst.h"
#include "instr_miscell.h"
#include "instr_mov.h"
#include "instr_pred.h"
// #define FUN_SIM 1

using namespace gpgpu;

// kernel info -> blockdim and griddim in kernel
Core::Core(const ArchConfig& arch, uint32_t sid) : sid_(sid), arch_(arch) {
  this->reset();
}

Core::~Core() {}

void Core::init() { registerfile.init(&kernel_uid); }

void Core::reset() {
  // threads reset;
  cta_vld.reset();
  cta_dim = dim3(0, 0, 0);
  cta_size = 0;
  cta_num_in_sm = 0;
  warp_num_in_cta = 0;
  kernel_uid = nullptr;
  warp_active.reset();
  warp_exit.reset();
  // icache_.clear();
  running_ = false;
}

void Core::connect_mem(RAM* ram) {
  // bind RAM to memory unit
  mem_ = ram;
}

void Core::cycle() {
  if ((cta_vld.none())) {
    running_ = false;
  } else {
    running_ = true;
  }

  Instruction* nextwarp = FetchNextWarpInstruction();
  if (nextwarp != nullptr) {
    nextwarp->decode();
    nextwarp->readOperand();
    nextwarp->execute();
    nextwarp->writeBack();
    delete nextwarp;
    nextwarp = nullptr;
  }
  thread_cycle();
}

bool Core::running() const {
  // bool is_running = true; //(committed_instrs_ != issued_instrs_);
  // return is_running;
  return running_;
}

void Core::issue_init_in_sm(uint32_t cta_size, uint32_t cta_num_in_sm,
                            uint32_t warp_num_in_cta, dim3 grid_dim,
                            dim3 cta_dim, KernelInfoType* kernel_uid) {
  assert(this->issue_kernel_ready());
  this->cta_size = cta_size;
  this->cta_num_in_sm = cta_num_in_sm;
  this->warp_num_in_cta = warp_num_in_cta;
  this->grid_dim = grid_dim;
  this->cta_dim = cta_dim;
  this->kernel_uid = kernel_uid;
  this->cta_threads.clear();
  this->cta_threads.resize(cta_num_in_sm, 0);
}

bool Core::issue_cta_ready() {
  bool status = cta_vld.count() < cta_num_in_sm;
  return status;
}

int Core::issue_free_cta_id() {
  int free_cta_hw_id = -1;
  assert(cta_vld.count() < cta_num_in_sm);

  for (int i = 0; i < MAX_CTA_PER_SM; ++i) {
    if (!cta_vld.test(i)) {
      free_cta_hw_id = i;
      break;
    }
  }
  return free_cta_hw_id;
}
void Core::thread_cycle() {
  if (bar_stall.any()) {
    for (int i = 0; i < cta_num_in_sm; i++) {
      // std::cout << std::dec << cta_threads[i] << std::endl;
      if (cta_threads[i] == 0) {
        for (int j = i * warp_num_in_cta; j < (i + 1) * warp_num_in_cta; j++) {
          bar_stall.reset(j);
        }
        for (int j = i * warp_num_in_cta; j < (i + 1) * warp_num_in_cta; j++) {
          cta_threads[i] += warp_info[j].thread_active.count();
        }
      }
    }
  }

  WarpInfoMask judgment = warp_active & warp_exit;
  if (judgment.none()) return;
  for (int i = 0; i < WARP_NUM; i++) {
    if (judgment.test(i)) {
      warp_active[i] = 0;
      warp_exit[i] = 0;
      uint32_t cta_idx = i / warp_num_in_cta;
      cta_warp_left[cta_idx]--;
      // cta done
      if (cta_warp_left[cta_idx] == 0) {
        cta_vld[cta_idx] = 0;
      }
      // kernel done
      if (cta_vld.none()) {
        this->reset();
      }
    }
  }
}

void Core::issue_cta_in_sm(dim3 cta_id, uint64_t code_address) {
  int cta_hw_id = -1;
  assert(cta_vld.count() < cta_num_in_sm);

  for (int i = 0; i < MAX_CTA_PER_SM; ++i) {
    if (!cta_vld.test(i)) {
      cta_hw_id = i;
      break;
    }
  }
  //  cta_hw_id指的是硬件id，而非kernel中的ctaid
  cta_vld.set(cta_hw_id);
  cta_id_list[cta_hw_id] = cta_id;
  uint32_t warp_base = cta_hw_id * warp_num_in_cta;
  // cta_threads.clear();
  // cta_threads.resize(cta_size, 0);
  for (uint32_t i = 0; i < warp_num_in_cta; ++i) {
    uint32_t warp_id = warp_base + i;
    warp_active.set(warp_id);
    cta_warp_left[cta_hw_id]++;
    warp_info[warp_id].wid_in_cta = i;
    warp_info[warp_id].ctaid = cta_id;
    warp_info[warp_id].pc = code_address;
    // ibuffer_bra_pending.reset(warp_id);
    // fetch_unit.fetch_pending.reset(warp_id);
    // fetch_unit.send_req.reset(warp_id);
    bar_stall.reset(warp_id);
    // configure thread_active_mask
    for (int lane_id = 0; lane_id < WARP_SIZE; ++lane_id) {
      u_int32_t tid_1d = i * WARP_SIZE + lane_id;
      if (tid_1d < cta_size) {
        warp_info[warp_id].thread_active.set();
        warp_thread_left[warp_id]++;
      } else {
        warp_info[warp_id].thread_active.reset(lane_id);
      }
    }
    cta_threads[warp_id / warp_num_in_cta] +=
        warp_info[warp_id].thread_active.count();
  }
}

Instruction* Core::FetchNextWarpInstruction() {
  //选择一个warp
  std::bitset<WARP_NUM> judgement;
  judgement = warp_active & (~(bar_stall));  //&
  // (~fetch_pending);  // warp选取判断

  if (!(judgement.any())) {
    return nullptr;
  }
  for (int i = 0; i < WARP_NUM; i++) {
    int pos = (last_warp_fetch + 1 + i) % WARP_NUM;
    if (judgement[pos] == 1) {
      last_warp_fetch = pos;
      break;
    }
  }
  uint32_t warp_id = last_warp_fetch;
  uint64_t pc = warp_info[warp_id].pc;
  warp_info[warp_id].pc += 0x10;

  SassCodeType sass;
  mem_->read(&sass.bin, pc, sizeof(SassCodeType));

  uint32_t opcode = sass.b64.L & opcode_mask;
  Instruction* ptr = nullptr;
  switch (opcode) {
    case MOV:
      ptr = new MovInstr(*this, pc, warp_id, sass);
      break;
    case SHFL:
      ptr = new SHFLInstr(*this, pc, warp_id, sass);
      break;
    case S2R:
      ptr = new S2RInstr(*this, pc, warp_id, sass);
      break;
    case IMAD:
      ptr = new IMADInstr(*this, pc, warp_id, sass);
      break;
    case IMAD_WIDE:
      ptr = new IMADWIDEInstr(*this, pc, warp_id, sass);
      break;
    case ISETP:
      ptr = new ISETPInstr(*this, pc, warp_id, sass);
      break;
    case SHF:
      ptr = new SHFInstr(*this, pc, warp_id, sass);
      break;
    case LOP3:
      ptr = new LOP3Instr(*this, pc, warp_id, sass);
      break;
    case LEA:
      ptr = new LEAInstr(*this, pc, warp_id, sass);
      break;
    case LDG:
      ptr = new LDGInstr(*this, pc, warp_id, sass);
      break;
    case STG:
      ptr = new STGInstr(*this, pc, warp_id, sass);
      break;
    case LDS:
      ptr = new LDSInstr(*this, pc, warp_id, sass);
      break;
    case STS:
      ptr = new STSInstr(*this, pc, warp_id, sass);
      break;
    case IADD3:
      ptr = new IADD3Instr(*this, pc, warp_id, sass);
      break;
    case EXIT:
      ptr = new EXITInstr(*this, pc, warp_id, sass);
      break;
    case BRA:
      ptr = new BRAInstr(*this, pc, warp_id, sass);
      break;
    case BREAK:
      ptr = new BREAKInstr(*this, pc, warp_id, sass);
      break;
    case BSSY:
      ptr = new BSSYInstr(*this, pc, warp_id, sass);
      break;
    case BSYNC:
      ptr = new BSYNCInstr(*this, pc, warp_id, sass);
      break;
    case BAR:
      ptr = new BARInstr(*this, pc, warp_id, sass);
      break;
    case FADD:
      ptr = new FADDInstr(*this, pc, warp_id, sass);
      break;
    case FFMA:
      ptr = new FFMAInstr(*this, pc, warp_id, sass);
      break;
    case FMUL:
      ptr = new FMULInstr(*this, pc, warp_id, sass);
      break;
    case FCHK:
      ptr = new FCHKInstr(*this, pc, warp_id, sass);
      break;
    case FSETP:
      ptr = new FSETPInstr(*this, pc, warp_id, sass);
      break;
    case MUFU:
      ptr = new MUFUInstr(*this, pc, warp_id, sass);
      break;
    case SEL:
      ptr = new SELInstr(*this, pc, warp_id, sass);
      break;
    case LDC:
      ptr = new LDCInstr(*this, pc, warp_id, sass);
      break;
    case CALL_ABS:
      ptr = new CALABSInstr(*this, pc, warp_id, sass);
      break;
    case CALL_REL:
      ptr = new CALRELInstr(*this, pc, warp_id, sass);
      break;
    case RET:
      ptr = new RETInstr(*this, pc, warp_id, sass);
      break;
    case PLOP3:
      ptr = new PLOP3Instr(*this, pc, warp_id, sass);
    case NOPI:
    case BMOV:
      break;
    default:
      std::cout << "unimplemented instruction:" << std::hex << pc - STARTUP_ADDR
                << std::endl;
  }
  return ptr;
}