#include "threadengine.h"

#include <fstream>
#include <vector>

#include "core.h"
#include "kernel.h"

using namespace gpgpu;

ThreadEngine::ThreadEngine(const ArchConfig &arch) : arch_(arch) { reset(); }

ThreadEngine::~ThreadEngine() {}
void ThreadEngine::reset() {
  running_ = false;
  sm_cores_ = nullptr;
  kernel_list_.clear();
}

void ThreadEngine::attach_sm(std::vector<Core *> *sm_cores) {
  sm_cores_ = sm_cores;
}

void ThreadEngine::push_command(void *cmd_q, int size) {
  struct command_list {
    uint64_t code_address;
    uint32_t code_size;
    dim3 grid_dim;
    dim3 block_dim;
    uint32_t smem_size;
    uint32_t regs_size;
    std::vector<std::string> *sass_code_text;  // 8B

    void clear() {
      smem_size = 0;
      regs_size = 0;
      sass_code_text = nullptr;
    }
  } __attribute__((__packed__));  // total 52B

  command_list cmd_l = *((command_list *)cmd_q);
  KernelInfoType *kernel = new KernelInfoType(
      cmd_l.code_address, cmd_l.block_dim, cmd_l.grid_dim, cmd_l.smem_size,
      cmd_l.code_size, cmd_l.regs_size, cmd_l.sass_code_text);
  kernel_list_.push_back(kernel);
}

KernelInfoType *ThreadEngine::current_kernel() {
  KernelInfoType *kernel = nullptr;
  while (more_kernel_in_gpu()) {
    // select the front kernel and check it
    // kernel_list: do something
    kernel = kernel_list_.front();
    if (kernel->more_cta_in_kernel()) {
      return kernel;
    } else {
      // delete kernel;
      kernel = nullptr;
      kernel_list_.pop_front();
    }
  }

  return kernel;
}
bool ThreadEngine::more_kernel_in_gpu() { return kernel_list_.size() > 0; }

// code from gpgpusim
unsigned int ThreadEngine::max_cta(KernelInfoType *kernel_info) {
  // access hardware parameter
  unsigned int threads_per_cta = kernel_info->threads_per_cta();
  unsigned int threads_per_sm = MAX_THREAD_PER_SM;
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = WARP_SIZE;

  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  // Limit by n_threads/shader
  unsigned int result_thread = threads_per_sm / padded_cta_size;

  // Limit by shmem/shader
  unsigned int result_shmem = (unsigned)-1;
  if (kernel_info->smem_size > 0)
    result_shmem = GPGPU_SHMEM_SIZE / kernel_info->smem_size;

  // Limit by register count, rounded up to multiple of 4.
  unsigned int result_regs = (unsigned)-1;
  if (kernel_info->regs_size > 0)
    result_regs = GPGPU_SM_REGISTERS /
                  (padded_cta_size * ((kernel_info->regs_size + 3) & ~3));

  // Limit by CTA
  unsigned int result_cta = MAX_CTA_PER_SM;

  unsigned result = result_thread;
  result = gs_min2(result, result_shmem);
  result = gs_min2(result, result_regs);
  result = gs_min2(result, result_cta);

  static const struct KernelInfoType *last_kinfo = NULL;
  if (last_kinfo !=
      kernel_info) {  // Only print out stats if kernel_info struct changes
    last_kinfo = kernel_info;
  }

  // gpu_max_cta_per_shader is limited by number of CTAs if not enough to keep
  // all cores busy
  if (kernel_info->num_blocks() < result * arch_.num_cores()) {
    result = kernel_info->num_blocks() / arch_.num_cores();
    if (kernel_info->num_blocks() % arch_.num_cores()) result++;
  }

  assert(result <= MAX_CTA_PER_SM);
  if (result < 1) {
    printf(
        "uArch: ERROR ** Kernel requires more resources than "
        "shader "
        "has.\n");
    abort();
  }
  return result;
}
// end code from gpgpusim

void ThreadEngine::issue_block(Core *select_sm, KernelInfoType *select_kernel) {
  int cta_size = select_kernel->threads_per_cta();
  int padded_cta_size = cta_size % WARP_SIZE
                            ? ((cta_size / WARP_SIZE) + 1) * (WARP_SIZE)
                            : cta_size;

  int warp_num_in_cta = padded_cta_size / WARP_SIZE;

  dim3 cta_id = select_kernel->get_cta_id();
  int free_cta_id = select_sm->issue_free_cta_id();
  int warp_hw_base_id = free_cta_id * warp_num_in_cta;

  select_sm->issue_cta_in_sm(cta_id, select_kernel->code_address);

  // redundance code
  WarpInfoType *warp_info = select_sm->get_warp_infos();
  for (int wid = 0; wid < warp_num_in_cta; wid++) {
    int hw_warp_id = warp_hw_base_id + wid;
    for (int lane_id = 0; lane_id < WARP_SIZE; lane_id++) {
      int thread_idx = wid * WARP_SIZE + lane_id;
      if (thread_idx >= cta_size) {
        break;
      }
      warp_info[hw_warp_id].tid[lane_id] = select_kernel->get_tid();
      select_kernel->increment_tid();
    }
  }
  // end redundance code;
  select_kernel->increment_ctaid();
}

void ThreadEngine::run() {
  KernelInfoType *select_kernel = current_kernel();

  if (select_kernel != nullptr) {
    for (auto select_core : *sm_cores_) {
      if (select_core->issue_kernel_ready()) {
        dim3 blockDim = select_kernel->block_dim;
        dim3 gridDim = select_kernel->grid_dim;
        // std::cout << "ThreadEgine: allocated the kernel to Core. Grid:" <<
        // gridDim << "Block:" << blockDim << std::endl;
        int cta_size = select_kernel->threads_per_cta();
        int padded_cta_size = cta_size % WARP_SIZE
                                  ? ((cta_size / WARP_SIZE) + 1) * (WARP_SIZE)
                                  : cta_size;
        // int cta_num_in_kernel = select_kernel->num_blocks();
        unsigned int warp_num_in_cta = padded_cta_size / WARP_SIZE;
        unsigned int cta_num_in_sm = max_cta(select_kernel);
        select_core->issue_init_in_sm(cta_size, cta_num_in_sm, warp_num_in_cta,
                                      gridDim, blockDim, select_kernel);
      }
    }

    for (auto select_core : *sm_cores_) {
      // A SM only is allocated a kernel;
      if (select_kernel != select_core->allocated_kernel_uid()) continue;
      if (select_core->issue_cta_ready() &&
          select_kernel->more_cta_in_kernel()) {
        issue_block(select_core, select_kernel);
      }
    }
    running_ = true;
  } else {
    running_ = false;
  }
}

bool ThreadEngine::running() { return running_; }