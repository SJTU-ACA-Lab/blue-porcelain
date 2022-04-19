#include "processor.h"

using namespace gpgpu;
using namespace std::chrono;

Processor::Processor(const ArchConfig& arch)
    : cores_(arch.num_cores()), thread_engine_(arch) {
  uint32_t num_cores = arch.num_cores();

  // create cores
  for (uint32_t i = 0; i < num_cores; ++i) {
    cores_.at(i) = new Core(arch, i);
  }
}

Processor::~Processor() {}

void Processor::init() {
  for (auto core : cores_) {
    core->init();
  }
}

void Processor::push_command(void* cmd_q, int size) {
  thread_engine_.push_command(cmd_q, size);
}

void Processor::connect_ram(RAM* mem) {
  thread_engine_.attach_sm(&cores_);
  for (auto core : cores_) {
    core->connect_mem(mem);
  }
}

int Processor::run() {
  bool running;
  int exitcode = 0;
  auto exec_start = high_resolution_clock::now();
  do {
    thread_engine_.run();
    running = false;
    for (auto& core : cores_) {
      core->cycle();
      if (core->running() || thread_engine_.running()) {
        running = true;
      }
    }
  } while (running);
  auto exec_end = high_resolution_clock::now();
  auto exec_duration = duration_cast<milliseconds>(exec_end - exec_start);
  // std::cout << "exec duration: " << std::dec << exec_duration.count() << "
  // ms"
  //           << std::endl;
  return exitcode;
}