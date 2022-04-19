#pragma once
#include "core.h"
#include "kernel.h"
#include "threadengine.h"
#include <chrono>

namespace gpgpu {

class ArchConfig;
class RAM;

class Processor {
 public:
  Processor(const ArchConfig &arch);
  ~Processor();

  void init();
  void connect_ram(RAM *mem);
  void push_command(void *cmd_q, int size);
  int run();

 private:
  std::vector<Core *> cores_;
  ThreadEngine thread_engine_;
};

}  // namespace gpgpu
