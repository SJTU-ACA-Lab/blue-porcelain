// This file created from vortex.cpp distributed with Vortex v0.2.3
// Changes Copyright 2022, ACA Lab of SJTU

/**
 * gpgpu.cpp
 *
 * Copyright (c) <2022>, <Advanced Computer Architecture Laboratory of SJTU>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of Advanced Computer Architecture Laboratory of SJTU
 *      nor the names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior written
 *      permission.
 *    * This version is distributed freely for non-commercial use only.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Copyright (c) <2020>, <Georgia Institute of Technology>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *    * Neither the name of the Georgia Institute of Technology nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <util.h>
#include <gpgpu.h>
#include <GPU_config.h>
#include <assert.h>
#include <mem.h>
#include <processor.h>
#include <chrono>
#include <future>
#include <iostream>

#include "archdef.h"

using namespace gpgpu;

///////////////////////////////////////////////////////////////////////////////

class gpgpu_device {
 public:
  gpgpu_device()
      : arch_(NUM_CORES, WARP_NUM, WARP_SIZE),
        ram_(RAM_PAGE_SIZE),
        processor_(arch_) {
    // attach memory module
    processor_.connect_ram(&ram_);
    processor_.init();
  }

  ~gpgpu_device() {
    if (future_.valid()) {
      future_.wait();
    }
  }

  void push_command(void* cmd_q, int size) {
    processor_.push_command(cmd_q, size);
  }

  int upload(const void* src, uint64_t dest_addr, uint64_t size,
             uint64_t src_offset) {
    ram_.write((const uint8_t*)src + src_offset, dest_addr, size);

    return 0;
  }

  int download(void* dest, uint64_t src_addr, uint64_t size,
               uint64_t dest_offset) {
    ram_.read((uint8_t*)dest + dest_offset, src_addr, size);

    return 0;
  }

  int start() {
    // ensure prior run completed
    if (future_.valid()) {
      future_.wait();
    }

    // start new run
    future_ = std::async(std::launch::async, [&] { processor_.run(); });

    return 0;
  }

  int wait(uint64_t timeout) {
    if (!future_.valid()) return 0;
    uint64_t timeout_sec = timeout / 1000;
    std::chrono::seconds wait_time(1);
    for (;;) {
      // wait for 1 sec and check status
      auto status = future_.wait_for(wait_time);
      if (status == std::future_status::ready || 0 == timeout_sec--) break;
    }
    return 0;
  }

 private:
  ArchConfig arch_;
  RAM ram_;
  Processor processor_;
  std::future<void> future_;
};

///////////////////////////////////////////////////////////////////////////////

extern int gpgpu_dev_open(gpgpu_device_h* hdevice) {
  if (nullptr == hdevice) return -1;

  *hdevice = new gpgpu_device();

  return 0;
}

extern int gpgpu_dev_close(gpgpu_device_h hdevice) {
  if (nullptr == hdevice) return -1;

  gpgpu_device* device = ((gpgpu_device*)hdevice);

  delete device;

  return 0;
}

extern int gpgpu_copy_to_dev(gpgpu_device_h hdevice, void* data,
                             uint64_t dev_maddr, uint64_t size,
                             uint64_t src_offset) {
  if (size <= 0) return -1;

  gpgpu_device* device = (gpgpu_device*)hdevice;
  return device->upload(data, dev_maddr, size, src_offset);
}

extern int gpgpu_copy_from_dev(gpgpu_device_h hdevice, void* data,
                               uint64_t dev_maddr, uint64_t size,
                               uint64_t dest_offset) {
  if (size <= 0) return -1;
  gpgpu_device* device = (gpgpu_device*)hdevice;
  return device->download(data, dev_maddr, size, dest_offset);
}

int gpgpu_push_command(gpgpu_device_h hdevice, void* cmd_q, int size) {
  if (nullptr == hdevice) return -1;
  gpgpu_device* device = ((gpgpu_device*)hdevice);

  device->push_command(cmd_q, size);
  return 0;
}

extern int gpgpu_start(gpgpu_device_h hdevice) {
  if (nullptr == hdevice) return -1;

  gpgpu_device* device = ((gpgpu_device*)hdevice);

  return device->start();
}

extern int gpgpu_ready_wait(gpgpu_device_h hdevice, uint64_t timeout) {
  if (nullptr == hdevice) return -1;

  gpgpu_device* device = ((gpgpu_device*)hdevice);

  return device->wait(timeout);
}
