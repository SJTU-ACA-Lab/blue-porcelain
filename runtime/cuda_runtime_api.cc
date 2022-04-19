// This file created from cuda_runtime_api.cc distributed with GPGPU-Sim 4.0
// Changes Copyright 2022, ACA Lab of SJTU

/**
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

/*
 * cuda_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
 * George L. Yuan and the University of British Columbia, Vancouver,
 * BC V6T 1Z4, All Rights Reserved.
 *
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and
 * benchmarks/template/ are derived from the CUDA SDK available from
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from
 * src/intersim/ are derived from Booksim (a simulator provided with the
 * textbook "Principles and Practices of Interconnection Networks" available
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by
 * the corresponding legal terms and conditions set forth separately (original
 * copyright notices are left in files from these sources and where we have
 * modified a file our copyright notice appears before the original copyright
 * notice).
 *
 * Using this version of GPGPU-Sim requires a complete installation of CUDA
 * which is distributed seperately by NVIDIA under separate terms and
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use
 * only.
 *
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 *
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung,
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia,
 * Vancouver, BC V6T 1Z4
 */

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <unistd.h>
#include <list>
#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h>  // Apple's version of GLUT is here
#else
#include <GL/gl.h>
#endif
#endif

#define __CUDA_RUNTIME_API_H__
// clang-format off
#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "cuda_api.h"
#include "cudaProfiler.h"
// clang-format on
#include "gpgpu.h"

#include <pthread.h>
#include <semaphore.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#ifndef OPENGL_SUPPORT
typedef unsigned long GLuint;
#endif

#ifndef CUDART_VERSION
#define CUDART_VERSION 10200
#endif

#ifndef RT_TRACE
#define RT_TRACE 0
#endif

int g_debug_execution = 0;
std::string debug_info = "gpgpu_trace.log";

#define ELFFILE "_cuobjdump_sass.elf"

struct deviceInfo {
  char* kernel_name;
  int offset = -1;
  int size = 0;
  int reg_num = 0;
  int const2_start_pos = -1;
  int const2_size = 0;
  int shared_memory_size = 0;
  int local_mem_size = 0;
  std::string elf_file_path_name;
  std::string app_binary;
};

std::map<const char*, deviceInfo> host_device_map;
// need to track the size allocated so that cudaHostGetDevicePointer() can
// function properly.
std::unordered_map<void*, void**> pinned_memory;
std::unordered_map<void*, size_t> pinned_memory_size;
cudaDeviceProp prop_;

std::map<std::string, struct cubin_info>
    cubin_info_table;  // key: cubin_filename

struct cubin_info {
  std::string elf_filename;
};

std::map<std::string, std::string>
    func_cubin_map;  // function name -> cubin file name containing the function

struct command_list {
  uint64_t code_address;
  uint32_t code_size;
  dim3 grid_dim;
  dim3 block_dim;
  uint32_t smem_size;
  uint32_t regs_size;
  std::vector<std::string>* sass_code_text;  // 8B

  void clear() {
    smem_size = 0;
    regs_size = 0;
    sass_code_text = nullptr;
  }
} __attribute__((__packed__)) cmd_l;  // total 52B

#include "GPU_config.h"

/*DEVICE_BUILTIN*/
struct cudaArray {
  void* devPtr;
  int devPtr32;
  struct cudaChannelFormatDesc desc;
  int width;
  int height;
  int size;  // in bytes
  unsigned dimensions;
};

struct CUevent_st {
 public:
  CUevent_st(bool blocking) {
    m_uid = 0;
    m_blocking = blocking;
    m_updates = 0;
    m_wallclock = 0;
    m_gpu_tot_sim_cycle = 0;
    m_issued = 0;
    m_done = false;
  }
  void update(double cycle, time_t clk) {
    m_updates++;
    m_wallclock = clk;
    m_gpu_tot_sim_cycle = cycle;
    m_done = true;
  }
  // void set_done() { assert(!m_done); m_done=true; }
  int get_uid() const { return m_uid; }
  unsigned num_updates() const { return m_updates; }
  bool done() const { return m_updates == m_issued; }
  time_t clock() const { return m_wallclock; }
  void issue() { m_issued++; }
  unsigned int num_issued() const { return m_issued; }

 private:
  int m_uid;
  bool m_blocking;
  bool m_done;
  int m_updates;
  unsigned int m_issued;
  time_t m_wallclock;
  double m_gpu_tot_sim_cycle;

  static int m_next_event_uid;
};

#if !defined(__dv)
#if defined(__cplusplus)
#define __dv(v) = v
#else /* __cplusplus */
#define __dv(v)
#endif /* __cplusplus */
#endif /* !__dv */

cudaError_t g_last_cudaError = cudaSuccess;

unsigned long long m_dev_malloc = GLOBAL_HEAP_START;

#if defined __APPLE__
#define __my_func__ __PRETTY_FUNCTION__
#else
#if defined __cplusplus ? __GNUC_PREREQ(2, 6) : __GNUC_PREREQ(2, 4)
#define __my_func__ __PRETTY_FUNCTION__
#else
#if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#define __my_func__ __func__
#else
#define __my_func__ ((__const char*)0)
#endif
#endif
#endif

void cuda_not_implemented(const char* func, unsigned line) {
  fflush(stdout);
  fflush(stderr);
  printf(
      "\n\nGPGPU: Execution error: CUDA API function \"%s()\" has not "
      "been implemented yet.\n"
      "                 [$GPGPU_ROOT/libcuda/%s around line %u]\n\n\n",
      func, __FILE__, line);
  fflush(stdout);
  abort();
}

void announce_call(const char* func) {
  printf("\n\nGPGPU: CUDA API function \"%s\" has been called.\n", func);
  fflush(stdout);
}

#define gpgpu_error(msg, ...) \
  gpgpu_error_impl(__func__, __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define gpgpu_assert(cond, msg, ...) \
  gpgpu_assert_impl((cond), __func__, __FILE__, __LINE__, msg, ##__VA_ARGS__)

// give the name of application
char* get_app_binary_name(std::string abs_path) {
  char* self_exe_path;

  char* buf = strdup(abs_path.c_str());
  char* token = strtok(buf, "/");
  while (token != NULL) {
    self_exe_path = token;
    token = strtok(NULL, "/");
  }

  self_exe_path = strtok(self_exe_path, ".");
  // printf("self exe links to: %s\n", self_exe_path);
  return self_exe_path;
}

gpgpu_device_h gpgpu_device = nullptr;
int sass_code_size = 0;

// Return the executable file of the process containing the SASS code
std::string get_app_binary(bool print = true) {
  char self_exe_path[1025];
#ifdef __APPLE__
  uint32_t size = sizeof(self_exe_path);
  if (_NSGetExecutablePath(self_exe_path, &size) != 0) {
    printf("GPGPU ** ERROR: _NSGetExecutablePath input buffer too small\n");
    exit(1);
  }
#else
  std::stringstream exec_link;
  exec_link << "/proc/self/exe";

  ssize_t path_length = readlink(exec_link.str().c_str(), self_exe_path, 1024);
  assert(path_length != -1);
  self_exe_path[path_length] = '\0';
#endif
  if (print) {
    printf("self exe links to : %s\n", self_exe_path);
  }

  return self_exe_path;
}

template <typename F>
void cpopen(std::fstream& fs, const char* command, const char* __modes,
            F const& f) {
  FILE* ptr;
  if ((ptr = popen(command, "r")) != NULL) {
    f(ptr);
    pclose(ptr);
  } else {
#if RT_TRACE == 1
    fs << "popen " << command << " error\n";
#endif
    printf("popen error:%s\n", strerror(errno));
    exit(-1);
  }
}

void setupAllArguments(std::string device_func_name, std::string elf_filename,
                       const void** args, gpgpu_device_h device) {
  char command[1000];
  std::fstream fs;
#if RT_TRACE == 1
  fs.open(debug_info, std::ios::app);
#endif

  char buf_ps1[1024] = {0};
  int begin = 0, end = 0;
  snprintf(command, 1000,
           "grep -n .nv.info.%s %s| tail -1 | awk -F ':' '{print $1}'",
           device_func_name.c_str(), elf_filename.c_str());
  cpopen(fs, command, "r", [&](FILE* p) {
    fgets(buf_ps1, sizeof(buf_ps1), p);
    begin = atoi(buf_ps1);
  });

  snprintf(command, 1000,
           "grep -n   '.nv\\|EIATTR_EXIT_INSTR_OFFSETS'   %s |  awk -F ':' "
           "'$1>%d{print $1}' | awk -F ':' 'NR==1{print $1}'",
           elf_filename.c_str(), begin);
  cpopen(fs, command, "r", [&](FILE* p) {
    fgets(buf_ps1, sizeof(buf_ps1), p);
    end = atoi(buf_ps1);
  });

  if (end <= begin) {
    printf("popen error, begin=%d end=%d :\n", begin, end);
    exit(-1);
  }

  char buf_ps[1024];
  std::list<int> line_nums;
  snprintf(command, 1000,
           "sed -n '%d,%dp' %s | grep -n EIATTR_KPARAM_INFO | awk -F ':' "
           "'{print $1}'",
           begin, end, elf_filename.c_str());
  cpopen(fs, command, "r", [&](FILE* p) {
    while (fgets(buf_ps, sizeof(buf_ps), p) != NULL) {
      line_nums.push_front(begin + atoi(buf_ps) + 1);
    }
  });

  std::vector<std::pair<int, int>> param_infos;
  int param_num = line_nums.size();
  for (int i = 0; i < param_num; i++) {
    // get each kernel param's offset and size
    snprintf(command, 1000, "sed -n %dp %s | awk -F ' ' '{print $10, $NF;}'",
             line_nums.front(), elf_filename.c_str());
    line_nums.pop_front();
    int offset, size;
    cpopen(fs, command, "r", [&](FILE* p) {
      while (fgets(buf_ps, sizeof(buf_ps), p) != NULL) {
        sscanf(buf_ps, "%x %x", &offset, &size);
        param_infos.push_back(std::make_pair(offset, size));
      }
    });
  }

  int totalParamSize = param_infos.back().first + param_infos.back().second;

  auto buffer = new uint8_t[totalParamSize];
#if RT_TRACE == 1
  fs << "upload kernel arguments" << std::endl;
  fs.close();
#endif
  std::cout << "upload kernel arguments" << std::endl;

  int offset = 0;
  for (int i = 0; i < param_infos.size(); ++i) {
    offset = param_infos.at(i).first;
    memcpy((void*)buffer + offset, (void*)args[i], param_infos.at(i).second);
  }

  gpgpu_copy_to_dev(device, (void*)buffer, KERNEL_PARAMS_START, totalParamSize,
                    0);
  delete[] buffer;
}

void addKernel(std::string cubin_filename, std::string device_func_name,
               std::string elf_filename, const char* hostFun,
               gpgpu_device_h device) {
  char command[1000];
  std::fstream fs;
  char buf_ps[1024] = {0};
  std::string app_binary = get_app_binary(false);
  // std::string cubin_file_relpath =
  //     "./" + cubin_filename;  // read file need the full path
  snprintf(command, 1000,
           "grep .text.%s %s | grep PROGBITS | awk -F' ' '{print $2, $3}'",
           device_func_name.c_str(), elf_filename.c_str());
  int offset, size;
  cpopen(fs, command, "r", [&](FILE* p) {
    fgets(buf_ps, sizeof(buf_ps), p);
    sscanf(buf_ps, "%x %x", &offset, &size);
  });

  host_device_map[hostFun].offset = offset;
  host_device_map[hostFun].size = size;
  host_device_map[hostFun].elf_file_path_name = cubin_filename;
  host_device_map[hostFun].app_binary = app_binary;
  // printf(
  //   "__cudaRegisterFunction %s : hostFun 0x%p\n",
  //   host_device_map[hostFun].kernel_name , hostFun);

  std::ifstream infile;
  infile.open(host_device_map[hostFun].elf_file_path_name, std::ios::in);
  infile.seekg(host_device_map[hostFun].offset);

  auto buffer = new char[host_device_map[hostFun].size];

  infile.read(buffer, host_device_map[hostFun].size);
#if RT_TRACE == 1
  fs.open(debug_info, std::ios::app);
  fs << "upload program" << std::endl;
  fs.close();
#endif
  std::cout << "upload program" << std::endl;
  gpgpu_copy_to_dev(device, (void*)buffer, STARTUP_ADDR,
                    host_device_map[hostFun].size, 0);

  delete[] buffer;
  infile.close();
}

void addCodeText(std::string cubin_filename, std::string device_func_name,
                 std::vector<std::string>* sass_code_text) {
  char command[1000];
  std::fstream fs;
#if RT_TRACE == 1
  fs.open(debug_info, std::ios::app);
#endif

  char buf_ps[1024] = {0};
  std::string sass_file_path_name =
      "./_cuobjdump_sass_text.sass";  // read file need the full path
  snprintf(command, 1000, "$CUDA_INSTALL_PATH/bin/cuobjdump -sass %s > %s",
           cubin_filename.c_str(), sass_file_path_name.c_str());
  if (system(command) != 0) {
    printf("ERROR: command: %s failed \n", command);
    exit(0);
  }
  snprintf(command, 1000, "grep -n %s %s | awk -F ':' '{print $1}'",
           device_func_name.c_str(), "./_cuobjdump_sass_text.sass");
  int sass_text_begin = 0, sass_text_end = 0;
  cpopen(fs, command, "r", [&](FILE* p) {
    fgets(buf_ps, sizeof(buf_ps), p);
    sass_text_begin = atoi(buf_ps);
  });

  snprintf(command, 1000,
           "sed -n '%d,$ p' %s | grep -n -m 1  \"[.]\\{10\\}\"  | awk -F ':' "
           "'{print $1}'",
           sass_text_begin, "./_cuobjdump_sass_text.sass");
  cpopen(fs, command, "r", [&](FILE* p) {
    fgets(buf_ps, sizeof(buf_ps), p);
    sass_text_end = atoi(buf_ps) + sass_text_begin - 1;
  });

  // add sass code text, read from file and convert to string vector
  snprintf(command, 1000,
           "sed -n '%d,%dp' %s | grep \";\" | awk -F \"/*\" '{print $3}' | awk "
           "'$1=$1'",
           sass_text_begin, sass_text_end, "./_cuobjdump_sass_text.sass");
  cpopen(fs, command, "r", [&](FILE* p) {
    while (fgets(buf_ps, sizeof(buf_ps), p) != NULL) {
      sass_code_text->emplace_back(buf_ps);
    }
  });

  fs.close();
}

void getRegNum(std::string device_func_name, std::string elf_filename,
               const char* hostFun) {
  char command[1000];
  std::fstream fs;

  char buf_ps[1024] = {0};
  int reginfo_linenum = 0;
#if RT_TRACE == 1
  fs.open(debug_info, std::ios::app);
#endif

  snprintf(command, 1000,
           "grep -n .text.%s %s | tail -1 | awk -F ':' '{print $1}'",
           device_func_name.c_str(), elf_filename.c_str());
  cpopen(fs, command, "r", [&](FILE* p) {
    while (fgets(buf_ps, sizeof(buf_ps), p) != NULL) {
      fgets(buf_ps, sizeof(buf_ps), p);
      reginfo_linenum = atoi(buf_ps) + 1;
      memset(buf_ps, '\0', sizeof(buf_ps));
    }
  });

  snprintf(command, 1000,
           "sed -n %dp %s | awk -F '\t' '{print $2}' | awk -F '=' '{print $2}' "
           "| awk '{gsub(/^\\s+|\\s+$/,\"\");print}'",
           reginfo_linenum, elf_filename.c_str());
  cpopen(fs, command, "r", [&](FILE* p) {
    fgets(buf_ps, sizeof(buf_ps), p);
    host_device_map[hostFun].reg_num = atoi(buf_ps);
    memset(buf_ps, '\0', sizeof(buf_ps));
  });

  fs.close();
}

void getSharedMemorySize(std::string device_func_name, std::string elf_filename,
                         const char* hostFun) {
  char command[1000];
  std::fstream fs;

  char buf_ps[1024] = {0};
#if RT_TRACE == 1
  fs.open(debug_info, std::ios::app);
#endif
  snprintf(command, 1000,
           "grep .nv.shared.%s %s| head -1 | awk -F' ' '{print $3;}'",
           device_func_name.c_str(), elf_filename.c_str());
  cpopen(fs, command, "r", [&](FILE* p) {
    fgets(buf_ps, sizeof(buf_ps), p);
    sscanf(buf_ps, "%x", &host_device_map[hostFun].shared_memory_size);
    fgets(buf_ps, sizeof(buf_ps), p);
  });

  fs.close();
}

void getLocalMemorySize(std::string device_func_name, std::string elf_filename,
                        const char* hostFun, gpgpu_device_h device) {
  char command[1000];
  std::fstream fs;

  char buf_ps[1024] = {0};
#if RT_TRACE == 1
  fs.open(debug_info, std::ios::app);
#endif
  snprintf(
      command, 1000,
      "grep -n \"frame size\" %s | grep -n [^$]%s | awk -F ' ' '{print $7;}'",
      elf_filename.c_str(), device_func_name.c_str());
  cpopen(fs, command, "r", [&](FILE* p) {
    fgets(buf_ps, sizeof(buf_ps), p);
    sscanf(buf_ps, "%x", &host_device_map[hostFun].local_mem_size);
  });

  auto buffer = new uint8_t[4];
  // upload local memory size to C[0x0][0x28]
  {
    memcpy((void*)buffer, &host_device_map[hostFun].local_mem_size, 4);
    gpgpu_copy_to_dev(device, (void*)buffer, 0x28, 4, 0);
  }

  delete[] buffer;
  fs.close();
}

void getConstant2(std::string device_func_name, std::string elf_filename,
                  const char* hostFun, gpgpu_device_h device) {
  char command[1000];
  std::fstream fs;

  char buf_ps[1024] = {0};
#if RT_TRACE == 1
  fs.open(debug_info, std::ios::app);
#endif
  // locate the starting point of the constant memory info
  snprintf(command, 1000,
           "grep -n '.constant2.%s' %s | head -1 | awk -F' ' '{print $3, $4;}'",
           device_func_name.c_str(), elf_filename.c_str());
  cpopen(fs, command, "r", [&](FILE* p) {
    if (fgets(buf_ps, sizeof(buf_ps), p) != NULL) {
      sscanf(buf_ps, "%x %x", &host_device_map[hostFun].const2_start_pos,
             &host_device_map[hostFun].const2_size);
    }
  });

  if (host_device_map[hostFun].const2_start_pos != -1) {
    std::ifstream infile;
    infile.open(host_device_map[hostFun].elf_file_path_name, std::ios::in);
    infile.seekg(host_device_map[hostFun].const2_start_pos);

    auto buffer = new char[host_device_map[hostFun].const2_size];
    infile.read(buffer, host_device_map[hostFun].const2_size);
    std::cout << "copy constant variable" << std::endl;
#if RT_TRACE == 1
    fs << "copy constant variable" << std::endl;
#endif
    gpgpu_copy_to_dev(device, (void*)buffer,
                      CONSTANT_MEM_START + CONSTANT_BANK_SIZE * 2,
                      host_device_map[hostFun].const2_size, 0);
    delete[] buffer;
    infile.close();
  }
  fs.close();
}

void** cudaRegisterFatBinaryInternal(void* fatCubin, gpgpu_device_h* device) {
  static unsigned next_fat_bin_handle = 1;
  char command[1000];
  gpgpu_dev_open(device);
  char* pytorch_bin = getenv("PYTORCH_BIN");
  std::string app_binary;
  if (pytorch_bin != NULL && strlen(pytorch_bin) != 0) {
    app_binary = std::string(pytorch_bin);
  } else {
    app_binary = get_app_binary(false);
  }
  std::string app_name = get_app_binary_name(app_binary);

  unsigned long long fat_cubin_handle = next_fat_bin_handle;
  next_fat_bin_handle++;
  if (fat_cubin_handle == 1) {
    // get cubin files' name
    snprintf(command, 1000,
             "$CUDA_INSTALL_PATH/bin/cuobjdump -lelf %s | awk '/sm_70/' | cut "
             "-d \":\" -f 2 | awk \'{gsub(/^\\s+|\\s+$/, \"\");print}\'",
             app_binary.c_str());

    char buf_ps[1024];
    std::vector<std::string> cubin_filenames;
    std::string cubin_filename;

    std::fstream fs;
#if RT_TRACE
    fs.open(debug_info, std::fstream::out);
#endif

    cpopen(fs, command, "r", [&](FILE* p) {
      while (fgets(buf_ps, sizeof(buf_ps), p) != NULL) {
        char* find;
        find = strchr(buf_ps, '\n');  // find '\n'
        if (find)                     // if find, replace with '\0'
          *find = '\0';
        cubin_filename = buf_ps;
        cubin_filename.erase(
            0, cubin_filename.find_first_not_of(" "));  // remove leading blank
        cubin_filenames.push_back(cubin_filename);
      }
    });

    for (auto& filename : cubin_filenames) {
      std::fstream cubin_fs;
      std::string cubin_filename = "/tmp/" + filename;
      cubin_fs.open(cubin_filename, std::ios::in);
      if (cubin_fs.good()) continue;
      // extract cubin
      snprintf(command, 1000,
               "$CUDA_INSTALL_PATH/bin/cuobjdump -xelf  %s %s > /dev/null",
               filename.c_str(), app_binary.c_str());
      if (system(command) != 0) {
        printf("ERROR: command: %s failed \n", command);
        exit(0);
      }

      std::string elf_filename = "/tmp/" + filename + ".elf";
      // dump elf file
      snprintf(command, 1000, "$CUDA_INSTALL_PATH/bin/cuobjdump -elf %s > %s",
               filename.c_str(), elf_filename.c_str());
      if (system(command) != 0) {
        printf("ERROR: command: %s failed \n", command);
        exit(0);
      }

      // mv cubin file to /tmp
      snprintf(command, 1000, "mv %s %s", filename.c_str(),
               ("/tmp/" + filename).c_str());
      if (system(command) != 0) {
        printf("ERROR: command: %s failed \n", command);
        exit(0);
      }
    }
    for (auto& filename : cubin_filenames) {
      std::string elf_filename = "/tmp/" + filename + ".elf";
      std::string cubin_filename = "/tmp/" + filename;
      struct cubin_info t;
      t.elf_filename = elf_filename;

      // extract function names contained in this cubin file
      snprintf(command, 1000,
               "$CUDA_INSTALL_PATH/bin/cuobjdump -symbols %s | grep STT_FUNC | "
               "awk -F\" \" '{print $4}'",
               cubin_filename.c_str());
      cpopen(fs, command, "r", [&](FILE* p) {
        while (fgets(buf_ps, sizeof(buf_ps), p) != NULL) {
          char* find;
          find = strchr(buf_ps, '\n');  // find '\n'
          if (find)                     // if find, replace with '\0'
            *find = '\0';
          std::string func_name = buf_ps;
          func_cubin_map[func_name] = cubin_filename;
        }
      });

      cubin_info_table[cubin_filename] = t;
    }
    fs.close();

    /**
     * @brief extract constant variable
     * .constant0: kernel config, params, driver/compiler controlled
     * .constant1: ?
     * .constant2: double immediate
     * .constant3: __constant__ veriable, array
     */
    for (auto it = cubin_info_table.begin(); it != cubin_info_table.end();
         ++it) {
      for (int i = 1; i <= 3; i++) {
        if (i == 2)  // move extract .constant2 to launch kernel
          continue;
        int const_start_pos = -1;
        int const_size = 0;

        // locate the starting point of the constant memory info
        snprintf(
            command, 1000,
            "grep -n '.constant%d' %s | head -1 | awk -F' ' '{print $3, $4;}'",
            i, it->second.elf_filename.c_str());
        cpopen(fs, command, "r", [&](FILE* p) {
          if (fgets(buf_ps, sizeof(buf_ps), p) != NULL) {
            sscanf(buf_ps, "%x %x", &const_start_pos, &const_size);
          }
        });

        if (const_start_pos != -1) {
          std::string elf_file_relpath = it->first;

          std::ifstream infile;
          infile.open(elf_file_relpath, std::ios::in);
          infile.seekg(const_start_pos);

          auto buffer = new char[const_size];
          infile.read(buffer, const_size);

          std::cout << "copy constant variable" << std::endl;
          gpgpu_copy_to_dev(*device, (void*)buffer,
                            CONSTANT_MEM_START + CONSTANT_BANK_SIZE * i,
                            const_size, 0);

          delete[] buffer;
          infile.close();
        }
      }
    }
  }
  return (void**)fat_cubin_handle;
}

cudaError_t cudaConfigureCallInternal(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, cudaStream_t stream,
                                      gpgpu_device_h device) {
  auto buffer = new uint8_t[2 * sizeof(dim3)];
  std::cout << "upload gridDim, blockDim" << std::endl;

  int offset = 0;
  memcpy((void*)buffer + offset, &blockDim, sizeof(dim3));
  offset += sizeof(dim3);

  memcpy((void*)buffer + offset, &gridDim, sizeof(dim3));
  offset += sizeof(dim3);

  gpgpu_copy_to_dev(device, (void*)buffer, CONSTANT_MEM_START, 2 * sizeof(dim3),
                    0);

  // TODO: edit the code address and size
  cmd_l.clear();

  cmd_l.block_dim = blockDim;
  cmd_l.grid_dim = gridDim;
  cmd_l.code_address = STARTUP_ADDR;
  cmd_l.smem_size = sharedMem;

  delete[] buffer;
#if RT_TRACE == 1
  std::fstream fs;
  fs.open(debug_info, std::ios::app);
  fs << "upload gridDim, blockDim" << std::endl;
  fs.close();
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaLaunchInternal(const char* hostFun,
                                                  gpgpu_device_h device) {
  std::string device_func_name(host_device_map[hostFun].kernel_name);
  std::string cubin_filename = func_cubin_map[device_func_name];
  std::string elf_filename = cubin_info_table[cubin_filename].elf_filename;

  // kernel launch 添加code段
  if (host_device_map[hostFun].offset == -1) {
    addKernel(cubin_filename, device_func_name, elf_filename, hostFun, device);
  }

  // get the sass code text file
  std::vector<std::string> sass_code_text;
  addCodeText(cubin_filename, device_func_name, &sass_code_text);

  // add smem_size, reg_num to kernelInfo struct
  if (host_device_map[hostFun].reg_num == 0) {
    getRegNum(device_func_name, elf_filename, hostFun);
  }

  if (host_device_map[hostFun].shared_memory_size == 0) {
    getSharedMemorySize(device_func_name, elf_filename, hostFun);
  }

  if (host_device_map[hostFun].local_mem_size == 0) {
    // extract local memory size
    getLocalMemorySize(device_func_name, elf_filename, hostFun, device);
  }

  if (host_device_map[hostFun].const2_start_pos == -1) {
    getConstant2(device_func_name, elf_filename, hostFun, device);
  }

  cmd_l.code_size = host_device_map[hostFun].size;
  cmd_l.regs_size = host_device_map[hostFun].reg_num;
  cmd_l.sass_code_text = &sass_code_text;
  if (cmd_l.smem_size == 0)
    cmd_l.smem_size = host_device_map[hostFun].shared_memory_size;

  gpgpu_push_command(device, (void*)(&cmd_l), sizeof(cmd_l));
  std::cout << "kernel launch" << std::endl;
#if RT_TRACE == 1
  std::fstream fs;
  fs.open(debug_info, std::ios::app);
  fs << "kernel launch" << std::endl;
  fs.close();
#endif

  gpgpu_start(device);
  gpgpu_ready_wait(device, MAX_TIMEOUT);
  // gpgpu_dev_close(device);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernelInternal(
    const char* hostFun, dim3 gridDim, dim3 blockDim, const void** args,
    size_t sharedMem, cudaStream_t stream, gpgpu_device_h device) {
#if RT_TRACE == 1
  std::fstream fs;
  fs.open(debug_info, std::ios::app);
#endif

  std::string device_func_name(host_device_map[hostFun].kernel_name);
  std::string cubin_filename = func_cubin_map[device_func_name];
  std::string elf_filename = cubin_info_table[cubin_filename].elf_filename;

  // locate the line nums of the param info's linenum
  setupAllArguments(device_func_name, elf_filename, args, device);

  // kernel launch 添加code段
  if (host_device_map[hostFun].offset == -1) {
    addKernel(cubin_filename, device_func_name, elf_filename, hostFun, device);
  } else {
    std::ifstream infile;
    infile.open(host_device_map[hostFun].elf_file_path_name, std::ios::in);
    infile.seekg(host_device_map[hostFun].offset);
    auto buffer = new char[host_device_map[hostFun].size];
    infile.read(buffer, host_device_map[hostFun].size);
    std::cout << "upload program" << std::endl;
    gpgpu_copy_to_dev(device, (void*)buffer, STARTUP_ADDR,
                      host_device_map[hostFun].size, 0);
    delete[] buffer;
    infile.close();
#if RT_TRACE == 1
    fs << "upload program" << std::endl;
    fs.close();
#endif
  }

  // get the sass code text file
  std::vector<std::string> sass_code_text;
  // addCodeText(cubin_filename, device_func_name, &sass_code_text);

  // add smem_size, reg_num to kernelInfo struct
  if (host_device_map[hostFun].reg_num == 0) {
    getRegNum(device_func_name, elf_filename, hostFun);
  }

  if (host_device_map[hostFun].shared_memory_size == 0) {
    getSharedMemorySize(device_func_name, elf_filename, hostFun);
  }

  if (host_device_map[hostFun].local_mem_size == 0) {
    // extract local memory size
    getLocalMemorySize(device_func_name, elf_filename, hostFun, device);
  }

  if (host_device_map[hostFun].const2_start_pos == -1) {
    getConstant2(device_func_name, elf_filename, hostFun, device);
  }

  cmd_l.code_size = host_device_map[hostFun].size;
  cmd_l.regs_size = host_device_map[hostFun].reg_num;
  cmd_l.sass_code_text = &sass_code_text;
  if (cmd_l.smem_size == 0)
    cmd_l.smem_size = host_device_map[hostFun].shared_memory_size;
  gpgpu_push_command(device, &cmd_l, sizeof(cmd_l));
  std::cout << "kernel launch" << std::endl;
#if RT_TRACE == 1
  fs << "kernel launch" << std::endl;
  fs.close();
#endif
  gpgpu_start(device);
  gpgpu_ready_wait(device, MAX_TIMEOUT);
  // gpgpu_dev_close(device);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyInternal(void* dst, const void* src,
                                                  size_t count,
                                                  enum cudaMemcpyKind kind,
                                                  gpgpu_device_h device) {
  if (kind == cudaMemcpyHostToDevice) {
    char* src_data = (char*)src;
    gpgpu_copy_to_dev(device, (void*)src_data, (uint64_t)dst, count, 0);
  } else if (kind == cudaMemcpyDeviceToHost) {
    unsigned char* dst_data = (unsigned char*)dst;
    gpgpu_copy_from_dev(device, (void*)dst_data, (uint64_t)src, count, 0);
  } else if (kind == cudaMemcpyDeviceToDevice) {
    auto buffer = new uint8_t[count];
    gpgpu_copy_from_dev(device, (void*)buffer, (uint64_t)src, count, 0);
    gpgpu_copy_to_dev(device, (void*)buffer, (uint64_t)dst, count, 0);
    delete[] buffer;
  } else {
#if RT_TRACE == 1
    std::fstream fs;
    fs.open(debug_info, std::ios::app);
    fs << "GPGPU RUNTIME ERROR: cudaMemcpy - ERROR : unsupported "
          "cudaMemcpyKind\n";
#endif
    printf(
        "GPGPU RUNTIME ERROR: cudaMemcpy - ERROR : unsupported "
        "cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemsetInternal(void* mem, int c,
                                                  size_t count,
                                                  gpgpu_device_h device) {
#if RT_TRACE == 1
  std::fstream fs;
  fs.open(debug_info, std::ios::app);
#endif

  auto buffer = new uint8_t[count];
  unsigned char c_value = (unsigned char)c;
  for (unsigned n = 0; n < count; n++)
    memcpy((void*)((unsigned char*)buffer + n), &c_value, 1);
  gpgpu_copy_to_dev(device, (void*)buffer, (uint64_t)mem, count, 0);

  delete[] buffer;
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

extern "C" {

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
cudaError_t cudaPeekAtLastError(void) { return g_last_cudaError; }

__host__ cudaError_t CUDARTAPI cudaMalloc(void** devPtr, size_t size) {
  unsigned long long result = m_dev_malloc;
  m_dev_malloc += size;
  size_t size_align = size;
  if (size % 256) {
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries
    size_align += (256 - size % 256);
  }
  void* init_data = NULL;
  init_data = calloc(1, size_align);
  gpgpu_device_h device = gpgpu_device;

  auto buffer = new uint8_t[size_align];
  memcpy((void*)buffer, (void*)init_data, size_align);
  gpgpu_copy_to_dev(device, (void*)buffer, (uint64_t)result, size_align, 0);

  delete[] buffer;

  *devPtr = (void*)result;
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMallocHost(void** ptr, size_t size) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}
__host__ cudaError_t CUDARTAPI cudaMallocPitch(void** devPtr, size_t* pitch,
                                               size_t width, size_t height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMallocArray(
    struct cudaArray** array, const struct cudaChannelFormatDesc* desc,
    size_t width, size_t height __dv(1)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaHostAlloc(void** pHost, size_t bytes,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *pHost = malloc(bytes);
  pinned_memory_size[*pHost] = bytes;
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaFree(void* devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
}
__host__ cudaError_t CUDARTAPI cudaFreeHost(void* ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  free(ptr);  // this will crash the system if called twice
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray* array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
};

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpy(void* dst, const void* src,
                                          size_t count,
                                          enum cudaMemcpyKind kind) {
  return cudaMemcpyInternal(dst, src, count, kind, gpgpu_device);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray* dst,
                                                 size_t wOffset, size_t hOffset,
                                                 const void* src, size_t count,
                                                 enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void* dst,
                                                   const struct cudaArray* src,
                                                   size_t wOffset,
                                                   size_t hOffset, size_t count,
                                                   enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(
    struct cudaArray* dst, size_t wOffsetDst, size_t hOffsetDst,
    const struct cudaArray* src, size_t wOffsetSrc, size_t hOffsetSrc,
    size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void* dst, size_t dpitch,
                                            const void* src, size_t spitch,
                                            size_t width, size_t height,
                                            enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(
    struct cudaArray* dst, size_t wOffset, size_t hOffset, const void* src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(
    void* dst, size_t dpitch, const struct cudaArray* src, size_t wOffset,
    size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(
    struct cudaArray* dst, size_t wOffsetDst, size_t hOffsetDst,
    const struct cudaArray* src, size_t wOffsetSrc, size_t hOffsetSrc,
    size_t width, size_t height,
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(
    const char* symbol, const void* src, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(
    void* dst, const char* symbol, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t* free, size_t* total) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // placeholder; should interact with cudaMalloc and cudaFree?
  *free = 10000000000;
  *total = 10000000000;

  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void* dst, const void* src,
                                               size_t count,
                                               enum cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaMemcpyInternal(dst, src, count, kind, gpgpu_device);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(
    struct cudaArray* dst, size_t wOffset, size_t hOffset, const void* src,
    size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(
    void* dst, const struct cudaArray* src, size_t wOffset, size_t hOffset,
    size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void* dst, size_t dpitch,
                                                 const void* src, size_t spitch,
                                                 size_t width, size_t height,
                                                 enum cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(
    struct cudaArray* dst, size_t wOffset, size_t hOffset, const void* src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(
    void* dst, size_t dpitch, const struct cudaArray* src, size_t wOffset,
    size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const char* hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ __device__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
__host__ cudaError_t CUDARTAPI cudaMemset(void* mem, int c, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaMemsetInternal(mem, c, count, gpgpu_device);
}

// memset operation is done but i think its not async?
__host__ cudaError_t CUDARTAPI cudaMemsetAsync(void* mem, int c, size_t count,
                                               cudaStream_t stream = 0) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemset2D(void* mem, size_t pitch, int c,
                                            size_t width, size_t height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void** devPtr,
                                                    const char* symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t* size,
                                                 const char* symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
__host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int* count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *count = 1;
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  snprintf(prop->name, 256, "GPGPU_v1");
  prop->major = 7;
  prop->minor = 0;
  prop->totalGlobalMem = 0x100000000; /* 4 GB */
  prop->memPitch = 0;

  prop->maxThreadsPerBlock = 1024;
  prop->maxThreadsDim[0] = 1024;
  prop->maxThreadsDim[1] = 1024;
  prop->maxThreadsDim[2] = 64;

  prop->maxGridSize[0] = 0x40000000;
  prop->maxGridSize[1] = 0x40000000;
  prop->maxGridSize[2] = 0x40000000;
  prop->totalConstMem = 0x40000000;
  prop->textureAlignment = 0;
  prop->sharedMemPerBlock = 65536;

  prop->regsPerMultiprocessor = 65536;
  prop->sharedMemPerMultiprocessor = 98304;

  prop->regsPerBlock = 65536;
  prop->warpSize = WARP_SIZE;
  prop->clockRate = 1200.0 * 1000000 / 1000;

  prop->multiProcessorCount = NUM_CORES;

  prop->maxThreadsPerMultiProcessor = MAX_THREAD_PER_SM;
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* value,
                                                      enum cudaDeviceAttr attr,
                                                      int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  switch (attr) {
    case 1:
      *value = 1024;  // prop->maxThreadsPerBlock;
      break;
    case 2:
      *value = 1024;  // prop->maxThreadsDim[0];
      break;
    case 3:
      *value = 1024;  // prop->maxThreadsDim[1];
      break;
    case 4:
      *value = 64;  // prop->maxThreadsDim[2];
      break;
    case 5:
      *value = 2147483647;  // prop->maxGridSize[0];
      break;
    case 6:
      *value = 65535;  // prop->maxGridSize[1];
      break;
    case 7:
      *value = 65535;  // prop->maxGridSize[2];
      break;
    case 8:
      *value = 49152;  // prop->sharedMemPerBlock;
      break;
    case 9:
      *value = 0x40000000;  // prop->totalConstMem;
      break;
    case 10:
      *value = 32;  // prop->warpSize;
      break;
    case 11:
      *value = 16;  // dummy value
      break;
    case 12:
      *value = 65536;  // prop->regsPerBlock;
      break;
    case 13:
      *value = 1480000;  // for 1080ti
      break;
    case 14:
      *value = 512;  // prop->textureAlignment;
      break;
    case 15:
      *value = 0;
      break;
    case 16:
      *value = 80;  // prop->multiProcessorCount;
      break;
    case 17:
    case 18:
    case 19:
      *value = 0;
      break;
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 42:
    case 45:
    case 46:
    case 47:
    case 48:
    case 49:
    case 52:
    case 53:
    case 55:
    case 56:
    case 57:
    case 58:
    case 59:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 66:
    case 67:
    case 69:
    case 70:
    case 71:
    case 73:
    case 74:
    case 77:
      *value = 1000;  // dummy value
      break;
    case 29:
    case 43:
    case 54:
    case 65:
    case 68:
    case 72:
      *value = 10;  // dummy value
      break;
    case 30:
    case 51:
      *value = 128;  // dummy value
      break;
    case 31:
      *value = 1;
      break;
    case 32:
      *value = 0;
      break;
    case 33:
    case 50:
      *value = 0;  // dummy value
      break;
    case 34:
      *value = 0;
      break;
    case 35:
      *value = 0;
      break;
    case 36:
      *value = 1250000;  // CK value for 1080ti
      break;
    case 37:
      *value = 352;  // value for 1080ti
      break;
    case 38:
      *value = 3000000;  // value for 1080ti
      break;
    case 39:
      *value = 2048;  // dev->get_gpgpu()->threads_per_core();
      break;
    case 40:
      *value = 0;
      break;
    case 41:
      *value = 0;
      break;
    case 75:  // cudaDevAttrComputeCapabilityMajor
      *value = 7;
      break;
    case 76:  // cudaDevAttrComputeCapabilityMinor
      *value = 0;
      break;
    case 78:
      *value = 0;  // TODO: as of now, we dont support stream priorities.
      break;
    case 79:
      *value = 0;
      break;
    case 80:
      *value = 0;
      break;
    case 81:
      *value = 98304;  // prop->sharedMemPerMultiprocessor;
      break;
    case 82:
      *value = 65536;  // prop->regsPerMultiprocessor;
      break;
    case 83:
    case 84:
    case 85:
    case 86:
      *value = 0;
      break;
    case 87:
      *value = 4;  // dummy value
      break;
    case 88:
    case 89:
      *value = 0;
      break;
    case 97:
      *value = 98304;
      break;
    default:
      printf("ERROR: Attribute number %d unimplemented \n", attr);
      abort();
  }
  // printf("getprop %d\n", attr);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaChooseDevice(int* device, const struct cudaDeviceProp* prop) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetDevice(int* device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *device = 0;
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t* pValue,
                                                  cudaLimit limit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream,
                                                     int* priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char* pciBusId, int len,
                                                     int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle,
                                                   void* devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t cudaIpcOpenMemHandle(void** devPtr,
                                          cudaIpcMemHandle_t handle,
                                          unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI
cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaBindTexture(
    size_t* offset, const struct textureReference* texref, const void* devPtr,
    const struct cudaChannelFormatDesc* desc, size_t size __dv(UINT_MAX)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToArray(
    const struct textureReference* texref, const struct cudaArray* array,
    const struct cudaChannelFormatDesc* desc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaUnbindTexture(const struct textureReference* texref) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(
    size_t* offset, const struct textureReference* texref) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureReference(
    const struct textureReference** texref, const char* symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(
    struct cudaChannelFormatDesc* desc, const struct cudaArray* array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *desc = array->desc;
  return g_last_cudaError = cudaSuccess;
}

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(
    int x, int y, int z, int w, enum cudaChannelFormatKind f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct cudaChannelFormatDesc dummy;
  dummy.x = x;
  dummy.y = y;
  dummy.z = z;
  dummy.w = w;
  dummy.f = f;
  return dummy;
}

__host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError;
}

__host__ const char* cudaGetErrorName(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return NULL;
}

__host__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (g_last_cudaError == cudaSuccess) return "no error";
  char buf[1024];
  snprintf(buf, 1024, "<<GPGPU: there was an error (code = %d)>>",
           g_last_cudaError);
  return strdup(buf);
}

__host__ cudaError_t CUDARTAPI cudaSetupArgumentInternal(
    const void* arg, size_t size, size_t offset, gpgpu_device_h device) {
#if RT_TRACE == 1
  std::fstream fs;
  fs.open(debug_info, std::ios::app);
  fs << "upload kernel arguments" << std::endl;
  fs.close();
#endif
  auto buffer = new uint8_t[size];
  std::cout << "upload kernel arguments" << std::endl;
  memcpy((void*)buffer, (void*)arg, size);
  gpgpu_copy_to_dev(device, (void*)buffer, KERNEL_PARAMS_START + offset, size,
                    0);

  delete[] buffer;
}

__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void* arg, size_t size,
                                                 size_t offset) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaSetupArgumentInternal(arg, size, offset, gpgpu_device);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaLaunch(const char* hostFun) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaLaunchInternal(hostFun, gpgpu_device);
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const char* hostFun,
                                                dim3 gridDim, dim3 blockDim,
                                                const void** args,
                                                size_t sharedMem,
                                                cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaLaunchKernelInternal(hostFun, gridDim, blockDim, args, sharedMem,
                                  stream, gpgpu_device);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t* stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

// TODO: introduce priorities
__host__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(
    cudaStream_t* stream, unsigned int flags, int priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI
cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaSuccess;
}

__host__ __device__ cudaError_t CUDARTAPI
cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (stream == NULL) return g_last_cudaError = cudaErrorInvalidResourceHandle;
  return cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                               cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream,
                                                   cudaEvent_t event,
                                                   unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float* ms,
                                                    cudaEvent_t start,
                                                    cudaEvent_t end) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaThreadExit(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaSuccess;
}

int CUDARTAPI __cudaSynchronizeThreads(void**, void*) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaThreadExit();
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

int dummy0() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return 0;
}

int dummy1() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return 2 << 20;
}

typedef int (*ExportedFunction)();

static ExportedFunction exportTable[3] = {&dummy0, &dummy0, &dummy0};

__host__ cudaError_t CUDARTAPI cudaGetExportTable(
    const void** ppExportTable, const cudaUUID_t* pExportTableId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("cudaGetExportTable: UUID = ");
  for (int s = 0; s < 16; s++) {
    printf("%#2x ", (unsigned char)(pExportTableId->bytes[s]));
  }
  printf("b\b");
  *ppExportTable = (void*)exportTable;
  printf("a\b");
  printf("\n");
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

//#include "../../cuobjdump_to_ptxplus/cuobjdump_parser.h"

//! Read file into char*
// TODO: convert this to C++ streams, will be way cleaner
char* readfile(const std::string filename) {
  assert(filename != "");
  FILE* fp = fopen(filename.c_str(), "r");
  if (!fp) {
    std::cout << "ERROR: Could not open file %s for reading\n"
              << filename << std::endl;
    assert(0);
  }
  // finding size of the file
  int filesize = 0;
  fseek(fp, 0, SEEK_END);

  filesize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  // allocate and copy the entire ptx
  char* ret = (char*)malloc((filesize + 1) * sizeof(char));
  fread(ret, 1, filesize, fp);
  ret[filesize] = '\0';
  fclose(fp);
  return ret;
}
}
extern "C" {

void** CUDARTAPI __cudaRegisterFatBinary(void* fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaRegisterFatBinaryInternal(fatCubin, &gpgpu_device);
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem = 0,
                                               struct CUstream_st* stream = 0) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream, gpgpu_device);
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                                 size_t* sharedMem,
                                                 void* stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

void CUDARTAPI __cudaRegisterFunction(void** fatCubinHandle,
                                      const char* hostFun, char* deviceFun,
                                      const char* deviceName, int thread_limit,
                                      uint3* tid, uint3* bid, dim3* bDim,
                                      dim3* gDim) {
  deviceInfo temp;
  // if(host_device_map.count(hostFun)!=0){
  //   printf("the same\n");
  // }
  // printf(
  // "__cudaRegisterFunction %s : hostFun %p : fatCubinHandle = %u\n",
  // deviceFun, hostFun,fatCubinHandle);
  temp.kernel_name = deviceFun;
  host_device_map[hostFun] = temp;
}

extern void __cudaRegisterVar(
    void** fatCubinHandle,
    char* hostVar,           // pointer to...something
    char* deviceAddress,     // name of variable
    const char* deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global) {}

__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                                 size_t sharedMem,
                                                 cudaStream_t stream) {
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream, gpgpu_device);
}

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // delete files
  char command[1000];
  for (auto& it : cubin_info_table) {
    snprintf(command, 1000, "rm -f %s", it.first.c_str());
    if (system(command) != 0) {
      printf("ERROR: command: %s failed \n", command);
      exit(0);
    }
    snprintf(command, 1000, "rm -f %s", it.second.elf_filename.c_str());
    if (system(command) != 0) {
      printf("ERROR: command: %s failed \n", command);
      exit(0);
    }
  }
}

cudaError_t cudaDeviceReset(void) {
  // Should reset the simulated GPU
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {}

void __cudaRegisterShared(void** fatCubinHandle, void** devicePtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU: __cudaRegisterShared\n");
}

void CUDARTAPI __cudaRegisterSharedVar(void** fatCubinHandle, void** devicePtr,
                                       size_t size, size_t alignment,
                                       int storage) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU: __cudaRegisterSharedVar\n");
}

void __cudaRegisterTexture(
    void** fatCubinHandle, const struct textureReference* hostVar,
    const void** deviceAddress, const char* deviceName, int dim, int norm,
    int ext)  // passes in a newly created textureReference
{}

char __cudaInitModule(void** fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t cudaGLRegisterBufferObject(GLuint bufferObj) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU: Execution warning: ignoring call to \"%s\"\n", __my_func__);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaGLMapBufferObject(void** devPtr, GLuint bufferObj) {
  return cudaSuccess;
}

cudaError_t cudaGLUnmapBufferObject(GLuint bufferObj) { return cudaSuccess; }

cudaError_t cudaGLUnregisterBufferObject(GLuint bufferObj) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU: Execution warning: ignoring call to \"%s\"\n", __my_func__);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer(void** pDevice, void* pHost,
                                               unsigned int flags) {
  assert(pinned_memory_size.count(pHost) != 0);
  size_t size = pinned_memory_size[pHost];
  unsigned long long result = m_dev_malloc;
  m_dev_malloc += size;
  if (size % 256)
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries
  *pDevice = (void*)result;

  // std::cout << "according cudaHostGetDevicePointer\n";
  if (*pDevice) {
    pinned_memory[pHost] = pDevice;
    // Copy contents in cpu to gpu
    cudaMemcpyInternal(*pDevice, pHost, size, cudaMemcpyHostToDevice,
                       gpgpu_device);
  }
  return cudaSuccess;
}

cudaError_t CUDAAPI cudaIpcCloseMemHandle(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit,
                                                  size_t value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                                    cudaGraph_t graph,
                                                    cudaGraphNode_t* pErrorNode,
                                                    char* pLogBuffer,
                                                    size_t bufferSize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t cudaLaunchCooperativeKernelMultiDevice(
    cudaLaunchParams* launchParamsList, unsigned int numDevices,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn,
                                        void* userData) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaStreamGetCaptureInfo(
    cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaHostUnregister(void* ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}
__host__ cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                            cudaStreamCaptureMode mode) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaDeviceGetByPCIBusId(int* device,
                                             const char* pciBusId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice,
                                                          unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec,
                                     cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaMallocManaged(void** devPtr, size_t size,
                                       unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaStreamIsCapturing(
    cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaStreamEndCapture(cudaStream_t stream,
                                          cudaGraph_t* pGraph) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t
cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaProfilerInitialize(const char* configFile,
                                            const char* outputFile,
                                            cudaOutputMode_t outputMode) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event,
                                            cudaIpcEventHandle_t handle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle,
                                           cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status,
                                     void* userData);
__host__ cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                           cudaStreamCallback_t callback,
                                           void* userData, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaCreateTextureObject(
    cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc,
    const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int* canAccessPeer,
                                                       int device,
                                                       int peerDevice) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetValidDevices(int* device_arr, int len) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetDeviceFlags(int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // This flag is implicitly always on (unless you are using the driver API)
  if (cudaDeviceMapHost == flags) {
    return g_last_cudaError = cudaSuccess;
  } else {
    cuda_not_implemented(__my_func__, __LINE__);
    return g_last_cudaError = cudaErrorUnknown;
  }
}

cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes* attr,
                                            const char* hostFun) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  CUevent_st* e = new CUevent_st(flags == cudaEventBlockingSync);

  *event = e;

  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int* driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *driverVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* runtimeVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *runtimeVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaFuncSetCacheConfig(const char* func, enum cudaFuncCache cacheConfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return cudaSuccess;
}

/**
 * \brief Set attributes for a given function
 *
 * This function sets the attributes of a function specified via \p entry.
 * The parameter \p entry must be a pointer to a function that executes
 * on the device. The parameter specified by \p entry must be declared as a \p
 * __global__ function. The enumeration defined by \p attr is set to the value
 * defined by \p value If the specified function does not exist, then
 * ::cudaErrorInvalidDeviceFunction is returned. If the specified attribute
 * cannot be written, or if the value is incorrect, then ::cudaErrorInvalidValue
 * is returned.
 *
 * Valid values for \p attr are:
 * ::cuFuncAttrMaxDynamicSharedMem - Maximum size of dynamic shared memory per
 * block
 * ::cudaFuncAttributePreferredSharedMemoryCarveout - Preferred shared memory-L1
 * cache split ratio
 *
 * \param entry - Function to get attributes of
 * \param attr  - Attribute to set
 * \param value - Value to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void
 * **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig
 * (C++ API)", \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const
 * void*) "cudaFuncGetAttributes (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(T, size_t) "cudaSetupArgument (C++ API)"
 */
cudaError_t CUDARTAPI cudaFuncSetAttribute(const void* func,
                                           enum cudaFuncAttribute attr,
                                           int value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf(
      "GPGPU: Execution warning: ignoring call to \"%s ( func=%p, "
      "attr=%d, value=%d )\"\n",
      __my_func__, func, attr, value);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaGLSetGLDevice(int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU: Execution warning: ignoring call to \"%s\"\n", __my_func__);
  return g_last_cudaError = cudaErrorUnknown;
}

typedef void* HGPUNV;

cudaError_t CUDARTAPI cudaWGLGetDevice(int* device, HGPUNV hGpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

void CUDARTAPI __cudaMutexOperation(int lock) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

void CUDARTAPI __cudaTextureFetch(const void* tex, void* index, int integer,
                                  void* val) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}
}

namespace cuda_math {

void CUDARTAPI __cudaMutexOperation(int lock) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

void CUDARTAPI __cudaTextureFetch(const void* tex, void* index, int integer,
                                  void* val) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

int CUDARTAPI __cudaSynchronizeThreads(void**, void*) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO This function should syncronize if we support Asyn kernel calls
  return g_last_cudaError = cudaSuccess;
}

}  // namespace cuda_math

////////

/// static functions

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
//***extra api for pytorch***

CUresult CUDAAPI cuGetErrorString(CUresult error, const char** pStr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char** pStr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion(int* driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *driverVersion = 10200;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  int deviceI = -1;
  cudaError_t e = cudaGetDevice(&deviceI);
  assert(e == cudaSuccess);
  assert(deviceI != -1);
  *device = deviceI;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int* count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaGetDeviceCount(count);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  assert(len >= 10);
  strcpy(name, "GPGPU");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *bytes = 20000000000;  // dummy value
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaDeviceGetAttribute(pi, (cudaDeviceAttr)attrib, dev);
  assert(e == cudaSuccess);

  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceComputeCapability(int* major, int* minor,
                                           CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags,
                                            int* active) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate(CUcontext* pctx, unsigned int flags,
                             CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext* pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetFlags(unsigned int* flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetLimit(size_t* pvalue, CUlimit limit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache* pconfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int* leastPriority,
                                             int* greatestPriority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxAttach(CUcontext* pctx, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDetach(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoad(CUmodule* module, const char* fname) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* image,
                                    unsigned int numOptions,
                                    CUjit_option* options,
                                    void** optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                                     const char* name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                                   CUmodule hmod, const char* name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod,
                                   const char* name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod,
                                    const char* name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option* options,
                              void** optionValues, CUlinkState* stateOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // currently do not support options or multiple CUlinkStates
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void* data, size_t size, const char* name,
                               unsigned int numOptions, CUjit_option* options,
                               void** optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  assert(type == CU_JIT_INPUT_PTX);
  cuda_not_implemented(__my_func__, __LINE__);
  return CUDA_ERROR_UNKNOWN;
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char* path, unsigned int numOptions,
                               CUjit_option* options, void** optionValues) {
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void** cubinOut,
                                size_t* sizeOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // all cuLink* function are implemented to block until completion so nothing
  // to do here
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // currently do not support options or multiple CUlinkStates
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetInfo(size_t* free, size_t* total) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize,
                                      CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocHost(void** pp, size_t bytesize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFreeHost(void* p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize,
                                unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p,
                                           unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostGetFlags(unsigned int* pFlags, void* p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize,
                                   unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent* phEvent,
                                      CUipcEventHandle handle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostRegister(void* p, size_t bytesize,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
__host__ cudaError_t cudaHostRegister(void* ptr, size_t size,
                                      unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t cudaProfilerStart() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t cudaProfilerStop() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

CUresult CUDAAPI cuMemHostUnregister(void* p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, size_t dstOffset,
                              CUdeviceptr srcDevice, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray,
                              size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset,
                              const void* srcHost, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoH(void* dstHost, CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset,
                              CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset,
                                   const void* srcHost, size_t ByteCount,
                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoHAsync(void* dstHost, CUarray srcArray,
                                   size_t srcOffset, size_t ByteCount,
                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy,
                                     CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us,
                             size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                              unsigned char uc, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned short us, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned int ui, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayCreate(CUarray* pHandle,
                               const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor,
                                      CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayDestroy(CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArray3DCreate(
    CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArray3DGetDescriptor(
    CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMipmappedArrayCreate(CUmipmappedArray* pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray* pLevelArray,
                                          CUmipmappedArray hMipmappedArray,
                                          unsigned int level) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_MEM */

CUresult CUDAAPI cuPointerGetAttribute(void* data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRangeGetAttribute(void* data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRangeGetAttributes(void** data, size_t* dataSizes,
                                         CUmem_range_attribute* attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuPointerSetAttribute(const void* value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute* attributes,
                                        void** data, CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_UNIFIED */

CUresult CUDAAPI cuStreamCreate(CUstream* phStream, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* phStream,
                                            unsigned int flags, int priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int* priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void* userData,
                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_STREAM */

CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventElapsedTime(float* pMilliseconds, CUevent hStart,
                                    CUevent hEnd) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams* paramArray,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_EVENT */

CUresult CUDAAPI cuFuncGetAttribute(int* pi, CUfunction_attribute attrib,
                                    CUfunction hfunc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void** kernelParams, void** extra) {
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_EXEC */

CUresult CUDAAPI cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetf(CUfunction hfunc, int offset, float value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetv(CUfunction hfunc, int offset, void* ptr,
                             unsigned int numbytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunch(CUfunction f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchGridAsync(CUfunction f, int grid_width,
                                   int grid_height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc, int texunit,
                                  CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/** @} */ /* END CUDA_EXEC_DEPRECATED */

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int* minGridSize, int* blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int* minGridSize, int* blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_OCCUPANCY */

CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                  unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                           CUmipmappedArray hMipmappedArray,
                                           unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddress(size_t* ByteOffset, CUtexref hTexRef,
                                    CUdeviceptr dptr, size_t bytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef,
                                      const CUDA_ARRAY_DESCRIPTOR* desc,
                                      CUdeviceptr dptr, size_t Pitch) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                   int NumPackedComponents) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                        CUaddress_mode am) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                             float minMipmapLevelClamp,
                                             float maxMipmapLevelClamp) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                          unsigned int maxAniso) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr* pdptr, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray,
                                           CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef,
                                        int dim) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels,
                                   CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm,
                                             CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp,
                                             float* pmaxMipmapLevelClamp,
                                             CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefCreate(CUtexref* pTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_SURFREF */

CUresult CUDAAPI
cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc,
                  const CUDA_TEXTURE_DESC* pTexDesc,
                  const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc,
                                            CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc,
                                           CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetResourceViewDesc(
    CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_TEXOBJECT */

CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject* pSurfObject,
                                    const CUDA_RESOURCE_DESC* pResDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc,
                                             CUsurfObject surfObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetP2PAttribute(int* value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_PEER_ACCESS */

CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(
    CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(
    CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                               unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource* resources,
                                        CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource* resources,
                                          CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_GRAPHICS */

CUresult CUDAAPI cuGetExportTable(const void** ppExportTable,
                                  const CUuuid* pExportTableId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  return CUDA_SUCCESS;
}

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 4000 && CUDART_VERSION < 6050)
CUresult CUDAAPI cuMemHostRegister(void* p, size_t bytesize,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* defined(CUDART_VERSION_INTERNAL) || (CUDART_VERSION >= 4000 && \
          CUDART_VERSION < 6050) */

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 5050 && CUDART_VERSION < 6050)
CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option* options,
                              void** optionValues, CUlinkState* stateOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void* data, size_t size, const char* name,
                               unsigned int numOptions, CUjit_option* options,
                               void** optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char* path, unsigned int numOptions,
                               CUjit_option* options, void** optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || (CUDART_VERSION >= 5050 && CUDART_VERSION \
          < 6050) */

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 3020 && CUDART_VERSION < 4010)
CUresult CUDAAPI cuTexRefSetAddress2D_v2(CUtexref hTexRef,
                                         const CUDA_ARRAY_DESCRIPTOR* desc,
                                         CUdeviceptr dptr, size_t Pitch) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || (CUDART_VERSION >= 3020 && CUDART_VERSION \
          < 4010) */

#if defined(CUDART_VERSION_INTERNAL) || CUDART_VERSION < 4000
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPopCurrent(CUcontext* pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || CUDART_VERSION < 4000 */

#if defined(CUDART_VERSION_INTERNAL)
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUdeviceptr srcDevice, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset,
                                 const void* srcHost, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUarray srcArray, size_t srcOffset,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset,
                                      const void* srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray,
                                      size_t srcOffset, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                      const void* srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
                                      CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy,
                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy,
                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc,
                               size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us,
                                size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui,
                                size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                 unsigned char uc, size_t Width,
                                 size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned short us, size_t Width,
                                  size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned int ui, size_t Width,
                                  size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy,
                                     CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int* priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void* userData,
                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void** kernelParams, void** extra) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource* resources,
                                        CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource* resources,
                                          CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams* paramArray,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif

CUresult cuProfilerInitialize(const char* configFile, const char* outputFile,
                              CUoutput_mode outputMode) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult cuProfilerStart(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult cuProfilerStop(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

//_ptds

extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src,
                                          size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyPeer_ptds(CUdeviceptr dstDevice,
                                              CUcontext dstContext,
                                              CUdeviceptr srcDevice,
                                              CUcontext srcContext,
                                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice,
                                                 const void* srcHost,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2_ptds(void* dstHost,
                                                 CUdeviceptr srcDevice,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice,
                                                 CUdeviceptr srcDevice,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD8_v2_ptds(CUdeviceptr dstDevice,
                                               unsigned char uc,
                                               unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD16_v2_ptds(CUdeviceptr dstDevice,
                                                unsigned short us,
                                                unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD32_v2_ptds(CUdeviceptr dstDevice,
                                                unsigned int ui,
                                                unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D8_v2_ptds(CUdeviceptr dstDevice,
                                                 unsigned int dstPitch,
                                                 unsigned char uc,
                                                 unsigned int Width,
                                                 unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D16_v2_ptds(CUdeviceptr dstDevice,
                                                  unsigned int dstPitch,
                                                  unsigned short us,
                                                  unsigned int Width,
                                                  unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D32_v2_ptds(CUdeviceptr dstDevice,
                                                  unsigned int dstPitch,
                                                  unsigned int ui,
                                                  unsigned int Width,
                                                  unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

//_ptsz
extern "C" CUresult CUDAAPI
cuMemcpy3DPeer_ptsz(const CUDA_MEMCPY3D_PEER* pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src,
                                               size_t ByteCount,
                                               CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyPeerAsync_ptsz(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
    CUcontext srcContext, size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoAAsync_v2_ptsz(CUarray dstArray,
                                                      size_t dstOffset,
                                                      const void* srcHost,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyAtoHAsync_v2_ptsz(void* dstHost,
                                                      CUarray srcArray,
                                                      size_t srcOffset,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                      const void* srcHost,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void* dstHost,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D* pCopy,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D* pCopy,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr dstDevice,
                                                 unsigned char uc, size_t N,
                                                 CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D8Async_ptsz(CUdeviceptr dstDevice,
                                                   size_t dstPitch,
                                                   unsigned char uc,
                                                   size_t Width, size_t Height,
                                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void** kernelParams, void** extra) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventRecord_ptsz(CUevent hEvent,
                                               CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWriteValue32_ptsz(CUstream stream,
                                                      CUdeviceptr addr,
                                                      cuuint32_t value,
                                                      unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWaitValue32_ptsz(CUstream stream,
                                                     CUdeviceptr addr,
                                                     cuuint32_t value,
                                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamBatchMemOp_ptsz(
    CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetPriority_ptsz(CUstream hStream,
                                                     int* priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetFlags_ptsz(CUstream hStream,
                                                  unsigned int* flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream hStream,
                                                   CUevent hEvent,
                                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamAddCallback_ptsz(CUstream hStream,
                                                     CUstreamCallback callback,
                                                     void* userData,
                                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamAttachMemAsync_ptsz(CUstream hStream,
                                                        CUdeviceptr dptr,
                                                        size_t length,
                                                        unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsMapResources_ptsz(
    unsigned int count, CUgraphicsResource* resources, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsUnmapResources_ptsz(
    unsigned int count, CUgraphicsResource* resources, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPrefetchAsync_ptsz(CUdeviceptr devPtr,
                                                    size_t count,
                                                    CUdevice dstDevice,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
