#pragma once

#include <GPU_config.h>
#include <stdint.h>
#include <string.h>
#include <util.h>

#include <bitset>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "kernel.h"
#include "vector_types.h"

namespace gpgpu {

// float and int rounding_mode
//  0x0 RN  尾数舍入到最接近的偶数
//  0x1 RM  尾数向负无穷大舍入
//  0x2 RP	  尾数向正无穷大舍入
// 0x3 RZ  尾数向零舍入
enum rounding_mode { RN_OPTION, RM_OPTION, RP_OPTION, RZ_OPTION };

enum oprand_data_type_t {
  INVALID = 0,
  REG32,
  IMM32,
  CMADDR,
  REG64,
  REG128,
  CMADDR64,
  IMM64,
  PR
};

typedef std::bitset<WARP_SIZE> ThreadMask;
typedef std::bitset<WARP_NUM> WarpInfoMask;

typedef unsigned long long new_addr_type;

union sass_reg_t {
  sass_reg_t() {
    bits.ms = 0;
    bits.ls = 0;
    u128.low = 0;
    u128.lowest = 0;
    u128.highest = 0;
    u128.high = 0;
    s8 = 0;
    s16 = 0;
    s32 = 0;
    s64 = 0;
    u8 = 0;
    u16 = 0;
    u64 = 0;
    f16 = 0;
    f32 = 0;
    f64 = 0;
    pred = 0;
  }
  sass_reg_t(unsigned x) {
    bits.ms = 0;
    bits.ls = 0;
    u128.low = 0;
    u128.lowest = 0;
    u128.highest = 0;
    u128.high = 0;
    s8 = 0;
    s16 = 0;
    s32 = 0;
    s64 = 0;
    u8 = 0;
    u16 = 0;
    u64 = 0;
    f16 = 0;
    f32 = 0;
    f64 = 0;
    pred = 0;
    u32 = x;
  }
  operator unsigned int() { return u32; }
  operator unsigned short() { return u16; }
  operator unsigned char() { return u8; }
  operator unsigned long long() { return u64; }

  void mask_and(unsigned ms, unsigned ls) {
    bits.ms &= ms;
    bits.ls &= ls;
  }

  void mask_or(unsigned ms, unsigned ls) {
    bits.ms |= ms;
    bits.ls |= ls;
  }
  int get_bit(unsigned bit) {
    if (bit < 32)
      return (bits.ls >> bit) & 1;
    else
      return (bits.ms >> (bit - 32)) & 1;
  }

  signed char s8;
  signed short s16;
  signed int s32;
  signed long long s64;
  unsigned char u8;
  unsigned short u16;
  unsigned int u32;
  unsigned long long u64;
// gcc 4.7.0
#if GCC_VERSION >= 40700
  half f16;
#else
  float f16;
#endif
  float f32;
  double f64;
  struct {
    unsigned ls;
    unsigned ms;
  } bits;
  struct {
    unsigned int lowest;
    unsigned int low;
    unsigned int high;
    unsigned int highest;
  } u128;
  unsigned pred : 1;
};

union SassCodeType {
  struct {
    uint64_t L;
    uint64_t H;
  } b64;

  __uint128_t bin;
  SassCodeType() : bin(0) {}
};

struct WarpInfoType {
  u_int64_t pc;
  u_int32_t wid_in_cta;
  dim3 tid[WARP_SIZE];
  dim3 ctaid;
  std::bitset<WARP_SIZE> thread_active;
  // Debug Infomation
  KernelInfoType* kernel;
  WarpInfoType() : pc(0), tid(), ctaid(), thread_active(0) {}
};

inline std::ostream& operator<<(std::ostream& os, const SassCodeType& sass) {
  os << "SASS CODE: 0x";
  os << std::setw(16) << std::setfill('0')
     << std::setiosflags(std::ios::uppercase) << std::hex << sass.b64.H;
  os << std::setw(16) << std::setfill('0')
     << std::setiosflags(std::ios::uppercase) << std::hex << sass.b64.L;
  os << std::dec;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const dim3& dim) {
  os << "(";
  os << std::setw(3);
  os << dim.x;
  os << ",";
  os << std::setw(3);
  os << dim.y;
  os << ",";
  os << std::setw(3);
  os << dim.z;
  os << ")";
  return os;
}

}  // namespace gpgpu