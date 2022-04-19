#pragma once
#include <cstdint>
#include "types.h"
using namespace gpgpu;
#define FSEL 0x008
#define FADD 0x021
#define FFMA 0x023
#define FMNMX 0x009
#define FMUL 0x020
#define FSETP 0x00b
#define HADD2 0x230
#define IABS 0x013
#define IADD3 0x010
#define IMAD 0x024
#define IMAD_WIDE 0x025
#define PRMT 0x016
#define IMNMX 0x017
#define ISETP 0x00c
#define LEA 0x011
#define LOP3 0x012
#define SHF 0x019
#define MOV 0x002
#define LD 0x180
#define LDG 0x181
#define LDC 0x182
#define LDL 0x183
#define LDS 0x184
#define ST 0x185
#define STG 0x186
#define STL 0x187
#define STS 0x188
#define YIELD 0x146
#define NOPI 0x118
#define BRA 0x147
#define EXIT 0x14d
#define BMOV 0x155
#define BSSY 0x145
#define BSYNC 0x141
#define CALL_REL 0x144
#define CALL_ABS 0x143
#define RET 0x150
#define BREAK 0x142
#define CS2R 0x005
#define S2R 0x119
#define SHFL 0x189
#define SEL 0x007
#define PLOP3 0x01c
#define BAR 0x11d
#define MUFU 0x108
#define FCHK 0x102
#define DADD 0x029
#define DMUL 0x028
#define DFMA 0x02b
#define DSETP 0x02a
#define I2F 0x106
#define I2F_64 0x112
#define F2I 0x105
#define F2I_64 0x111
#define P2R 0x003
#define F2F 0x110
#define LEPC 0x14e
#define R2P 0x004
#define BMSK 0x01b
#define MEMBAR 0x192
#define WARPSYNC 0x148
#define ATOMG 0x1a8
// 以下尚未修改
#define F2F_R 0x310
#define F2F_C 0xb10

// SR
#define SR_TID_X 0x21
#define SR_TID_Y 0x22
#define SR_TID_Z 0x23
#define SR_CTID_X 0x25
#define SR_CTID_Y 0x26
#define SR_CTID_Z 0x27
#define SR_Z 0x0

constexpr uint32_t opcode_mask = 0x1ff;
constexpr uint32_t opcode_type_mask = 0x7;
constexpr uint32_t predicate_mask = 0x7;
constexpr uint32_t reg_mask = 0xff;
constexpr uint32_t local_mask = 0xfffff;
constexpr uint32_t control_code_mask = 0xffffff;
constexpr uint32_t csr_mask = 0x1ff;
constexpr int pred_shf = 12;
constexpr int opcode_type_shf = 9;
constexpr int rd_shf = 16;
constexpr int rs0_shf = 24;
constexpr int local_shf = 8;
constexpr int ctrl_code_shf = 40;
constexpr int immed_shf = 32;
constexpr int rs0neg_shf = 8;
constexpr int rs1neg_shf = 63;
constexpr int rs0abs_shf = 9;
constexpr int rs1abs_shf = 62;
constexpr int dtype_shf = 9;
constexpr int shf_dytpe_shf = 8;
constexpr int X_shf = 10;
constexpr int rs2neg_shf = 11;
constexpr int rs2abs_shf = 10;

constexpr int shfpos_shf = 12;
constexpr int shfhi_shf = 16;
constexpr int leahi_shf = 16;
constexpr int leaimm_shf = 11;
constexpr int leasx32_shf = 9;
constexpr int ps1_shf = 13;
constexpr int pd0_shf = 17;
constexpr int pd1_shf = 20;
constexpr int ps0_shf = 23;
constexpr int E_shf = 8;
constexpr int scope_shf = 13;
constexpr int strong_shf = 15;
constexpr int cache_shf = 20;
constexpr int csr_shf = 8;
constexpr int src_dtype_shf = 20;
constexpr int dest_dtype_shf = 11;
constexpr int rnd_shf = 14;
constexpr int ftz_shf = 16;
constexpr int ntz_shf = 13;
constexpr int dsigned_shf = 8;
constexpr int dwidth_shf = 11;

constexpr int rs1_shf = 32;
constexpr int cm_shf = 38;
// constexpr uint32_t reg_mask = 0xff;
// constexpr int immed_shf = 32;
constexpr uint32_t immed_mask = 0xffffffff;
constexpr uint32_t const_mem_addr_mask = 0xfffff;
