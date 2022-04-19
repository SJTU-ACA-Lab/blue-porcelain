#include "mem.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "util.h"

using namespace gpgpu;

RAM::RAM(uint32_t page_size)
    : size_(0),
      page_bits_(log2ceil(page_size)),
      last_page_(nullptr),
      last_page_index_(0) {
  assert(ispow2(page_size));
}

RAM::~RAM() { this->clear(); }

void RAM::clear() {
  for (auto &page : pages_) {
    delete[] page.second;
  }
}

uint64_t RAM::size() const { return uint64_t(pages_.size()) << page_bits_; }

uint8_t *RAM::get(uint64_t address) const {
  uint32_t page_size = 1 << page_bits_;
  uint32_t page_offset = address & (page_size - 1);
  uint64_t page_index = address >> page_bits_;

  uint8_t *page;
  if (last_page_ && last_page_index_ == page_index) {
    page = last_page_;
  } else {
    auto it = pages_.find(page_index);
    if (it != pages_.end()) {
      page = it->second;
    } else {
      uint8_t *ptr = new uint8_t[page_size];
      // set uninitialized data to "baadf00d"
      for (uint32_t i = 0; i < page_size; ++i) {
        ptr[i] = (0xbaadf00d >> ((i & 0x3) * 8)) & 0xff;
      }
      pages_.emplace(page_index, ptr);
      page = ptr;
    }
    last_page_ = page;
    last_page_index_ = page_index;
  }

  return page + page_offset;
}

void RAM::read(void *data, uint64_t addr, uint64_t size) {
  uint64_t e = addr + (size - 1);
  assert(e >= addr);
  if (addr >= 0 && e <= 0xFFFFFFFF) {
    uint8_t *d = (uint8_t *)data;
    for (uint64_t i = 0; i < size; i++) {
      d[i] = *this->get(addr + i);
    }
  } else {
    std::cout << "lookup of 0x" << std::hex << addr << " failed.\n";
    throw BadAddress();
  }
}

void RAM::write(const void *data, uint64_t addr, uint64_t size) {
  uint64_t e = addr + (size - 1);
  assert(e >= addr);
  if (addr >= 0 && e <= 0xFFFFFFFF) {
    const uint8_t *d = (const uint8_t *)data;
    for (uint64_t i = 0; i < size; i++) {
      *this->get(addr + i) = d[i];
    }
  } else {
    std::cout << "lookup of 0x" << std::hex << addr << " failed.\n";
    throw BadAddress();
  }
}
