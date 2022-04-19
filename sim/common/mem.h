#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace gpgpu {
struct BadAddress {};

class MemDevice {
 public:
  virtual ~MemDevice() {}
  virtual uint64_t size() const = 0;
  virtual void read(void *data, uint64_t addr, uint64_t size) = 0;
  virtual void write(const void *data, uint64_t addr, uint64_t size) = 0;
};

class RAM : public MemDevice {
 public:
  RAM(uint32_t page_size);
  ~RAM();

  void clear();

  uint64_t size() const override;

  void read(void *data, uint64_t addr, uint64_t size) override;
  void write(const void *data, uint64_t addr, uint64_t size) override;

  uint8_t &operator[](uint64_t address) { return *this->get(address); }

  const uint8_t &operator[](uint64_t address) const {
    return *this->get(address);
  }

 private:
  uint8_t *get(uint64_t address) const;

  uint64_t size_;
  uint32_t page_bits_;
  mutable std::unordered_map<uint64_t, uint8_t *> pages_;
  mutable uint8_t *last_page_;
  mutable uint64_t last_page_index_;
};

}  // namespace gpgpu