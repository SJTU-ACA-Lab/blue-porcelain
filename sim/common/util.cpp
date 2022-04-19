#include "util.h"
#include <string.h>

// return file extension
const char* fileExtension(const char* filepath) {
  const char* ext = strrchr(filepath, '.');
  if (ext == NULL || ext == filepath) return "";
  return ext + 1;
}

uint64_t line_size_based_tag_func(uint64_t address, uint64_t line_size) {
  // gives the tag for an address based on a given line size
  return address & ~(line_size - 1);
}