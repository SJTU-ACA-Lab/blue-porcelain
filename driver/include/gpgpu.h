// This file created from vortex.h distributed with Vortex v0.2.3
// Changes Copyright 2022, ACA Lab of SJTU

/**
 * gpgpu.h
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

#ifndef __GPGPU_DRIVER_H__
#define __GPGPU_DRIVER_H__

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <stdio.h>
#include "vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* gpgpu_device_h;

#define MAX_TIMEOUT (60 * 60 * 1000)  // 1hr

// open the device and connect to it
int gpgpu_dev_open(gpgpu_device_h* hdevice);

// Close the device when all the operations are done
int gpgpu_dev_close(gpgpu_device_h hdevice);

// Copy bytes from buffer to device local memory
int gpgpu_copy_to_dev(gpgpu_device_h hdevice, void* data, uint64_t dev_maddr,
                      uint64_t size, uint64_t src_offset);

// Copy bytes from device local memory to buffer
int gpgpu_copy_from_dev(gpgpu_device_h hdevice, void* data, uint64_t dev_maddr,
                        uint64_t size, uint64_t dest_offset);

// Configure the kernel to device
int gpgpu_push_command(gpgpu_device_h hdevice, void* cmd_q, int size);

// Start device execution
int gpgpu_start(gpgpu_device_h hdevice);

// Wait for device ready with milliseconds timeout
int gpgpu_ready_wait(gpgpu_device_h hdevice, uint64_t timeout);

#ifdef __cplusplus
}
#endif

#endif  // __GPGPU_DRIVER_H__
