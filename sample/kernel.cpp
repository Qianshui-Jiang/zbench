/*
 * Copyright (c) 2020-2023, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cm/cm.h>

#ifdef SHIM
#include "shim_support.h"
#else
#define SHIM_API_EXPORT
#endif
extern "C" SHIM_API_EXPORT void vector_add(svmptr_t, svmptr_t, svmptr_t);

// shim layer (CM kernel, OpenCL runtime, GPU)
#ifdef SHIM
EXPORT_SIGNATURE(vector_add);
#endif

#if defined(SHIM) || defined(CMRT_EMU)
#define ATTR
#else
#define ATTR [[type("svmptr_t")]]
#endif

#define SZ 16

_GENX_MAIN_ void vector_add(svmptr_t ibuf1 ATTR, svmptr_t ibuf2 ATTR,
                            svmptr_t obuf ATTR) {
  vector<int, SZ> ivector1;
  vector<int, SZ> ivector2;
  vector<int, SZ> ovector;

  // printf("gid(0)=%d, gid(1)=%d, lid(0)=%d, lid(1)=%d\n", cm_group_id(0),
  // cm_group_id(1), cm_local_id(0), cm_local_id(1));
  unsigned offset = sizeof(int) * SZ * cm_group_id(0);

  //
  // read-in the arguments
  // make use of LSC loads/stores where they are available
  // otherwise we have to use the legacy translation for Battlemage and onward
#ifdef CM_HAS_LSC
  ivector1 = cm_ptr_load<int, SZ>(reinterpret_cast<int *>(ibuf1), offset);
  ivector2 = cm_ptr_load<int, SZ>(reinterpret_cast<int *>(ibuf2), offset);
#else
  cm_svm_block_read(ibuf1 + offset, ivector1);
  cm_svm_block_read(ibuf2 + offset, ivector2);
#endif // CM_HAS_LSC

  // perform addition
  ovector = ivector1 + ivector2;

  // write-out the results
#ifdef CM_HAS_LSC
  cm_ptr_store(reinterpret_cast<int *>(obuf), offset, ovector);
#else
  cm_svm_block_write(obuf + offset, ovector);
#endif // CM_HAS_LSC
}
