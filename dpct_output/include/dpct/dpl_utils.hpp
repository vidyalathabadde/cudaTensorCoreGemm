//==---- dpl_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_DPL_UTILS_HPP__
#define __DPCT_DPL_UTILS_HPP__

#define ONEDPL_USE_DPCPP_BACKEND 1
#define __USE_DPCT 1

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>

#include "compat_service.hpp"

#include "dpl_extras/memory.h"
#include "dpl_extras/algorithm.h"
#include "dpl_extras/numeric.h"
#include "dpl_extras/iterators.h"
#include "dpl_extras/vector.h"
#include "dpl_extras/dpcpp_extensions.h"

// Only include iterator adaptor (and therefore boost) if necessary
#ifdef ITERATOR_ADAPTOR_REQUIRED
#include "dpl_extras/iterator_adaptor.h"
#endif // ITERATOR_ADAPTOR_REQUIRED

#endif // __DPCT_DPL_UTILS_HPP__
