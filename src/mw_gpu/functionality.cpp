/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <cstdint>

#include "job.hpp"

void set_val(float *data, uint32_t idx, float v)
{
    data[idx] = v;
}
