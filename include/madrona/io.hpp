/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/types.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>

namespace madrona {

char * readBinaryFile(const char *path,
                      size_t buffer_alignment,
                      size_t *out_num_bytes);

}
