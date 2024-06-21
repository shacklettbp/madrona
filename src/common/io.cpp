/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/io.hpp>

#include <madrona/crash.hpp>
#include <madrona/memory.hpp>

#include <fstream>

namespace madrona {

char * readBinaryFile(const char *path,
                      size_t buffer_alignment,
                      size_t *out_num_bytes)
{
    // FIXME: look into platform specific alternatives for better
    // errors

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return nullptr;
    }

    size_t num_bytes = file.tellg();
    file.seekg(std::ios::beg);

    if (buffer_alignment < sizeof(void *)) {
        buffer_alignment = sizeof(void *);
    }

    size_t alloc_size = utils::roundUpPow2(num_bytes, buffer_alignment);

    char *data = (char *)rawAllocAligned(alloc_size, buffer_alignment);
    file.read(data, num_bytes);
    if (file.fail()) {
        rawDeallocAligned(data);
        return nullptr;
    }

    *out_num_bytes = num_bytes;
    return data;
}

}
