/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
namespace madrona {

void * Table::getValue(uint32_t column_idx, uint32_t row)
{
    return (char *)columns_[column_idx] +
        uint64_t(row) * uint64_t(bytes_per_column_[column_idx]);
}

const void * Table::getValue(uint32_t column_idx, uint32_t row) const
{
    return (char *)columns_[column_idx] +
        uint64_t(row) * uint64_t(bytes_per_column_[column_idx]);
}

void * Table::data(uint32_t col_idx)
{
    return columns_[col_idx];
}

const void * Table::data(uint32_t col_idx) const
{
    return columns_[col_idx];
}

}
