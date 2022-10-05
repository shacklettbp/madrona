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
    VirtualStore &col = columns_[column_idx];
    return col[row];
}

const void * Table::getValue(uint32_t column_idx, uint32_t row) const
{
    const VirtualStore &col = columns_[column_idx];
    return col[row];
}

void * Table::data(uint32_t col_idx)
{
    return columns_[col_idx].data();
}

const void * Table::data(uint32_t col_idx) const
{
    return columns_[col_idx].data();
}

}
