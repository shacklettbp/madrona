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
