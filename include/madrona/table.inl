namespace madrona {

uint32_t Table::idToIndex(Entity e) const
{
    GenIndex idx = *(GenIndex *)id_to_idx_[e.id];

    if (idx.gen != e.gen) {
        return ~0u;
    }

    return idx.idx;
}

Table::Index Table::getIndex(Entity e) const
{
    return Index {
        idToIndex(e),
    };
}

void * Table::getValue(uint32_t column_idx, Table::Index idx)
{
    VirtualStore &col = columns_[column_idx];
    return col[idx.idx];
}

const void * Table::getValue(uint32_t column_idx, Table::Index idx) const
{
    const VirtualStore &col = columns_[column_idx];
    return col[idx.idx];
}

}
