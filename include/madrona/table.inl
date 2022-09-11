namespace madrona {

uint32_t Table::idToIndex(Entity e) const
{
    if (e.id >= num_ids_) {
        return ~0u;
    }

    GenIndex idx = *(GenIndex *)id_to_idx_[e.id];

    if (idx.gen != e.gen) {
        return ~0u;
    }

    return idx.idx;
}

Loc Table::getLoc(Entity e) const
{
    return Loc {
        idToIndex(e),
    };
}

void * Table::getValue(uint32_t column_idx, Loc loc)
{
    VirtualStore &col = columns_[column_idx];
    return col[loc.idx];
}

const void * Table::getValue(uint32_t column_idx, Loc loc) const
{
    const VirtualStore &col = columns_[column_idx];
    return col[loc.idx];
}

void * Table::data(uint32_t col_idx)
{
    return columns_[col_idx][0];
}

const void * Table::data(uint32_t col_idx) const
{
    return columns_[col_idx][0];
}

}
