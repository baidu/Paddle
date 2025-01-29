def HashCombine(lhs: int, rhs: int):
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2)
    return lhs