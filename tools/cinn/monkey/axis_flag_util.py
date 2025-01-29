
def IsLhsGreaterThanRhs(lhs, rhs):
    assert len(lhs) == len(rhs)
    if not _AllLhsGreaterEqualRhs(lhs, rhs):
        return False
    for i in range(len(lhs)):
        if lhs[i] > rhs[i]:
            return True
    return False

def _AllLhsGreaterEqualRhs(lhs, rhs):
    assert len(lhs) == len(rhs)
    for i in range(len(lhs)):
        if not (lhs[i] >= rhs[i]):
            return False
    return True