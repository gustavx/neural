from neural import num


class affinity:
    """
    affiniy class determines the connectivity between layers.
    If it is required that two layers limite their connections the 'do'
    method will zerod the appropiated weights.
    """
    def do( _, W):
        return W

class full( affinity):
    def do( _, W):
        return W

class single( affinity):
    def do( _, W):
        """Returns a matrix of the same dimension of W but only with the values of his diagonal."""
        return num.diag( num.diag( W))

