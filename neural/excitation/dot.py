from neural import num
from excitation import excitation

class dot( excitation):
    """
    dot excitation class uses the dot product between de vector units and
    the weights matrix.
    """
    def forw( _, X, W):
        return num.dot( X, W)

    def back( _, W, Y):
        return num.dot( Y, W.T)

