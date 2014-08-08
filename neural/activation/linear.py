from neural import num

from activation import activation

class linear( activation):
    def function( _, X):
        return X

    def derivate( _, X):
        return num.ones_like( X)

