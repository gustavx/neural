
from neural import num
from activation import activation


class step( activation):

    def function( _, X):
        return num.where( X<((_.cod[0]+_.cod[1])/2.0), _.cod[0], _.cod[1])

    def derivate( _, X):
        return num.ones_like( X)

