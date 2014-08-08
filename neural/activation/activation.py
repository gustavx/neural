
class activation
    """
    activation class provides the transfer fucntion and his first derivate
    for the activation potetial of the units. It takes as parameters:
    * cod       : the codomain (inf,sup) when corresponds.
    * steepness : the steepness of the curve when corresponds.
    """
    def __init__( _, codomain=(0.0,1.0), steepness=1.0):
        _.cod = codomain
        _.stp = steepness

    def function( _, X):
        NotImplementedError

    def derivate( _, X):
        NotImplementedError

