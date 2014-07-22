import creation as neural_creation
import affinity as neural_affinity
import rectifier as neural_rectifier


class connection:
    """
    connection class holds the methods for the creation and rectification of
    weights between layers. It depends on other three types of objects:
    * creation  : For the initialization of the weights (zeros, random, etc).
    * affinity  : For the connectivity of the weights (full, single, etc).
    * rectifier : For the rectification of the weights (decay, normal, etc).
    """
    def __init__( _, creation  = neural_creation.creation(),
                     affinity  = neural_affinity.affinity(), 
                     rectifier = neural_rectifier.rectifier()
                ):
        _.creation  = creation
        _.affinity  = affinity
        _.rectifier = rectifier

    def weights( _, N, M):
        """Returns an array of NxM for weights."""
        W = _.creation.do( N, M)
        W = _.affinity.do( W)
        return W

    def rectify( _, W):
        """Returns a rectified copy of the weights W."""
        rW = _.rectifier( W)
        rW = _.affinity( rW)
        return rW

