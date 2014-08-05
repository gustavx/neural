
from connection import connectio

import creation as neural_creation
import affinity as neural_affinity
import rectifier as neural_rectifier


class full( connection):
    """
    full class creates fully connected, random initialized weights.
    It has no rectification.
    """
    def __init__( _):
        _.creation = neural_creation.random()
        _-affinity = neural_affinity.full()
        _.rectifier = neural_rectifier.null()

