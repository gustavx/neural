import neural

class topology:
    def __init__( _, layers=[], connection=None, excitation=None, activation=None, correction=None, estimation=None):
        _.layers = layers
        _.connection = connection
        _.excitation = excitation
        _.activation = activation
        _.correction = correction
        _.estimation = estimation

