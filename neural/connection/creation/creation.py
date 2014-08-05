from neural import num


class creation:
    """
    creation class implements the 'do' method for the weight's creation.
    It takes two natural numbers for the dimension and two real numbers
    as a bound for the weights values.
    - N   : Size of the pre-synaptic layer.
    - M   : Size of the pos-synaptic layer.
    - inf : Inferion bound for the values.
    - sup : Superior bound for the values.
    It always must return an array of NxM of real values.
    """
    def do( _, N, M, inf=-0.1, sup=0.1):
        raise NotImplementedError

class random( creation):
    def do( _, N, M, inf=-0.1, sup=0.1):
        return num.random.uniform( inf, sup, (N, M))

class zeros( creation):
    def do( _, N, M, inf=0, sup=0):
        return num.zeros( (N, M) )

