from neural import num


class creation:
    def do( _, N, M, inf=-0.1, sup=0.1):
        raise NotImplementedError

class random( creation):
    def do( _, N, M, inf=-0.1, sup=0.1):
        return num.random.uniform( inf, sup, (N, M))

class zeros( creation):
    def do( _, N, M, inf=0, sup=0):
        return num.zeros( (N, M) )

