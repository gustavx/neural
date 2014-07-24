from neural import num


class affinity:
    def do( _, W):
        return W

class full( affinity):
    def do( _, W):
        return W

class single( affinity):
    def do( _, W):
        return num.diag( num.diag( W))

