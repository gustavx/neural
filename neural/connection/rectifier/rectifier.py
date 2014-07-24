from neural import num


class rectifier:
    def do( _, W):
        return W

class null( rectifier):
    def do( _, W):
        return W

class decay( rectifier):
    def __init__( _, a=0.5):
        _.a = a

    def do( _, W):
        return (1.0-_.a)*W

class normalize( rectifier):
    def do( _, W):
        return W/num.sum( W)

