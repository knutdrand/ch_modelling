
registry = {}

def register_model(cls):
    registry[cls.__name__] = cls
    return cls
