from contextlib import contextmanager

@contextmanager
def train(model, mode=True):
    is_training = model.training
    model.train(mode=mode)
    try:
        yield model
    finally:
        model.train(mode=is_training)

def eval(model):
    return train(model, mode=False)
