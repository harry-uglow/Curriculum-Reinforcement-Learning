from torch import Tensor


class TupleTensor:

    def __init__(self, tensor0, tensor1):
        assert type(tensor0) is Tensor
        assert type(tensor1) is Tensor
        self.fst = tensor0
        self.snd = tensor1

    def __getattribute__(self, name: str):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            attr = object.__getattribute__(self.fst, name)
            print("Tensor attribute: " + attr.__name__)
            if hasattr(attr, '__call__'):
                def wrapper_func(*args, **kwargs):
                    args0 = [*args]
                    args1 = [*args]
                    for i in range(len(args)):
                        if type(args[i]) is TupleTensor:
                            args0[i] = args[i].fst
                            args1[i] = args[i].snd

                    res0 = object.__getattribute__(self.fst, name)(*args0, **kwargs)
                    res1 = object.__getattribute__(self.snd, name)(*args1, **kwargs)
                    return TupleTensor(res0, res1)
                return wrapper_func
            return attr

    def __getitem__(self, key):
        result = TupleTensor(self.fst[key], self.snd[key])
        return result
