from __future__ import absolute_import
import torch
from itertools import izip


class TorchTuple(object):
    pass


class TupleTensor(TorchTuple):

    def __init__(self, tensor0, tensor1):
        assert type(tensor0) is torch.Tensor
        assert type(tensor1) is torch.Tensor
        self.items = [tensor0, tensor1]

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            attr = object.__getattribute__(self.items[0], name)
            if hasattr(attr, u'__call__'):
                def wrapper_func(*args, **kwargs):
                    args_lists = []
                    for i, item in enumerate(self.items):
                        args_list = []
                        for arg in args:
                            to_add = arg.items[i] if issubclass(type(arg), TorchTuple) else arg
                            if to_add is not None:
                                args_list += [to_add]
                        args_lists += [args_list]
                    ress = [object.__getattribute__(item, name)(*args_list, **kwargs)
                            for args_list, item in izip(args_lists, self.items)]
                    if all([type(res) is torch.Size for res in ress]):
                        return TupleSize(*ress)
                    return TupleTensor(*ress)
                return wrapper_func
            return attr

    def __getitem__(self, key):
        result = TupleTensor(*[item[key] for item in self.items])
        return result


class TupleSize(TorchTuple):
    def __init__(self, size0, size1):
        self.items = [size0, size1]

    def __getitem__(self, key):
        result = TupleSize(*[item[key] for item in self.items])
        return result

    def __iter__(self):
        if len(self.items[0]) > len(self.items[1]):
            self.items[1] = self.items[1] + [None]*(len(self.items[0]) - len(self.items[1]))
        elif len(self.items[0]) < len(self.items[1]):
            self.items[0] = self.items[0] + [None]*(len(self.items[1]) - len(self.items[0]))

        return (TupleSize(a, b) for a, b in izip(self.items[0], self.items[1]))
