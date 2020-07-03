from typing import NamedTuple, TypedDict
from prodict import Prodict
from nptyping import NDArray, Float64, UInt8
import numpy as np

class Inventory(Prodict):
    name: str
    price: float


def get_inventory(name: str, price: float):
    return Inventory(name=name, price=price)


inventory = get_inventory('hammer', 2.3)
print(inventory.name)
print(inventory.price)

class Mytest(Prodict):
    key1: str
    key2: float

    def __init__(self,
                 key1arg: str,
                 key2arg: float):
        super(Mytest, self).__init__(
            key1=key1arg,
            key2=key2arg
        )

class MySecondtest(Prodict):
    key1: str
    key2: float

    def __init__(self, *args, **kwargs):
        super(MySecondtest, self).__init__(*args, **kwargs)

test_inst = MySecondtest()



test = Mytest(key1arg='key1', key2arg=2.2)

print(test.key1)
print(test.key2)

def testfun(arr: NDArray[Float64]):
    print(arr)

myarr = np.array([1., 2.])
print(myarr.dtype)

testfun(arr=myarr)


