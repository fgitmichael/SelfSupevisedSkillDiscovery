from typing import Iterable

def len_iterable(i: Iterable):
    return sum([1 for el in i])