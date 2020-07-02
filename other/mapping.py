from typing import NamedTuple
from prodict import Prodict

class Point(NamedTuple):
    x: int
    y: int=1

Point(3)
pass


class Inv(Prodict):
    name
