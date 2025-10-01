from .tact import TACT, TACT_adapt
from .t3a import T3A
from.lame import LAME
from .foa import FOA
from .shot import SHOT
from .tent import Tent
from .sar import SAR
from .deyo import DeYO
from .tast import TAST
from .tsd import TSD
from .pasle import PASLE

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


