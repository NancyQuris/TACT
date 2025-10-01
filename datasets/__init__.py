from .birdcalls import birdcalls
from .camelyon import camelyon
from .civil import civil
from .imagenet import imagenet, imagenet_r, imagenetv2

from .birdcalls import NUM_CLASSES as birdcalls_n_class
from .camelyon import NUM_CLASSES as camelyon_n_class
from .civil import NUM_CLASSES as civil_n_class
from .imagenet import NUM_CLASSES as imagenet_n_class

from .birdcalls import PRETRAINED as birdcalls_pretrained
from .camelyon import PRETRAINED as camelyon_pretrained
from .civil import PRETRAINED as civil_pretrained
from .imagenet import PRETRAINED as imagenet_pretrained 


__all__ = [birdcalls, civil, camelyon, imagenet, imagenet_r, imagenetv2]
