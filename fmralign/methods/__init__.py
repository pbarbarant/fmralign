from .identity import Identity
from .optimal_transport import OptimalTransport, SparseUOT
from .procrustes import Procrustes
from .ridge import RidgeAlignment
from .srm import DetSRM

__all__ = [
    "Identity",
    "OptimalTransport",
    "SparseUOT",
    "Procrustes",
    "RidgeAlignment",
    "DetSRM",
]
