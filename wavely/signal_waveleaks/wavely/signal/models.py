from enum import Enum, unique


@unique
class FilterOutput(Enum):
    """An enumeration of the different filter types.

    Attributes:
        ba: Output of type numerator/denominator.
        sos: Output of type seconds-order-section

    """

    ba = "ba"
    sos = "sos"
