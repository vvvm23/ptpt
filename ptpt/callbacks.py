from enum import Enum, auto
# TODO: figure out enums with more data
# TODO: frequency /could/ be done as adding an additional function. pass a regular enum. we only trigger a frequency one at a frequency specified initially in the function.
# TODO: more of a design todo
class CallbackType(Enum):
    TrainStep           = auto()
    TestStep            = auto()
    TrainEpoch          = auto()
    TestEpoch           = auto()
    TrainDataExhaust    = auto()
    TestDataExhaust     = auto()
    FrequencyWallTime   = auto()
    FrequencyUpdates    = auto()
    ParameterUpdate     = auto()
    Termination         = auto()
    Start               = auto()
