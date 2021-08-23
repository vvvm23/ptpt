from enum import Enum, auto
# TODO: figure out enums with more data
# TODO: frequency /could/ be done as adding an additional function. pass a regular enum. we only trigger a frequency one at a frequency specified initially in the function.
# TODO: more of a design todo
class CallbackType(Enum):
    TrainStep           = auto()
    EvalStep            = auto()
    TrainEpoch          = auto()
    EvalEpoch           = auto()
    TrainDataExhaust    = auto()
    EvalDataExhaust     = auto()
    ParameterUpdate     = auto()
    Termination         = auto()
    Start               = auto()

class CallbackCounter:
    def __init__(self, frequency):
        self.frequency = frequency
        self.counter = 0

    def check(self):
        self.counter += 1
        if self.counter % self.frequency:
            return False
        self.counter = 0
        return True
