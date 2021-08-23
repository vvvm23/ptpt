from enum import Enum, auto
class CallbackType(Enum):
    TrainStep           = auto()
    EvalStep            = auto()
    TrainEpoch          = auto()
    EvalEpoch           = auto()
    TrainDataExhaust    = auto()
    EvalDataExhaust     = auto()
    ParameterUpdate     = auto()
    SaveCheckpoint      = auto()
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
