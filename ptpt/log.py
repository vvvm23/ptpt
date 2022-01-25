import torch

import logging
from rich.logging import RichHandler

# TODO: needs way to set logging level globally
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.NOTSET,
    format=FORMAT, 
    datefmt="[%x | %X]", 
    handlers=[
        RichHandler(rich_tracebacks=True),
    ], 
)

logger = logging.getLogger("rich")

def wrap_log(f):
    noop = lambda *args, **kwargs: None
    return lambda f: f if torch.distributed.get_rank() == 0 else noop

debug = wrap_log(logger.debug)
info = wrap_log(logger.info)
warning = wrap_log(logger.warning)
error = wrap_log(logger.error)
critical = wrap_log(logger.critical)

