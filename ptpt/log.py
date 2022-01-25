import logging
import accelerate
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

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

def _wrap_fn(f, accelerator):
    def new_fn(*args, **kwargs):
        if accelerator.is_local_main_process:
            f(*args, **kwargs)
    return new_fn

def wrap_log(accelerator):
    globals()['debug'] = _wrap_fn(logger.debug, accelerator)
    globals()['info'] = _wrap_fn(logger.info, accelerator)
    globals()['warning'] = _wrap_fn(logger.warning, accelerator)
    globals()['error'] = _wrap_fn(logger.error, accelerator)
    globals()['critical'] = _wrap_fn(logger.critical, accelerator)

