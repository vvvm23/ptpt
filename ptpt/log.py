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
    return f if accelerator.is_local_main_process else noop

def wrap_log(accelerator):
    noop = lambda *args, **kwargs: None
    # wrap_fn = lambda f: f if accelerator.is_local_main_process else noop

    globals()['debug'] = _wrap_fn(logger.debug, accelerator)
    globals()['info'] = _wrap_fn(logger.info, accelerator)
    globals()['warning'] = _wrap_fn(logger.warning, accelerator)
    globals()['error'] = _wrap_fn(logger.error, accelerator)
    globals()['critical'] = _wrap_fn(logger.critical, accelerator)

