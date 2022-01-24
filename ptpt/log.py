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

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

def wrap_log(accelerator):
    global debug, info, warning, error, critical
    noop = lambda *args, **kwargs: None
    wrapping_fn = lambda f: f if accelerator.is_local_main_process else noop

    debug = wrapping_fn(debug)
    info = wrapping_fn(info)
    warning = wrapping_fn(warning)
    error = wrapping_fn(error)
    critical = wrapping_fn(critical)
