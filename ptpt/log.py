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
