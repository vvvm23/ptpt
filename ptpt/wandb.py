import wandb
import dataclasses

@dataclasses.dataclass()
class WandbConfig:
    project: str = None
    entity: str = None
    name: str = None
    config: dict = None
    log_net: bool = False
    log_metrics: bool = True
