import logging
from typing import Optional, Dict, Any

import wandb

logger = logging.getLogger(__name__)


def init_wandb(
        use_wand: bool,
        project: str,
        run_name: str,
        tags,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
        ) -> Optional[wandb.wandb_sdk.wandb_run.Run]:
    """
    Initialize a Weights & Biases (wandb) run if use_wand is True.

    Args:
        use_wand (bool): Whether to initialize wandb.
        project (str): The name of the wandb project.
        run_name (str): The name of the wandb run.
        tags (list): List of tags for the wandb run.
        entity (Optional[str]): The wandb entity (username or team name).
        config (Optional[Dict[str, Any]]): Configuration dictionary to log with wandb.

    Returns:
        Optional[wandb.wandb_sdk.wandb_run.Run]: The initialized wandb run, or
        None if logging is disabled or initialization failed.
    """
    if not use_wand:
        return None

    try:
        wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            config=config
        )
        logger.info(f"wandb run initialized: {wandb_run.url}")
        return wandb_run
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None
