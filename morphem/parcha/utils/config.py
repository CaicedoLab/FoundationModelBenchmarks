# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
import os

from omegaconf import OmegaConf

import dinov2.distributed as distributed
from dinov2.logging import setup_logging
from dinov2.utils import utils
from dinov2.configs.config import Dinov2Config

logger = logging.getLogger("dinov2")


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.total_batch_size / 1024.0) #1024.0
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    elif cfg.optim.scaling_rule is None:
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        logger.info(f"base learning rate; base: {base_lr}")    
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    
    args.output_dir = os.path.abspath(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, cfg.train.name)
    # default_cfg = OmegaConf.create(dinov2_default_config)
    default_cfg:Dinov2Config = OmegaConf.structured(Dinov2Config)
    cfg: Dinov2Config = OmegaConf.merge(default_cfg, cfg)
    cfg.train.output_dir = args.output_dir
    return cfg 


def default_setup(args):
    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args)
    apply_scaling_rules_to_cfg(cfg)
    return cfg
