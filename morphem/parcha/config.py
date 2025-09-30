from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

@dataclass
class MixedPrecisionDetail:
    param_dtype: str
    reduce_dtype: str
    buffer_dtype: str

@dataclass
class ModulePrecision:
    sharding_strategy: str
    mixed_precision: MixedPrecisionDetail

@dataclass
class ArchPrecision:
    backbone: ModulePrecision
    dino_head: ModulePrecision
    ibot_head: ModulePrecision

@dataclass
class ModelConfig:
    WEIGHTS: str = ""

@dataclass
class ComputePrecisionConfig:
    grad_scaler: bool = True
    teacher: ArchPrecision = field(default_factory=lambda: ArchPrecision(
        backbone=ModulePrecision("SHARD_GRAD_OP", MixedPrecisionDetail("fp16", "fp16", "fp32")),
        dino_head=ModulePrecision("SHARD_GRAD_OP", MixedPrecisionDetail("fp16", "fp16", "fp32")),
        ibot_head=ModulePrecision("SHARD_GRAD_OP", MixedPrecisionDetail("fp16", "fp16", "fp32")),
    ))
    student: ArchPrecision = field(default_factory=lambda: ArchPrecision(
        backbone=ModulePrecision("SHARD_GRAD_OP", MixedPrecisionDetail("fp16", "fp16", "fp32")),
        dino_head=ModulePrecision("SHARD_GRAD_OP", MixedPrecisionDetail("fp16", "fp32", "fp32")),
        ibot_head=ModulePrecision("SHARD_GRAD_OP", MixedPrecisionDetail("fp16", "fp32", "fp32")),
    ))

class WandbMode(str, Enum):
    disabled = "disabled"
    enabled = None

@dataclass
class TrainConfig:
    name: str = 'unnamed'
    batch_size_per_gpu: Optional[int] = 64 # Autoset when total_batch_size is set.
    dataset_path: str =  "pathy" 
    output_dir: str = "."
    saveckp_freq: int = 20
    seed: int = 42
    num_workers: int = 10
    wandb_mode: WandbMode = WandbMode.enabled
    dataset_config: Optional[str] = None # /scr/data/CHAMMI/multi_channel_chammi_metadata.csv 

class Arch(str, Enum):
    vit_small_single_channel = "vit_small_single_channel" 
    vit_base_single_channel  = "vit_base_single_channel" 
    vit_small_ngram = "vit_small_ngram"
    vit_medium_single_channel = "vit_medium_single_channel"

@dataclass
class ModelConfig:
    arch: Arch = Arch.vit_small_single_channel
    drop_path_rate: float = 0.0 # 0.3, needs to be 0 unless embed dim = 768
    layerscale: float = 1.0e-05
    drop_path_uniform: bool = True
    pretrained_weights: str = ""
    ffn_layer: str = "swiglufused"
    block_chunks: int = 0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    num_register_tokens: int = 2
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1


@dataclass
class OptimConfig:
    epochs: int = 100
    weight_decay: float = 0.04
    weight_decay_end: float = 0.2
    base_lr: float = 0.002
    lr: float = 0.0 # auto-gen with respect to scaling_rule
    warmup_epochs: int = 10
    min_lr: float = 1.0e-06
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    layerwise_decay: float = 1.0
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    min_grad: float = 0.5
    min_its: int = 2000 # Applies min grad run stopping after this many its

@dataclass
class CropsConfig:
    global_crops_scale: List[float] = field(default_factory=lambda: [0.32, 1.0])
    local_crops_number: int = 4
    local_crops_scale: List[float] = field(default_factory=lambda: [0.05, 0.32])
    global_crops_size: int = 224
    local_crops_size: int = 96

@dataclass
class EvaluationConfig:
    eval_period_iterations: int = 12500

@dataclass
class Dinov2Config:
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    compute_precision: ComputePrecisionConfig = field(default_factory=ComputePrecisionConfig)
    dino: DinoConfig = field(default_factory=DinoConfig)
    ibot: IbotConfig = field(default_factory=IbotConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    crops: CropsConfig = field(default_factory=CropsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)