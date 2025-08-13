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

@dataclass
class DinoConfig:
    loss_weight: float = 1.0
    head_n_prototypes: int = 65536
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048
    koleo_loss_weight: float = 0.1

@dataclass
class IbotConfig:
    loss_weight: float = 1.0
    mask_sample_probability: float = 0.5
    mask_ratio_min_max: List[float] = field(default_factory=lambda: [0.1, 0.5])
    separate_head: bool = False
    head_n_prototypes: int = 65536
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048

class CenteringType(str, Enum):
    sinkhorn_knopp = "sinkhorn_knopp"
    centering = "centering"

class WandbMode(str, Enum):
    disabled = "disabled"
    enabled = None

@dataclass
class TrainConfig:
    name: str = 'unnamed'
    batch_size_per_gpu: Optional[int] = None # Autoset when total_batch_size is set.
    # dataset_path alternatively supports ngram: when ngram is wanted.
    dataset_path: str =  "zip:dataset_root_dir=../chammi_train.zip" # zip:dataset_root_dir=/scr/data/CHAMMI/chammi_train.zip 
    split_fns: list[str] = field(default_factory=lambda: ["randomize", "get_proc_split", "split_for_workers"])
    output_dir: str = "."
    saveckp_freq: int = 20
    seed: int = 42
    num_workers: int = 10
    OFFICIAL_EPOCH_LENGTH: int = 1250
    cache_dataset: bool = True
    centering: CenteringType = CenteringType.sinkhorn_knopp 
    wandb_mode: WandbMode = WandbMode.enabled
    sampler: str = "none" 
    dataset_size: Optional[int] = None # number of image in the dataset, leave null for autocomputation.
    total_batch_size: Optional[int] = 512 # set to 0 and use batch_size_per_gpu
    # ngram arguments. Used when dataset_path starts with ngram: instead of zip:.
    ngram_sample_rate: Optional[float] = None # 0.5. 
    dataset_config: Optional[str] = None # /scr/data/CHAMMI/multi_channel_chammi_metadata.csv 
    

class Arch(str, Enum):
    vit_small_single_channel = "vit_small_single_channel" 
    vit_base_single_channel  = "vit_base_single_channel" 
    vit_small_ngram = "vit_small_ngram"

@dataclass
class StudentConfig:
    arch: Arch = Arch.vit_small_single_channel
    patch_size: int = 16
    drop_path_rate: float = 0.0 # 0.3, needs to be 0 unless embed dim = 768
    layerscale: float = 1.0e-05
    drop_path_uniform: bool = True
    pretrained_weights: str = ""
    ffn_layer: str = "swiglufused"
    block_chunks: int = 0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    num_register_tokens: int = 0
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1

@dataclass
class TeacherConfig:
    momentum_teacher: float = 0.994
    final_momentum_teacher: float = 1.0
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30

class ScalingRule(str, Enum):
    sqrt_wrt_1024 = "sqrt_wrt_1024" 

@dataclass
class OptimConfig:
    epochs: int = 100
    weight_decay: float = 0.04
    weight_decay_end: float = 0.2
    base_lr: float = 0.0002
    lr: float = 0.0 # auto-gen with respect to scaling_rule
    warmup_epochs: int = 10
    min_lr: float = 1.0e-06
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    scaling_rule: Optional[ScalingRule] = None 
    patch_embed_lr_mult: float = 0.2
    layerwise_decay: float = 1.0
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    min_grad: float = 0.5
    min_its: int = 2000 # Applies min grad run stopping after this many its

@dataclass
class CropsConfig:
    global_crops_scale: List[float] = field(default_factory=lambda: [0.32, 1.0])
    local_crops_number: int = 8
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