import torch.nn as nn
from ultralytics.utils import callbacks
import torch
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Concat,
    Conv,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
)
from pipelearn.models.yolo6D.head import Pose6DHead
from copy import deepcopy
from typing import Dict


class YOLO6DModel(nn.Module):
    def __init__(
        self, ckpt_path, freeze_backbone=False, override_mapping: Dict[int, str] = None
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.callbacks = callbacks.get_default_callbacks()
        self.override_mapping = override_mapping
        self._load(ckpt_path)

    def _load(self, ckpt_path):
        # Load original model and initialize basic attributes
        orig_model, self.ckpt = self.attempt_load_one_weight(ckpt_path)
        self._initialize_basic_attributes(orig_model, ckpt_path)

        self.names = (
            self.override_mapping
            if self.override_mapping
            else orig_model.yaml.get("names")
        )
        # Configure model architecture
        ch = self.model_cfg.get("ch", 3)
        self._setup_model_architecture(ch)

        # Handle weights transfer
        self._transfer_weights()

        # Finalize model setup
        self.freeze_layers()

    def _initialize_basic_attributes(self, orig_model, ckpt_path):
        """Initialize basic model attributes from original model."""
        self.model_cfg = orig_model.yaml
        if self.model_cfg["backbone"][0][2] == "Silence":
            self.model_cfg["backbone"][0][2] = "nn.Identity"

        self.model_cfg["head"][-1][
            2
        ] = "Pose6DHead"

        self.orig_nc = self.model_cfg.get("nc")
        self.task = "6dpose"
        self.ckpt_path = orig_model.pt_path
        self.model_name = ckpt_path

    def _setup_model_architecture(self, ch):
        """Configure model architecture with proper number of classes."""
        # Initial model parse
        self.model, self.save = self.parse_model(deepcopy(self.model_cfg), ch)
        self.stride = torch.tensor([32.0])

        # Update model configuration with new number of classes
        self.nc = max(int(idx) for idx in self.names.keys()) + 1
        self.model_cfg.update({"nc": self.nc, "names": self.names})

        # Rebuild model with updated configuration
        self.model, self.save = self.parse_model(deepcopy(self.model_cfg), ch)
        self.inplace = self.model_cfg.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        self._setup_pose6d_head(ch)
        initialize_weights(self)

    def _setup_pose6d_head(self, ch):
        """Configure detection head and strides."""
        m = self.model[-1]
        if isinstance(m, Pose6DHead):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            # Configure stride
            m.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self._forward_detect(torch.zeros(1, ch, s, s))
                ]
            )
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

    def _forward_detect(self, x):
        """Helper method for detection forward pass."""
        if self.end2end:
            return self.forward(x)["one2many"]
        return self.forward(x)

    def _transfer_weights(self):
        """Handle weight transfer from pretrained model."""
        model = self.ckpt["model"] if isinstance(self.ckpt, dict) else self.ckpt
        csd = model.float().state_dict()

        # Load weights
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)

        print(
            f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights"
        )

    def freeze_layers(self):
        # Freeze DFL layers
        always_freeze_names = [".dfl"]  # always freeze these layers

        for k, v in self.model.named_parameters():
            # First handle DFL freezing
            if any(x in k for x in always_freeze_names):
                print(f"Freezing DFL layer '{k}'")
                v.requires_grad = False

            # Then handle backbone freezing if enabled
            elif self.freeze_backbone:
                # Check if parameter belongs to any layer before the final detection layer
                if not k.startswith(f"{self.model[-1].i}."):  # if not in final layer
                    print(f"Freezing backbone layer '{k}'")
                    v.requires_grad = False

            # Ensure floating point tensors that should have gradients do have them
            elif not v.requires_grad and v.dtype.is_floating_point:
                v.requires_grad = True

    def forward(self, x, *args, **kwargs):
        input_x = x
        y, dt, embeddings = [], [], []
        embed = kwargs.get("embed", None)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if embed and m.i in embed:
                embeddings.append(
                    nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                )  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        return x

    def attempt_load_one_weight(self, ckpt_path, inplace=True):
        ckpt = torch.load(ckpt_path)  # load ckpt
        args = ckpt.get("train_args", {})
        model = ckpt["model"]

        # Model compatibility updates
        model.args = {k: v for k, v in args.items()}  # attach args to model
        model.pt_path = ckpt_path
        model.task = "detect"
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        model = model.eval()  # model in eval mode

        # Module updates
        for m in model.modules():
            if hasattr(m, "inplace"):
                m.inplace = inplace
            elif isinstance(m, nn.Upsample) and not hasattr(
                m, "recompute_scale_factor"
            ):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # Return model and ckpt
        return model, ckpt

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        Resets specific arguments when loading a PyTorch model checkpoint.

        This static method filters the input arguments dictionary to retain only a specific set of keys that are
        considered important for model loading. It's used to ensure that only relevant arguments are preserved
        when loading a model from a checkpoint, discarding any unnecessary or potentially conflicting settings.

        Args:
            args (dict): A dictionary containing various model arguments and settings.

        Returns:
            (dict): A new dictionary containing only the specified include keys from the input arguments.

        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {
            "imgsz",
            "data",
            "task",
            "single_cls",
        }  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def parse_model(self, d, ch, verbose=True):  # model_dict, input_channels(3)
        """Parse a YOLO model.yaml dictionary into a PyTorch model."""
        import ast

        # Args
        legacy = True  # backward compatibility for v3/v5/v8/v9 models
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (
            d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
        )
        if scales:
            scale = d.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
                # LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(
                act
            )  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
            if verbose:
                print(f"activation: {act}")  # print

        if verbose:
            print(
                f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
            )
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(
            d["backbone"] + d["head"]
        ):  # from, number, module, args
            m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
            for j, a in enumerate(args):
                if isinstance(a, str):
                    try:
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    except ValueError:
                        pass
            n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
            if m in {
                Conv,
                ConvTranspose,
                GhostConv,
                Bottleneck,
                GhostBottleneck,
                SPP,
                SPPF,
                C2fPSA,
                C2PSA,
                DWConv,
                Focus,
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                RepNCSPELAN4,
                ELAN1,
                ADown,
                AConv,
                SPPELAN,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                nn.ConvTranspose2d,
                DWConvTranspose2d,
                C3x,
                RepC3,
                PSA,
                SCDown,
                C2fCIB,
            }:
                c1, c2 = ch[f], args[0]
                if (
                    c2 != nc
                ):  # if c2 not equal to number of classes (i.e. for Classify() output)
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is C2fAttn:
                    args[1] = make_divisible(
                        min(args[1], max_channels // 2) * width, 8
                    )  # embed channels
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                        if args[2] > 1
                        else args[2]
                    )  # num heads

                args = [c1, c2, *args[1:]]
                if m in {
                    BottleneckCSP,
                    C1,
                    C2,
                    C2f,
                    C3k2,
                    C2fAttn,
                    C3,
                    C3TR,
                    C3Ghost,
                    C3x,
                    RepC3,
                    C2fPSA,
                    C2fCIB,
                    C2PSA,
                }:
                    args.insert(2, n)  # number of repeats
                    n = 1
                if m is C3k2:  # for M/L/X sizes
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
            elif m is AIFI:
                args = [ch[f], *args]
            elif m in {HGStem, HGBlock}:
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)  # number of repeats
                    n = 1
            elif m is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is Pose6DHead:
                args.append([ch[x] for x in f])
                m.legacy = legacy
            elif (
                m is RTDETRDecoder
            ):  # special case, channels arg must be passed in index 1
                args.insert(1, [ch[x] for x in f])
            elif m is CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch[f[-1]]
            else:
                c2 = ch[f]

            m_ = (
                nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            )  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            m_.np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            if verbose:
                print(
                    f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}"
                )  # print
            save.extend(
                x % i for x in ([f] if isinstance(f, int) else f) if x != -1
            )  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    model = YOLO6DModel(
        ckpt_path="/home/tilak/projects/learning/checkpoints/yolo11x.pt",
    )
    print(model)
