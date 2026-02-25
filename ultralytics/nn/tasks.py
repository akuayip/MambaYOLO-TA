# Ultralytics YOLO 🚀, AGPL-3.0 license
# Refactored for Mamba-YOLO Object Detection only

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    ResNetLayer,
    SimpleStem,
    VisionClueMerge,
    VSSBlock,
    XSSBlock,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    make_divisible,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Perform a forward pass through the network."""
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Perform a single forward pass through the network."""
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. "
            f"Reverting to single-scale inference instead."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """Profile the computation time and FLOPs of a single layer."""
        c = m == self.model[-1] and isinstance(x, list)
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """Fuse Conv2d and BatchNorm2d layers for improved computation efficiency."""
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)
        return self

    def is_fused(self, thresh=10):
        """Check if the model has less than a threshold of BatchNorm layers."""
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information."""
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """Apply a function to all tensors in the model."""
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load pre-trained weights into the model."""
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """Compute loss."""
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """Mamba-YOLO detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            try:
                m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            except Exception as e:
                if (
                    "Not implemented on the CPU" in str(e)
                    or "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)" in str(e)
                    or "CUDA tensor" in str(e)
                    or "is_cuda()" in str(e)
                    or "Cannot find backend for cpu" in str(e)
                ):
                    self.model.to(torch.device("cuda"))
                    m.stride = torch.tensor(
                        [s / x.shape[-2] for x in forward(torch.zeros(2, ch, s, s).to(torch.device("cuda")))]
                    )
                else:
                    raise e
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference."""
        p[:, :4] /= scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))
        e = 1
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))
        y[0] = y[0][..., :-i]
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)  # nms ensemble
        return y, None


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None):
    """Context manager for temporarily adding or modifying modules in Python's module cache."""
    if not modules:
        modules = {}

    import importlib
    import sys

    try:
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)
        yield
    finally:
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    """Load a PyTorch model safely, handling missing modules."""
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)
    try:
        with temporary_modules(
            {
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            }
        ):
            ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
            f"\\nAutoInstall will run now for '{e.name}'."
        )
        check_requirements(e.name)
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Load an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()

        model.args = args
        model.pt_path = w
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())

    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None

    if len(ensemble) == 1:
        return ensemble[-1]

    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Load a single model weights."""
    ckpt, weight = torch_safe_load(weight)
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()

    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    model.pt_path = weight
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()

    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None

    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a Mamba-YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in {
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            RepNCSPELAN4,
            ADown,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            SimpleStem,
            VisionClueMerge,
            VSSBlock,
            XSSBlock,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, XSSBlock}:
                args.insert(2, n)
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a Mamba-YOLO model from a YAML file."""
    import re

    path = Path(path)
    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """Extract the size character of the model scale from the YAML file name."""
    with contextlib.suppress(AttributeError):
        import re

        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(1)
    return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.
    For Mamba-YOLO, this always returns 'detect'.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()
        if m == "detect":
            return "detect"

    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    if isinstance(model, nn.Module):
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Detect):
                return "detect"

    if isinstance(model, (str, Path)):
        model = Path(model)
        if "detect" in model.parts:
            return "detect"

    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect'."
    )
    return "detect"
