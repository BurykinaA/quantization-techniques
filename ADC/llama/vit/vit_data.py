"""
ImageNet data loading for ViT ADC quantization.

Ported from vit_adc/datasets.py (ViTImageNetLoaderGenerator).  The transforms
are derived from the timm model's own ``default_cfg`` so preprocessing exactly
matches how the pretrained weights were trained (resize/crop/normalisation).

Directory layout expected under ``$IMAGENET_ROOT``::

    <root>/train/<class>/*.JPEG
    <root>/val/<class>/*.JPEG

The val class directories in this workspace are named ``n00000000..n00000999``.
ImageFolder sorts class dirs alphabetically, which reproduces the standard
sorted-wnid order that timm's pretrained classifier expects — verified: a
pretrained vit_tiny scores ~76% top-1, so no label remapping is needed.
"""

import os
import logging

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from timm.data import resolve_data_config, create_transform

logger = logging.getLogger(__name__)


def resolve_imagenet_root(explicit: str | None = None) -> str:
    """Return the ImageNet root, preferring an explicit path then $IMAGENET_ROOT."""
    root = explicit or os.environ.get("IMAGENET_ROOT")
    if not root:
        raise ValueError(
            "ImageNet root not set. Pass --data_dir or export IMAGENET_ROOT "
            "(e.g. /home/coder/project/imagenet/data)."
        )
    root = root.rstrip("/")
    if not os.path.isdir(os.path.join(root, "val")):
        raise FileNotFoundError(f"No 'val' directory under ImageNet root: {root}")
    return root


class ViTImageNetLoaderGenerator:
    """Builds train/val/calibration loaders with timm-derived transforms.

    Parameters
    ----------
    root : str
        ImageNet root containing ``train`` and ``val`` subdirectories.
    model : nn.Module
        A timm model — its ``default_cfg`` determines the transform pipeline.
    val_batch_size, calib_batch_size : int
    num_workers : int
    """

    def __init__(self, root, model, val_batch_size=128, calib_batch_size=32,
                 num_workers=8):
        self.root = resolve_imagenet_root(root)
        self.val_batch_size = val_batch_size
        self.calib_batch_size = calib_batch_size
        self.num_workers = num_workers

        config = resolve_data_config(model.default_cfg, model=model)
        # is_training=False for both — calibration uses the val (deterministic)
        # transform, matching the vit_adc convention.
        self.transform = create_transform(**config)
        logger.info(f"ViT transform: {self.transform}")

    def _val_dataset(self):
        return datasets.ImageFolder(os.path.join(self.root, "val"),
                                    transform=self.transform)

    def _train_dataset(self):
        return datasets.ImageFolder(os.path.join(self.root, "train"),
                                    transform=self.transform)

    def val_loader(self, portion: float = 1.0, seed: int = 0) -> DataLoader:
        """Validation loader.  ``portion`` < 1.0 uses a strided subset that
        spans all classes (every Nth image) for a fast, representative check."""
        ds = self._val_dataset()
        if portion < 1.0:
            stride = max(1, int(round(1.0 / portion)))
            idx = list(range(0, len(ds), stride))
            ds = Subset(ds, idx)
            logger.info(f"val subset: {len(ds)} / stride={stride} (portion={portion})")
        return DataLoader(ds, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def calib_loader(self, num: int = 1024, seed: int = 3,
                     batch_size: int | None = None) -> DataLoader:
        """Calibration loader: ``num`` random images from the train split
        (val transform, fixed seed for reproducibility).

        ``batch_size`` overrides the generator's ``calib_batch_size`` (used e.g.
        by LoRA calibration, which needs a smaller batch than FlatQuant)."""
        ds = self._train_dataset()
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ds), generator=g)[:num].tolist()
        sub = Subset(ds, perm)
        bsz = batch_size if batch_size is not None else self.calib_batch_size
        logger.info(f"calib subset: {len(sub)} images from train "
                    f"(seed={seed}, batch_size={bsz})")
        return DataLoader(sub, batch_size=bsz, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
