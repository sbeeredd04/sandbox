# aug_utils.py  (final API)
from __future__ import annotations
import torch
import kornia as K

DEBUG = False

class ImageAugmentation:
    """Horizontally flip an image.

    Workflow
    --------
    img_aug = ImageAugmentation(horizontal_flip=0.5)
    flip_flag = img_aug.renew_augmentation()   # decides True/False and returns it
    out = img_aug(image_tensor)                # applies that decision
    """

    def __init__(self, *, horizontal_flip: float = 0.0) -> None:
        if not 0.0 <= horizontal_flip <= 1.0:
            raise ValueError("horizontal_flip must be in [0, 1]")
        self._p: float = horizontal_flip
        self._do_flip: bool | None = None

    # ------------------------------------------------------------------
    def renew_augmentation(self, should_flip: bool | None = None) -> bool:
        """Set and **return** the flip decision for the next call.

        Parameters
        ----------
        should_flip : bool | None
            * True / False → force that decision.
            * None         → sample with probability *horizontal_flip*.

        Returns
        -------
        bool
            The decision that will be applied.
        """
        if should_flip is None:
            should_flip = bool(torch.rand(()) < self._p)
        self._do_flip = bool(should_flip)
        if DEBUG:
            print(f"ImageAugmentation: renew_augmentation() -> {self._do_flip}")

        return self._do_flip

    # ------------------------------------------------------------------
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the current cached decision (sample if uninitialised)."""
        if self._do_flip is None:        # lazy sampling
            self.renew_augmentation()
        return K.geometry.transform.hflip(img) if self._do_flip else img


# ----------------------------------------------------------------------
class ActionAugmentation:
    """
    Mirror the **y** component of 2-D *(x y)* or 3-D *(x y z)* coordinates.  
    If the input has only two channels the behaviour is identical to the old
    `XYAugmentation`.

    Parameters
    ----------
    horizontal_flip : float, default=0.0  
        Probability *p* with which the augmentation will flip **y**.
        Must be in **[0.0, 1.0]**.
    """

    def __init__(self, *, horizontal_flip: float = 0.0) -> None:
        if not 0.0 <= horizontal_flip <= 1.0:
            raise ValueError("horizontal_flip must be in [0, 1]")
        self._p: float = horizontal_flip
        self._do_flip: bool | None = None  # decided per sample/batch

    # ------------------------------------------------------------------ #
    #                         sampling the policy                        #
    # ------------------------------------------------------------------ #
    def renew_augmentation(self, should_flip: bool | None = None) -> bool:
        """
        Decide (and cache) whether the next call should flip.

        Returns
        -------
        bool
            `True`  → flip **y**  
            `False` → leave input unchanged
        """
        if should_flip is None:
            should_flip = bool(torch.rand(()) < self._p)
        self._do_flip = bool(should_flip)

        if DEBUG:
            print(f"ActionAugmentation: renew_augmentation() -> {self._do_flip}")
        return self._do_flip

    # ------------------------------------------------------------------ #
    #                          apply to a tensor                         #
    # ------------------------------------------------------------------ #
    def __call__(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : `torch.Tensor`
            Any shape, **last dimension must be 2 or 3** (…,2) or (…,3).

        Returns
        -------
        `torch.Tensor`
            Same shape & dtype as input, with **y** mirrored if the
            augmentation was activated.
        """
        if xyz.size(-1) not in (2, 3):
            raise ValueError(
                "ActionAugmentation expects last dim to be 2 (xy) or 3 (xyz)"
            )

        if self._do_flip is None:                     # first call in epoch
            self.renew_augmentation()

        if self._do_flip:
            xyz = xyz.clone()
            xyz[..., 1] = -xyz[..., 1]                # mirror y

        return xyz


class HeadingAugmentation:
    """Mirror heading angles across the x-axis (angle → −angle)."""

    def __init__(self, *, horizontal_flip: float = 0.0) -> None:
        if not 0.0 <= horizontal_flip <= 1.0:
            raise ValueError("horizontal_flip must be in [0, 1]")
        self._p = horizontal_flip
        self._do_flip: bool | None = None

    def renew_augmentation(self, should_flip: bool | None = None) -> bool:
        if should_flip is None:
            should_flip = bool(torch.rand(()) < self._p)
        self._do_flip = bool(should_flip)
        return self._do_flip

    def __call__(self, heading: torch.Tensor) -> torch.Tensor:
        if self._do_flip is None:
            self.renew_augmentation()
        if self._do_flip:
            heading = -heading.clone()
        return heading