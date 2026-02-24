import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }
    
def make_libero_example_forward(
    model,
    *,
    batch_size: int = 2,
    prompt_len: int = 16,
    image_hw: tuple[int, int] = (224, 224),
    right_wrist_present: bool = False,
    device: str | torch.device | None = None,
):
    if device is None:
        device = next(model.parameters()).device

    H, W = image_hw

    action_horizon = int(getattr(model.config, "action_horizon", 10))
    action_dim = int(getattr(model.config, "action_dim", 32))
    if action_dim != 32:
        raise ValueError(f"Expected action_dim=32, got {action_dim}")

    state_dim = 32

    vocab_size = None
    try:
        vocab_size = int(model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.shape[0])
    except Exception:
        pass
    if vocab_size is None:
        # fallback: ありがちな場所
        try:
            vocab_size = int(model.paligemma_with_expert.paligemma.config.vocab_size)
        except Exception:
            vocab_size = 32000  # 最終fallback（環境依存）

    def _rand_img_chw_float_m11():
        x = torch.randint(0, 256, (batch_size, 3, H, W), device=device, dtype=torch.uint8)
        x = x.to(torch.float32) / 127.5 - 1.0   # [-1,1]
        return x

    images = {
        "base_0_rgb": _rand_img_chw_float_m11(),
        "left_wrist_0_rgb": _rand_img_chw_float_m11(),
        "right_wrist_0_rgb": _rand_img_chw_float_m11() if right_wrist_present
                           else torch.zeros((batch_size, 3, H, W), device=device, dtype=torch.float32),
    }

    image_masks = {
        "base_0_rgb": torch.ones((batch_size,), device=device, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones((batch_size,), device=device, dtype=torch.bool),
        "right_wrist_0_rgb": torch.ones((batch_size,), device=device, dtype=torch.bool) if right_wrist_present
                           else torch.zeros((batch_size,), device=device, dtype=torch.bool),
    }

    tokenized_prompt = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device, dtype=torch.long)
    tokenized_prompt_mask = torch.ones((batch_size, prompt_len), device=device, dtype=torch.bool)
    token_ar_mask = torch.ones((batch_size, prompt_len), device=device, dtype=torch.bool)
    token_loss_mask = torch.ones((batch_size, prompt_len), device=device, dtype=torch.bool)

    state = torch.randn((batch_size, state_dim), device=device, dtype=torch.float32)

    # actions: (B,T,32) — 先頭7次元だけLIBEROっぽく、残り0
    actions = torch.zeros((batch_size, action_horizon, action_dim), device=device, dtype=torch.float32)
    actions[:, :, :6] = torch.rand((batch_size, action_horizon, 6), device=device) * 0.1 - 0.05
    actions[:, :, 6:7] = torch.rand((batch_size, action_horizon, 1), device=device)

    class SimpleObservation:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    observation = SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )

    return observation, actions

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
