"""Compute meanIOU."""

import os

import torch as th

from data_loader import Loader
from train import UNet3D, normalize


def compute_iou(preds: th.Tensor, target: th.Tensor) -> th.Tensor:
    """Calculate meanIoU for a given batch.

    Args:
        preds (jnp.ndarray): Predictions from network
        target (jnp.ndarray): Labels

    Returns:
        jnp.ndarray: Mean Intersection over Union values
    """
    assert preds.shape == target.shape
    # TODO: Implement meanIoU
    return th.tensor(0.0)


if __name__ == "__main__":
    keys = os.listdir("./data/gtexport/Test")
    model_path = "./weights/unet_softmaxfl_124.pkl"  # Change this model path in case name differs
    input_shape = [128, 128, 21]
    mean = th.tensor([206.12558])
    std = th.tensor([164.74423])

    dataset = Loader(input_shape=input_shape, val_keys=keys)
    test_imgs = dataset.get_val(test=True)
    test_imgs, test_labels = test_imgs["images"], test_imgs["annotation"]
    batch_size = 2

    model = UNet3D()
    model.load_state_dict(th.load(model_path))

    batched_imgs = th.split(test_imgs, batch_size, dim=0)
    batched_labels = th.split(test_labels, batch_size, dim=0)

    ious = []
    for batch_index in range(len(batched_imgs)):
        imgs, lbls = (
            normalize(batched_imgs[batch_index], mean, std),
            batched_labels[batch_index],
        )
        preds = model(imgs)
        preds = preds.permute((0, 2, 3, 4, 1))
        preds = th.argmax(preds, dim=-1)
        ious.append(compute_iou(preds, lbls))

    mean_iou = th.mean(th.tensor(ious))
    print(f"Mean IoU: {mean_iou}")
