"""Samples the test data, to asses segmentation quality."""

from typing import Dict

import matplotlib.pyplot as plt
import torch as th

from data_loader import Loader
from train import UNet3D, normalize
from util import softmax_focal_loss

if __name__ == "__main__":
    checkpoint_name = "./weights/unet_softmaxfl_124.pkl"
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
    mean = th.Tensor([206.12558]).to(device)
    std = th.Tensor([164.74423]).to(device)
    input_shape = [128, 128, 21]
    val_keys = ["ProstateX-0004", "ProstateX-0007"]
    data_set = Loader(input_shape=input_shape, val_keys=val_keys)
    model = UNet3D()
    model.load_state_dict(th.load(checkpoint_name))
    model = model.to(device)
    model.eval()

    val_data = data_set.get_val(True)

    input_val = normalize(val_data["images"].to(device), mean=mean, std=std)
    with th.no_grad():
        val_out = model(input_val)
    val_out = val_out.permute((0, 2, 3, 4, 1))
    label_val = th.nn.functional.one_hot(
        val_data["annotation"].type(th.int64), num_classes=5
    ).to(device)
    val_loss = th.mean(
        softmax_focal_loss(val_out, label_val, th.ones((val_out.shape[-1])).to(device))
    )
    print(f"Validation loss: {val_loss:2.6f}")
    val_out = val_out.cpu()

    def disp_result(
        sample: int,
        data: Dict[str, th.Tensor],
        out: th.Tensor,
        name: str,
        slice: int = 11,
    ):
        """Plot the original image, network output and annotation."""
        plt.title("scan")
        plt.imshow(data["images"].squeeze(1)[sample, :, :, slice])
        plt.savefig(f"test_scan_{name}.png")
        plt.title("network")
        plt.imshow(th.argmax(out[sample, :, :, slice], dim=-1), vmin=0, vmax=5)

        plt.savefig(f"test_network_{name}.png")
        plt.title("human expert")
        plt.imshow(data["annotation"][sample, :, :, slice], vmin=0, vmax=5)
        plt.savefig(f"test_expert_{name}.png")

    disp_result(0, val_data, val_out, "val_0")
    disp_result(1, val_data, val_out, "val_1")

    test_data = data_set.get_test_set()
    input_test = normalize(test_data["images"].to(device), mean=mean, std=std)
    with th.no_grad():
        test_out = model(input_test)
    test_out = test_out.permute((0, 2, 3, 4, 1))
    label_test = th.nn.functional.one_hot(
        test_data["annotation"].type(th.int64), num_classes=5
    ).to(device)
    test_loss = th.mean(
        softmax_focal_loss(
            test_out, label_test, th.ones((test_out.shape[-1])).to(device)
        )
    )
    print(f"Test loss: {test_loss:2.6f}")
    test_out = test_out.cpu()

    for i in range(20):
        disp_result(i, test_data, test_out, f"test_{str(i)}")
