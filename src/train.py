"""Train a U-Net for medical image segmentation.

Methods as described in
https://arxiv.org/pdf/1505.04597.pdf and
https://www.var.ovgu.de/pub/2019_Meyer_ISBI_Zone_Segmentation.pdf.
"""

import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from clu import metric_writers
from tqdm import tqdm

from data_loader import Loader
from util import softmax_focal_loss


def normalize(data: th.Tensor, mean: th.Tensor, std: th.Tensor) -> th.Tensor:
    """Normalize input data.

    Args:
        data (th.Tensor): Input tensor.
        mean (th.Tensor): Mean.
        std (th.Tensor): Standard deviation.

    Returns:
        th.Tensor: Normalized input.
    """
    return (data - mean) / std


def save_network(model: th.nn.Module, epoch: int) -> None:
    """Save network module.

    Args:
        model (th.nn.Module): Model instance.
        epoch (int): Epoch.
    """
    th.save(model.state_dict(), f"./weights/unet_softmaxfl_{epoch}.pth")


def pad_odd(input_x: th.Tensor) -> th.Tensor:
    """Padd odd input dimension.

    Args:
        input_x (th.Tensor): Input tensor.

    Returns:
        th.Tensor: Padded input.
    """
    # return input_x
    pad_list = []
    input_shape = input_x.shape[::-1]
    for axis_shape in input_shape[:-2]:
        if axis_shape % 2 != 0:
            pad_list += [0, 1]
        else:
            pad_list += [0, 0]
    pad_list.extend([0, 0, 0, 0])
    return th.nn.functional.pad(input_x, pad_list)


class UNet3D(th.nn.Module):
    """3D UNet."""

    def __init__(self):
        """Intialize network."""
        super().__init__()
        input_feat = 1
        init_feat = 16
        out_neurons = 5
        # Five Downscale blocks
        self.downscale_1 = th.nn.Sequential(
            th.nn.Conv3d(input_feat, init_feat, (3, 3, 3), padding=1),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat, init_feat, (3, 3, 3), padding=1),
            th.nn.ReLU(),
        )
        self.downscale_2 = th.nn.Sequential(
            th.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            # th.nn.BatchNorm3d(init_feat),
            th.nn.Conv3d(init_feat, init_feat * 2, (3, 3, 3), padding=1),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat * 2, init_feat * 2, (3, 3, 3), padding=1),
            th.nn.ReLU(),
        )
        self.downscale_3 = th.nn.Sequential(
            th.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            # th.nn.BatchNorm3d(init_feat * 2),
            th.nn.Conv3d(init_feat * 2, init_feat * 4, (3, 3, 3), padding=1),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat * 4, init_feat * 4, (3, 3, 3), padding=1),
            th.nn.ReLU(),
        )
        self.downscale_4 = th.nn.Sequential(
            th.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            # th.nn.BatchNorm3d(init_feat * 4),
            th.nn.Conv3d(init_feat * 4, init_feat * 8, (3, 3, 3), padding=1),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat * 8, init_feat * 8, (3, 3, 3), padding=1),
            th.nn.ReLU(),
        )
        self.downscale_5 = th.nn.Sequential(
            th.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            # th.nn.BatchNorm3d(init_feat * 8),
            th.nn.Conv3d(init_feat * 8, init_feat * 16, (3, 3, 3), padding=1),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat * 16, init_feat * 16, (3, 3, 3), padding=1),
            th.nn.ReLU(),
        )
        # Four Upscale conv blocks
        self.upscale_4 = th.nn.Sequential(
            # th.nn.BatchNorm3d(init_feat * 16  + init_feat * 8),
            th.nn.Conv3d(
                init_feat * 16 + init_feat * 8, init_feat * 8, (3, 3, 3), padding=1
            ),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat * 8, init_feat * 8, (3, 3, 3), padding=1),
            th.nn.ReLU(),
        )
        self.upscale_3 = th.nn.Sequential(
            # th.nn.BatchNorm3d(init_feat * 8 + init_feat * 4),
            th.nn.Conv3d(
                init_feat * 8 + init_feat * 4, init_feat * 4, (3, 3, 3), padding=1
            ),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat * 4, init_feat * 4, (3, 3, 3), padding=1),
            th.nn.ReLU(),
        )
        self.upscale_2 = th.nn.Sequential(
            # th.nn.BatchNorm3d(init_feat * 4 + init_feat * 2),
            th.nn.Conv3d(
                init_feat * 4 + init_feat * 2, init_feat * 2, (3, 3, 3), padding=1
            ),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat * 2, init_feat * 2, (3, 3, 3), padding=1),
            th.nn.ReLU(),
        )
        self.upscale_1 = th.nn.Sequential(
            # th.nn.BatchNorm3d(init_feat * 2 + init_feat),
            th.nn.Conv3d(init_feat * 2 + init_feat, init_feat, (3, 3, 3), padding=1),
            th.nn.ReLU(),
            th.nn.Conv3d(init_feat, out_neurons, (3, 3, 3), padding=1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Segmented output.
        """
        x1 = self.downscale_1(x)
        x1 = pad_odd(x1)

        x2 = self.downscale_2(x1)
        x2 = pad_odd(x2)

        x3 = self.downscale_3(x2)
        x3 = pad_odd(x3)

        x4 = self.downscale_4(x3)
        x4 = pad_odd(x4)

        x5 = self.downscale_5(x4)
        x5 = pad_odd(x5)

        x6 = self.__upsize(x5)
        x6 = x6[:, :, : x4.shape[2], : x4.shape[3], : x4.shape[4]]
        x6 = th.cat([x4, x6], dim=1)
        x6 = self.upscale_4(x6)

        x7 = self.__upsize(x6)
        x7 = x7[:, :, : x3.shape[2], : x3.shape[3], : x3.shape[4]]
        x7 = th.cat([x3, x7], dim=1)
        x7 = self.upscale_3(x7)

        x8 = self.__upsize(x7)
        x8 = x8[:, :, : x2.shape[2], : x2.shape[3], : x2.shape[4]]
        x8 = th.cat([x2, x8], dim=1)
        x8 = self.upscale_2(x8)

        x9 = self.__upsize(x8)
        x9 = x9[:, :, : x1.shape[2], : x1.shape[3], : x1.shape[4]]
        x9 = th.cat([x1, x9], dim=1)
        x9 = self.upscale_1(x9)
        out = x9[:, :, : x.shape[2], : x.shape[3], : x.shape[4]]
        return out

    def __upsize(self, input_: th.Tensor) -> th.Tensor:
        """Upsample image.

        Args:
            input_ (th.Tensor): Input image.

        Returns:
            th.Tensor: Upsampled image.
        """
        _, _, d, h, w = input_.shape
        return th.nn.Upsample(size=(d, h * 2, w * 2), mode="nearest")(input_)


def train():
    """Train the UNet."""
    # Choose two scans for validation.
    val_keys = ["ProstateX-0004", "ProstateX-0007"]
    input_shape = [128, 128, 21]
    data_set = Loader(
        input_shape=input_shape,
        val_keys=val_keys,
        data_path="/home/lveerama/Courses/Foundations_of_Machine_Learning/day_13_exercise_segmentation_solution/data/",
    )
    epochs = 125
    batch_size = 2

    device = th.device("cuda:0") if th.cuda.is_available() else th.device("cpu")
    print(f"Using device: {device}...")

    model = UNet3D().to(device)
    opt = th.optim.Adam(model.parameters(), lr=1e-4)
    load_new = False

    writer = metric_writers.create_default_writer(
        "./runs/" + str(datetime.now()), asynchronous=False
    )

    os.makedirs("./weights/", exist_ok=True)
    th.manual_seed(42)
    epoch_batches = None
    if load_new:
        epoch_batches = list(data_set.get_epoch(batch_size))
        os.makedirs("./data/pickled/", exist_ok=True)
        with open("./data/pickled/batch_dump.pkl", "wb") as fp:
            pickle.dump(epoch_batches, fp)
    else:
        with open("./data/pickled/batch_dump.pkl", "rb") as fp:
            epoch_batches = pickle.load(fp)

    val_data = data_set.get_val()
    mean = th.Tensor([206.12558])
    mean = mean.to(device)
    std = th.Tensor([164.74423])
    std = std.to(device)
    val_loss_list = []
    train_loss_lost = []
    iter_count = 0
    # loss_fn = th.nn.CrossEntropyLoss()

    for e in range(epochs):
        random.shuffle(epoch_batches)
        epoch_batches_pre = iter(epoch_batches)
        model.train()
        for data_batch in (
            bar := tqdm(
                epoch_batches_pre, desc="Training 3D UNet", total=len(epoch_batches)
            )
        ):
            input_x = data_batch["images"].to(device)
            labels_y = data_batch["annotation"].to(device)

            labels_y = th.nn.functional.one_hot(labels_y.type(th.int64), num_classes=5)
            input_x = normalize(input_x, mean=mean, std=std)

            opt.zero_grad()
            preds = model(input_x)
            preds = preds.permute((0, 2, 3, 4, 1))
            loss = th.mean(
                softmax_focal_loss(
                    preds, labels_y, th.ones((preds.shape[-1])).to(device)
                )
            )
            # loss = loss_fn(preds, labels_y.type(th.LongTensor).to(device))
            loss.backward()
            opt.step()

            iter_count += 1
            bar.set_description(f"Epoch: {e}/{epochs}, Loss: {loss.item():2.6f}")
            train_loss_lost.append((iter_count, loss.item()))
            writer.write_scalars(iter_count, {"train_loss": loss.item()})

        model.eval()
        input_val = normalize(val_data["images"].to(device), mean=mean, std=std)
        label_val = th.nn.functional.one_hot(
            val_data["annotation"].type(th.int64), num_classes=5
        ).to(device)
        with th.no_grad():
            val_out = model(input_val)
        val_out = val_out.permute((0, 2, 3, 4, 1))
        val_loss = th.mean(
            softmax_focal_loss(
                val_out, label_val, th.ones((val_out.shape[-1])).to(device)
            )
        )
        # label_val = val_data["annotation"].to(device)
        # val_loss = loss_fn(val_out, label_val.type(th.LongTensor).to(device))
        val_loss_list.append((e, val_loss.item()))
        writer.write_scalars(e, {"validation_loss": val_loss.item()})
        val_out = val_out.cpu()
        print(f"Validation loss: {val_loss.item()}")
        # val_out = val_out.permute((0, 2, 3, 4, 1))
        for i in range(len(val_keys)):
            writer.write_images(
                e,
                {
                    f"{i}_val_netw_seg": th.argmax(val_out[i, :, :, 12], -1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    / 5.0
                },
            )
            writer.write_images(
                e,
                {
                    f"{i}_val_true_seg": val_data["annotation"][i, :, :, 12]
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    / 5.0
                },
            )
        if e % 20 == 0:
            save_network(model, e)
        if e > 100:
            for g in opt.param_groups:
                g["lr"] = 1e-5

    tll = np.stack(train_loss_lost, -1)
    vll = np.stack(val_loss_list, -1)
    plt.semilogy(tll[0], tll[1])
    plt.title("Training loss")
    plt.show()
    plt.semilogy(vll[0], vll[1])
    plt.title("Validation loss")
    plt.show()

    plt.subplot(121)
    plt.imshow(val_data["annotation"][1, :, :, 12])
    plt.title("Annotation")
    plt.subplot(122)
    plt.imshow(th.argmax(val_out[1, :, :, 12], axis=-1))
    plt.title("Network Output")
    plt.show()

    print("Training done.")
    save_network(model, e)


if __name__ == "__main__":
    train()
