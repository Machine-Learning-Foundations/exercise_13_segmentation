"""Medical image segmentation helper functions."""

from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch as th
from SimpleITK.SimpleITK import Image

# from . import zone_segmentation_utils as utils


def resample_image(input_image, new_spacing, interpolator, default_value):
    """Resample the input scans.

    Adapted from
    https://github.com/AnnekeMeyer/zone-segmentation/blob/c1a5f584c10afd31cbe5356d7e2f4371cb880b06/utils.py#L113
    """
    cast_image_filter = sitk.CastImageFilter()
    cast_image_filter.SetOutputPixelType(sitk.sitkFloat32)
    input_image = cast_image_filter.Execute(input_image)

    old_size = input_image.GetSize()
    old_spacing = input_image.GetSpacing()
    new_width = old_spacing[0] / new_spacing[0] * old_size[0]
    new_height = old_spacing[1] / new_spacing[1] * old_size[1]
    new_depth = old_spacing[2] / new_spacing[2] * old_size[2]
    new_size = [int(new_width), int(new_height), int(new_depth)]

    min_filter = sitk.StatisticsImageFilter()
    min_filter.Execute(input_image)
    # min_value = min_filter.GetMinimum()

    filter = sitk.ResampleImageFilter()
    input_image.GetSpacing()
    filter.SetOutputSpacing(new_spacing)
    filter.SetInterpolator(interpolator)
    filter.SetOutputOrigin(input_image.GetOrigin())
    filter.SetOutputDirection(input_image.GetDirection())
    filter.SetSize(new_size)
    filter.SetDefaultPixelValue(default_value)
    out_image = filter.Execute(input_image)

    return out_image


def plot_box(box: List[np.ndarray]) -> None:
    """Plot a box as a matplotlib figure.

    Args:
        box (List[np.ndarray]): A list of lines
            as produced by the box_lines function.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for linepos, line in enumerate(box):
        if linepos == 0:
            ax.plot(line[0, 0], line[0, 1], line[0, 2], "s")
        ax.plot(line[:, 0], line[:, 1], line[:, 2], "-.")
    plt.show()


origin = np.array([0, 0, 0])


def box_lines(size: np.ndarray, start: np.ndarray = origin) -> List[np.ndarray]:
    """Create a box of a given size.

    Args:
        size (np.ndarray): A 3d array which specifies the
            height, widht and depth of the box.
        start (np.ndarray): A 3d dimensional displacement
            vector for the bottom front edge of the box.
            Defaults to the origin at [0, 0, 0].

    Returns:
        List[np.ndarray]: A list of boundary lines,
            which form a box.
            Use the `plot_box` function to visualize
            what happens.
    """
    stop = start + size
    bc = np.array([start[0], start[1], start[2]])
    br = np.array([stop[0], start[1], start[2]])
    bl = np.array([start[0], stop[1], start[2]])
    bb = np.array([stop[0], stop[1], start[2]])
    tc = np.array([start[0], start[1], stop[2]])
    tr = np.array([stop[0], start[1], stop[2]])
    tl = np.array([start[0], stop[1], stop[2]])
    tb = np.array([stop[0], stop[1], stop[2]])
    lines = [
        np.linspace(bc, br, 100),
        np.linspace(br, bb, 100),
        np.linspace(bb, bl, 100),
        np.linspace(bl, bc, 100),
        np.linspace(bb, tb, 100),
        np.linspace(bl, tl, 100),
        np.linspace(bc, tc, 100),
        np.linspace(br, tr, 100),
        np.linspace(tc, tr, 100),
        np.linspace(tr, tb, 100),
        np.linspace(tb, tl, 100),
        np.linspace(tl, tc, 100),
    ]
    return lines


def compute_roi(images: Tuple[Image, Image, Image]):
    """Find the region of interest (roi) of our medical scan tensors.

    Args:
        images (List[sitk.SimpleITK.Image]):
            A tuple with the axial t2w (t2w), saggital t2w (sag),
            and coronal t2w (cor) images.
            See i.e. https://en.wikipedia.org/wiki/Anatomical_plane
            for a defenition of these terms.

    Returns:
        List[List[np.ndarray], List[slice]]:
            'intersections', a list of rois for every input scan
            and 'box_indices' a List with the start and end indices of
            every scan in the original tensor.
            See https://docs.python.org/3/library/functions.html#slice
            for more information regarding python slices.
    """
    assert len(images) == 3

    # get the displacement vectors from the origin for every scan.
    origins = [np.asarray(img.GetOrigin()) for img in images]
    # find height, width and depth of every image-tensor.
    sizes = [np.asarray(img.GetSpacing()) * np.asarray(img.GetSize()) for img in images]
    # create a list with the rotation matrices for every scan.
    rotation = [np.asarray(img.GetDirection()).reshape(3, 3) for img in images]

    rects = []
    for pos, size in enumerate(sizes):
        lines = box_lines(size)
        # TODO: Rotate and shift the lines.
        rotated = []
        shifted = []
        rects.append(shifted)

    # find the intersection.
    rects_stacked = np.stack(rects)  # Had to rename because of mypy
    # TODO: Find the axis maxima and minima
    bbs = [
        (
            np.zeros_like(rect[0, 0]),
            np.zeros_like(rect[0, 0]),
        )  # TODO: fixme
        for rect in rects_stacked
    ]

    # compute intersection
    # TODO: Implement me.
    lower_end = np.zeros_like(bbs[0][0])
    upper_end = np.zeros_like(bbs[0][1])
    roi_bb = np.stack((lower_end, upper_end))
    roi_bb_size = roi_bb[1] - roi_bb[0]

    roi_bb_lines = np.stack(box_lines(roi_bb_size, roi_bb[0]))
    rects_stacked = np.concatenate([rects_stacked, np.expand_dims(roi_bb_lines, 0)])

    spacings = [image.GetSpacing() for image in images]
    # compute roi coordinates in image space.
    img_coord_rois = [
        (
            np.zeros_like(roi_bb[0]),  # TODO: Implement me
            np.zeros_like(roi_bb[1]),  # TODO: Implement me
        )
        for rot, offset, spacing in zip(rotation, origins, spacings)
    ]

    # use the roi-box to extract the corresponding array elements.
    arrays = [sitk.GetArrayFromImage(image).transpose((1, 2, 0)) for image in images]
    box_indices = []
    for ib, array in zip(img_coord_rois, arrays):
        img_indices = []
        low, up = np.amin(ib, axis=0), np.amax(ib, axis=0)
        # sometimes the prostate is centered on all images.
        # add a security margin.
        low = low - 20
        up = up + 20
        for pos, dim in enumerate(array.shape):

            def in_array(in_int, dim):
                in_int = int(in_int)
                in_int = 0 if in_int < 0 else in_int
                in_int = dim if in_int > dim else in_int
                return in_int

            img_indices.append(slice(in_array(low[pos], dim), in_array(up[pos], dim)))
        box_indices.append(img_indices)

    intersections = [i[tuple(box_inds)] for box_inds, i in zip(box_indices, arrays)]

    if False:
        # plot rects
        names = ["tra", "cor", "sag", "roi"]
        color_keys = list(mcolors.TABLEAU_COLORS.keys())
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for pos, rect in enumerate(rects_stacked):
            color = color_keys[pos % len(color_keys)]
            for linepos, line in enumerate(rect):
                if linepos == 0:
                    ax.plot(
                        line[0, 0],
                        line[0, 1],
                        line[0, 2],
                        "s",
                        color=color,
                        label=names[pos],
                    )
                ax.plot(line[:, 0], line[:, 1], line[:, 2], "-.", color=color)
        plt.legend()
        plt.savefig("test1.png")

        for pos, rect in enumerate(rects_stacked):
            color = color_keys[pos % len(color_keys)]
            for linepos, line in enumerate(rect):
                if linepos == 0:
                    plt.plot(line[0, 0], line[0, 1], "s", color=color, label=names[pos])
                plt.plot(line[:, 0], line[:, 1], "-.", color=color)
        plt.legend(loc="upper right")
        plt.title("X,Y-View")
        plt.savefig("test2.png")

        for pos, rect in enumerate(rects_stacked):
            color = color_keys[pos % len(color_keys)]
            for linepos, line in enumerate(rect):
                if linepos == 0:
                    plt.plot(line[0, 1], line[0, 2], "s", color=color, label=names[pos])
                plt.plot(line[:, 1], line[:, 2], "-.", color=color)
        plt.legend(loc="upper right")
        plt.title("Y-Z-View")
        plt.show()

        # img_coord_tra = img_coord_rois[0]
        # sitk getShape and GetArrayFromImage return transposed results.

        plt.imshow(intersections[0][:, :, 10])
        plt.show()

    return intersections, box_indices


def softmax_focal_loss(
    logits: th.Tensor,
    labels: th.Tensor,
    alpha: th.Tensor,
    gamma: float = 2,
) -> th.Tensor:
    """Compute a softmax focal loss."""
    # chex.assert_type([logits], float)
    # # see also the original sigmoid implementation at:
    # # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    # chex.assert_type([logits], float)
    # focus = jnp.power(1.0 - jax.nn.softmax(logits, axis=-1), gamma)
    # loss = -labels * focus * alpha * jax.nn.log_softmax(logits, axis=-1)
    # return jnp.sum(loss, axis=-1)
    logits = logits.float()
    labels = labels.float()
    # TODO: Implement softmax focal loss.
    return th.tensor(0.0)
