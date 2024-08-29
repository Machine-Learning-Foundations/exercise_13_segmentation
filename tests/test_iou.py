"""Test the IoU compute function."""

import sys

sys.path.insert(0, "./src")

import pytest
import torch as th

from src.data_loader import Loader
from src.meanIoU import compute_iou

input_records = ["ProstateX-0004", "ProstateX-0007", "ProstateX-0311"]


@pytest.mark.offline
@pytest.mark.parametrize("input_record", input_records)
def test_iou(input_record: str) -> None:
    """Test for IoU.

    Args:
        input_record (str): Input record for test
    """
    loader = Loader(input_shape=(128, 128, 21))
    record = loader.get_record(input_record)
    target = record["annotation"].unsqueeze(0)
    iou = compute_iou(target, target)
    assert iou == 1.0


def test_zero_iou() -> None:
    """Test zero IoU."""
    loader = Loader(input_shape=(128, 128, 21))
    record = loader.get_record("ProstateX-0311")
    target = record["annotation"].unsqueeze(0)
    black_img = th.zeros_like(target)
    iou = compute_iou(target, black_img)
    # IOU cannot be zero, as background is also zeros.
    assert iou < 0.3
