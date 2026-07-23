import numpy as np
from anylabeling.services.auto_labeling.__base__.sam2 import iter_point_batches


def test_iter_point_batches_splits_evenly():
    pts = np.arange(20).reshape(10, 2)
    batches = list(iter_point_batches(pts, 4))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert np.array_equal(np.concatenate(batches, axis=0), pts)


def test_iter_point_batches_single_batch():
    pts = np.arange(6).reshape(3, 2)
    batches = list(iter_point_batches(pts, 64))
    assert len(batches) == 1
    assert np.array_equal(batches[0], pts)
