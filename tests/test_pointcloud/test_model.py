from pathlib import Path

import numpy as np
import pytest

from anylabeling.views.labeling.pointcloud.model import (
    AnnotationDocument,
    Frame,
    validate_labels,
)


def document(labels):
    return AnnotationDocument(
        Frame(
            Path("frame.bin"),
            np.zeros((len(labels), 4), dtype=np.float32),
            np.asarray(labels, dtype=np.uint32),
        )
    )


def packed(semantic, instance):
    return (instance << 16) | semantic


def test_semantic_assignment_preserves_same_class_instances_and_unknown_ids():
    original = [
        0,
        packed(10, 7),
        packed(30, 7),
        packed(60000, 65535),
        packed(0, 9),
    ]
    doc = document(original)
    assert doc.assign_semantic(range(5), 10) == 2
    np.testing.assert_array_equal(
        doc.labels, [10, original[1], original[2], original[3], 10]
    )
    assert doc.assign_semantic([1], 10, overwrite=True) == 0
    assert doc.assign_semantic([2], 10, overwrite=True) == 1
    assert doc.labels[2] == 10
    assert doc.undo()
    assert doc.labels[2] == original[2]
    assert doc.undo()
    np.testing.assert_array_equal(doc.labels, original)
    assert not doc.dirty


def test_instance_lifecycle_uses_combined_identity_and_preserves_semantics():
    doc = document([10, 10, 10, 10, packed(30, 1), packed(60000, 500)])
    original = doc.labels.copy()
    first = doc.create_instance([0, 1, 4], 10)
    assert first == (10, 1)
    second = doc.create_instance([2], 10)
    assert second == (10, 2)
    assert doc.add_to_instance([1, 3, 4], second) == 2
    assert doc.remove_from_instance([0, 1, 4], second) == 1
    split = doc.split_instance([2], second)
    assert split == (10, 3)
    assert doc.merge_instances(first, [second, split]) == 2
    assert doc.instance_counts() == {(10, 1): 3, (30, 1): 1, (60000, 500): 1}
    assert doc.delete_instance(first) == 3
    np.testing.assert_array_equal(doc.labels, original)
    assert not doc.dirty
    snapshots = []
    while doc.can_undo:
        snapshots.append(doc.labels.copy())
        doc.undo()
    np.testing.assert_array_equal(doc.labels, original)
    for expected in reversed(snapshots):
        assert doc.redo()
        np.testing.assert_array_equal(doc.labels, expected)


def test_invalid_instance_operations_are_atomic():
    doc = document([packed(10, 1), packed(10, 1), packed(30, 1), 0])
    original = doc.labels.copy()
    operations = [
        lambda: doc.create_instance([], 10),
        lambda: doc.create_instance([3], 0),
        lambda: doc.create_instance([2], 10),
        lambda: doc.split_instance([], (10, 1)),
        lambda: doc.split_instance([0, 1], (10, 1)),
        lambda: doc.merge_instances((10, 1), [(30, 1)]),
        lambda: doc.merge_instances((10, 1), [(10, 1)]),
        lambda: doc.merge_instances((10, 1), [(10, 2)]),
        lambda: doc.add_to_instance([0], (10, 3)),
        lambda: doc.remove_from_instance([0], (10, 0)),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
        np.testing.assert_array_equal(doc.labels, original)
        assert not doc.dirty
        assert not doc.can_undo


def test_twenty_operations_save_baseline_and_new_branch():
    doc = document([0] * 25)
    snapshots = [doc.labels.copy()]
    for index in range(25):
        doc.assign_semantic([index], 10)
        if index % 2 == 0:
            doc.create_instance([index], 10)
        snapshots.append(doc.labels.copy())
    final = doc.labels.copy()
    doc.mark_saved(Path("output.label"))
    assert not doc.dirty
    assert doc.frame.label_path == Path("output.label")
    for _ in range(38):
        assert doc.undo()
    assert not doc.can_undo
    assert doc.dirty
    np.testing.assert_array_equal(doc.labels, np.zeros(25, dtype=np.uint32))
    for _ in range(38):
        assert doc.redo()
    np.testing.assert_array_equal(doc.labels, final)
    assert not doc.dirty
    doc.undo()
    assert doc.can_redo
    assert doc.assign_semantic([0], 10, overwrite=True) == 0
    assert doc.can_redo
    assert doc.clear([0]) == 1
    assert not doc.can_redo


def test_dirty_is_content_based_beyond_history_limit():
    doc = AnnotationDocument(document([0]).frame, history_limit=20)
    for class_id in range(1, 26):
        doc.assign_semantic([0], class_id, overwrite=True)
    for _ in range(20):
        assert doc.undo()
    assert not doc.can_undo
    assert doc.dirty
    doc.clear([0])
    assert not doc.dirty
    doc.redo()
    assert not doc.dirty
    doc.mark_saved()
    doc.assign_semantic([0], 10)
    doc.assign_semantic([0], 0, overwrite=True)
    assert not doc.dirty


def test_id_exhaustion_and_maximum_id_no_wraparound():
    labels = (np.arange(1, 65536, dtype=np.uint32) << 16) | 65535
    doc = document(np.append(labels, np.uint32(65535)))
    original = doc.labels.copy()
    with pytest.raises(ValueError, match="No instance IDs"):
        doc.create_instance([65535], 65535)
    np.testing.assert_array_equal(doc.labels, original)
    doc.delete_instance((65535, 65535))
    assert doc.create_instance([65535], 65535) == (65535, 65535)
    assert doc.labels[-1] == 0xFFFFFFFF
    doc.undo()
    doc.undo()
    np.testing.assert_array_equal(doc.labels, original)


def test_special_zero_semantic_instance_is_preserved_until_explicit_edit():
    doc = document([packed(0, 7), packed(0, 7)])
    assert doc.instance_counts() == {(0, 7): 2}
    assert doc.assign_semantic([0], 0) == 0
    with pytest.raises(ValueError):
        doc.add_to_instance([1], (0, 7))
    assert doc.remove_from_instance([0], (0, 7)) == 1
    assert doc.delete_instance((0, 7)) == 1
    np.testing.assert_array_equal(doc.labels, [0, 0])
    doc.undo()
    doc.undo()
    np.testing.assert_array_equal(doc.labels, [packed(0, 7), packed(0, 7)])


@pytest.mark.parametrize("value", [-1, 65536, 2.5, True, "10"])
def test_invalid_semantic_ids_rejected_without_edit(value):
    doc = document([0])
    with pytest.raises(ValueError):
        doc.assign_semantic([0], value)
    assert not doc.dirty


@pytest.mark.parametrize("indices", [[-1], [3], [0.5], [[0]], [True, False]])
def test_invalid_point_selection_rejected(indices):
    doc = document([0, 0, 0])
    with pytest.raises(ValueError):
        doc.clear(indices)


def test_selection_duplicates_masks_and_read_only_arrays():
    doc = document([0, 0, 0])
    assert doc.assign_semantic([0, 0, 2], 10) == 2
    assert doc.clear(np.array([True, False, False])) == 1
    with pytest.raises(ValueError):
        doc.labels[0] = 4
    with pytest.raises(ValueError):
        doc.frame.labels[0] = 4
    with pytest.raises(ValueError):
        doc.frame.points[0, 0] = 1


@pytest.mark.parametrize(
    "labels",
    [
        np.array([-1]),
        np.array([2**32]),
        np.array([1.5]),
        np.zeros((2, 1), dtype=int),
    ],
)
def test_invalid_labels_never_silently_wrap(labels):
    with pytest.raises(ValueError):
        validate_labels(labels)


def test_label_views_are_live_read_only_and_snapshots_remain_independent():
    doc = document([packed(10, 7), packed(65535, 65535), packed(0, 9)])
    semantic = doc.semantic
    instance = doc.instance
    semantic_view = doc.semantic_view
    instance_view = doc.instance_view
    counts = doc.semantic_counts()
    assert semantic.dtype == instance.dtype == np.uint32
    assert semantic_view.dtype == instance_view.dtype == np.uint16
    assert counts.dtype == np.int64
    assert counts.shape == (65536,)
    assert np.shares_memory(semantic_view, doc.labels)
    assert np.shares_memory(instance_view, doc.labels)
    np.testing.assert_array_equal(semantic_view, semantic)
    np.testing.assert_array_equal(instance_view, instance)
    for values in (semantic_view, instance_view, counts):
        with pytest.raises(ValueError):
            values[0] = 0
    semantic[:] = 1
    instance[:] = 1
    np.testing.assert_array_equal(
        doc.labels, [packed(10, 7), 0xFFFFFFFF, packed(0, 9)]
    )
    doc.assign_semantic([0, 2], 30, overwrite=True)
    np.testing.assert_array_equal(semantic_view, [30, 65535, 30])
    np.testing.assert_array_equal(instance_view, [0, 65535, 0])
    assert counts[30] == 2
    assert counts[0] == counts[10] == 0
    assert counts[65535] == 1
    np.testing.assert_array_equal(semantic, [1, 1, 1])
    np.testing.assert_array_equal(instance, [1, 1, 1])
    doc.undo()
    np.testing.assert_array_equal(semantic_view, [10, 65535, 0])
    np.testing.assert_array_equal(instance_view, [7, 65535, 9])


def test_edit_revisions_track_actual_changes_and_preserve_history():
    doc = document([0, 10, 0])
    assert doc.revision == 0
    assert not doc.last_changed_indices.size
    doc.assign_semantic([2, 0, 0, 1], 10)
    assert doc.revision == 1
    changed = doc.last_changed_indices
    np.testing.assert_array_equal(changed, [0, 2])
    with pytest.raises(ValueError):
        changed[0] = 2
    doc.assign_semantic([1], 10)
    assert doc.revision == 1
    doc.mark_saved()
    assert doc.revision == 1
    assert doc.undo()
    assert doc.revision == 2
    np.testing.assert_array_equal(doc.last_changed_indices, [0, 2])
    doc.clear([0, 2])
    assert doc.revision == 2
    assert doc.can_redo
    assert doc.redo()
    assert doc.revision == 3
    assert not doc.redo()
    assert doc.revision == 3
    doc.clear([1])
    assert doc.revision == 4
    np.testing.assert_array_equal(doc.last_changed_indices, [1])
    np.testing.assert_array_equal(changed, [0, 2])


def test_live_view_metadata_changes_do_not_change_document_views():
    doc = document([packed(10, 7), packed(65535, 65535)])
    for values in (doc.semantic_view, doc.instance_view):
        values.flags.writeable = True
    assert not doc.semantic_view.flags.writeable
    assert not doc.instance_view.flags.writeable
    assert doc.semantic_view.dtype == np.uint16
    assert doc.instance_view.dtype == np.uint16
    np.testing.assert_array_equal(doc.semantic_view, [10, 65535])
    np.testing.assert_array_equal(doc.instance_view, [7, 65535])


def test_cached_counts_follow_edits_save_undo_redo_and_history_eviction():
    rng = np.random.default_rng(195)
    classes = np.array([0, 10, 30, 60000, 65535], dtype=np.uint32)
    labels = rng.choice(classes, 300) | (
        rng.choice(np.array([0, 1, 7, 65535], dtype=np.uint32), 300) << 16
    )
    doc = AnnotationDocument(document(labels).frame, history_limit=20)

    def assert_counts():
        labels = doc.labels
        expected = np.bincount(labels & 65535, minlength=65536)
        np.testing.assert_array_equal(doc.semantic_counts(), expected)
        values, counts = np.unique(labels, return_counts=True)
        expected = {
            (int(value) & 65535, int(value) >> 16): int(count)
            for value, count in zip(values, counts)
            if int(value) >> 16
        }
        assert doc.instance_counts() == expected
        snapshot = doc.instance_counts()
        snapshot.clear()
        assert doc.instance_counts() == expected

    assert_counts()
    for iteration in range(65):
        indices = rng.integers(0, len(labels), size=20)
        semantic_id = int(rng.choice(classes))
        doc.assign_semantic(indices, semantic_id, overwrite=True)
        assert_counts()
        if semantic_id:
            key = doc.create_instance(indices, semantic_id)
            assert_counts()
            doc.remove_from_instance(indices[:3], key)
            assert_counts()
        if iteration % 7 == 0:
            doc.mark_saved()
            assert_counts()
    for _ in range(20):
        assert doc.undo()
        assert_counts()
    assert not doc.undo()
    for _ in range(20):
        assert doc.redo()
        assert_counts()
    assert not doc.redo()
