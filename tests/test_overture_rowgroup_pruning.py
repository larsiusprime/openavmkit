"""Row-group pruning for the Overture buildings fetch (see _prune_dataset_to_bbox).

Uses small local Parquet files shaped like the Overture buildings theme (a ``bbox``
struct of float32 xmin/xmax/ymin/ymax) so the pruning logic runs without network access.
"""
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.parquet as pq

from openavmkit.utilities.overture import _prune_dataset_to_bbox, _row_group_may_overlap


BBOX_TYPE = pa.struct([
    ("xmin", pa.float32()), ("xmax", pa.float32()),
    ("ymin", pa.float32()), ("ymax", pa.float32()),
])


def _buildings(ids, x0s):
    """Unit-square buildings at (x0, 0) .. (x0 + 1, 1)."""
    return pa.table({
        "id": pa.array(ids, pa.string()),
        "bbox": pa.array(
            [{"xmin": x, "xmax": x + 1, "ymin": 0.0, "ymax": 1.0} for x in x0s], BBOX_TYPE
        ),
    })


def _write(path, table, row_group_size, write_statistics=True):
    pq.write_table(table, path, row_group_size=row_group_size, write_statistics=write_statistics)


def _bbox_filter(bbox):
    xmin, ymin, xmax, ymax = bbox
    return (
        (pc.field("bbox", "xmin") < xmax)
        & (pc.field("bbox", "xmax") > xmin)
        & (pc.field("bbox", "ymin") < ymax)
        & (pc.field("bbox", "ymax") > ymin)
    )


def _ids(dataset, bbox):
    table = dataset.to_table(filter=_bbox_filter(bbox), columns=["id"])
    return sorted(table.column("id").to_pylist())


def _scanned_row_groups(dataset):
    return sum(len(frag.row_groups) for frag in dataset.get_fragments())


def _make_dataset(tmp_path):
    # File a: x in 0..9, file b: x in 100..109; 2 buildings per row group -> 5 row groups each.
    _write(tmp_path / "a.parquet", _buildings([f"a{i}" for i in range(10)], range(10)), 2)
    _write(tmp_path / "b.parquet", _buildings([f"b{i}" for i in range(10)], range(100, 110)), 2)
    return ds.dataset(str(tmp_path), format="parquet", filesystem=fs.LocalFileSystem())


def test_pruning_keeps_only_overlapping_row_groups(tmp_path):
    dataset = _make_dataset(tmp_path)
    bbox = (2.5, 0.25, 3.5, 0.75)  # overlaps buildings a2 and a3, both in row group 1 of a

    pruned = _prune_dataset_to_bbox(dataset, bbox)

    assert _scanned_row_groups(pruned) == 1
    assert [frag.path.rsplit("/", 1)[-1] for frag in pruned.get_fragments()] == ["a.parquet"]
    assert _ids(pruned, bbox) == _ids(dataset, bbox) == ["a2", "a3"]


def test_pruning_matches_unpruned_scan_across_files_and_row_groups(tmp_path):
    dataset = _make_dataset(tmp_path)
    for bbox in [
        (-50.0, -1.0, 200.0, 2.0),   # everything
        (3.5, 0.0, 6.5, 1.0),        # spans row groups 1..3 of a
        (8.5, 0.0, 101.5, 1.0),      # spans both files
        (0.0, 0.0, 1.0, 1.0),        # touching edges only: strict inequalities
        (20.0, 0.0, 30.0, 1.0),      # gap between files
        (2.5, 5.0, 3.5, 6.0),        # right x, wrong y
    ]:
        assert _ids(_prune_dataset_to_bbox(dataset, bbox), bbox) == _ids(dataset, bbox), bbox


def test_no_matching_row_groups_gives_empty_scan(tmp_path):
    dataset = _make_dataset(tmp_path)
    bbox = (20.0, 0.0, 30.0, 1.0)

    pruned = _prune_dataset_to_bbox(dataset, bbox)

    assert _scanned_row_groups(pruned) == 0
    assert pruned.to_table(filter=_bbox_filter(bbox)).num_rows == 0
    assert pruned.schema == dataset.schema


def test_row_groups_without_statistics_are_kept(tmp_path):
    _write(tmp_path / "nostats.parquet", _buildings(["n0", "n1"], [0, 50]), 1, write_statistics=False)
    dataset = ds.dataset(str(tmp_path), format="parquet", filesystem=fs.LocalFileSystem())
    bbox = (49.5, 0.0, 50.5, 1.0)

    pruned = _prune_dataset_to_bbox(dataset, bbox)

    # No statistics means nothing can be ruled out; the row filter still applies.
    assert _scanned_row_groups(pruned) == 2
    assert _ids(pruned, bbox) == ["n1"]


def test_row_group_without_bbox_columns_is_kept(tmp_path):
    pq.write_table(pa.table({"id": ["x"]}), tmp_path / "nobbox.parquet")
    md = pq.read_metadata(tmp_path / "nobbox.parquet")
    rg = md.row_group(0)
    col_index = {rg.column(i).path_in_schema: i for i in range(rg.num_columns)}

    assert _row_group_may_overlap(rg, col_index, (0.0, 0.0, 1.0, 1.0)) is True


def test_empty_dataset_is_returned_unchanged(tmp_path):
    dataset = ds.dataset(str(tmp_path), format="parquet", filesystem=fs.LocalFileSystem())

    assert _prune_dataset_to_bbox(dataset, (0.0, 0.0, 1.0, 1.0)) is dataset
