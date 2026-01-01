"""Tests for renamer module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from focal_femme.renamer import (
    FileRenamer,
    RenameOperation,
    RenameResult,
    summarize_operations,
    summarize_results,
)
from focal_femme.utils import ClusterState, FaceData


def create_face_data(file_path: Path, cluster_id: int) -> FaceData:
    """Helper to create FaceData with cluster assignment."""
    return FaceData(
        file_path=file_path,
        embedding=np.array([1.0, 0.0, 0.0]),
        bbox=(0, 100, 100, 0),
        is_female=True,
        cluster_id=cluster_id,
    )


class TestFileRenamer:
    def test_plan_renames_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            # Create test file
            test_file = folder / "IMG_1234.jpg"
            test_file.touch()

            state = ClusterState()
            state.faces[str(test_file)] = create_face_data(test_file, 1)

            renamer = FileRenamer()
            operations = renamer.plan_renames(state, folder)

            assert len(operations) == 1
            assert operations[0].source == test_file
            assert operations[0].destination.name == "person_001_IMG_1234.jpg"

    def test_skip_already_prefixed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            # Create file with correct prefix
            test_file = folder / "person_001_IMG_1234.jpg"
            test_file.touch()

            state = ClusterState()
            state.faces[str(test_file)] = create_face_data(test_file, 1)

            renamer = FileRenamer()
            operations = renamer.plan_renames(state, folder)

            assert len(operations) == 0

    def test_reprefix_wrong_cluster(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            # Create file with wrong prefix
            test_file = folder / "person_002_IMG_1234.jpg"
            test_file.touch()

            state = ClusterState()
            state.faces[str(test_file)] = create_face_data(test_file, 1)

            renamer = FileRenamer()
            operations = renamer.plan_renames(state, folder)

            assert len(operations) == 1
            assert operations[0].destination.name == "person_001_IMG_1234.jpg"

    def test_handle_collisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            # Create two files that would have the same prefixed name
            file1 = folder / "IMG_1234.jpg"
            file1.touch()
            file2 = folder / "subdir_IMG_1234.jpg"
            file2.touch()

            state = ClusterState()
            state.faces[str(file1)] = create_face_data(file1, 1)
            state.faces[str(file2)] = create_face_data(file2, 1)

            renamer = FileRenamer()
            operations = renamer.plan_renames(state, folder)

            # Both files should be renamed
            assert len(operations) == 2

            # Destination names should be unique
            dest_names = {op.destination.name for op in operations}
            assert len(dest_names) == 2

    def test_dry_run_no_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            test_file = folder / "IMG_1234.jpg"
            test_file.touch()

            state = ClusterState()
            state.faces[str(test_file)] = create_face_data(test_file, 1)

            renamer = FileRenamer(dry_run=True)
            operations = renamer.plan_renames(state, folder)
            results = renamer.execute_all(operations)

            # File should not be renamed
            assert test_file.exists()
            assert not (folder / "person_001_IMG_1234.jpg").exists()
            assert all(r.success for r in results)

    def test_execute_renames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            test_file = folder / "IMG_1234.jpg"
            test_file.write_text("test content")

            state = ClusterState()
            state.faces[str(test_file)] = create_face_data(test_file, 1)

            renamer = FileRenamer()
            operations = renamer.plan_renames(state, folder)
            results = renamer.execute_all(operations)

            assert len(results) == 1
            assert results[0].success

            # Original should be gone, new file should exist
            assert not test_file.exists()
            new_file = folder / "person_001_IMG_1234.jpg"
            assert new_file.exists()
            assert new_file.read_text() == "test content"

    def test_update_state_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            test_file = folder / "IMG_1234.jpg"
            test_file.touch()

            state = ClusterState()
            state.faces[str(test_file)] = create_face_data(test_file, 1)

            renamer = FileRenamer()
            operations = renamer.plan_renames(state, folder)
            results = renamer.execute_all(operations)
            renamer.update_state_paths(state, results)

            # State should now have new path as key
            new_path = folder / "person_001_IMG_1234.jpg"
            assert str(test_file) not in state.faces
            assert str(new_path) in state.faces


class TestRenameOperation:
    def test_properties(self):
        op = RenameOperation(
            source=Path("/photos/IMG_1234.jpg"),
            destination=Path("/photos/person_001_IMG_1234.jpg"),
            cluster_id=1,
        )

        assert op.source_name == "IMG_1234.jpg"
        assert op.destination_name == "person_001_IMG_1234.jpg"


class TestSummarizeOperations:
    def test_empty(self):
        summary = summarize_operations([])
        assert summary == {}

    def test_multiple_clusters(self):
        operations = [
            RenameOperation(Path("a.jpg"), Path("x.jpg"), 1),
            RenameOperation(Path("b.jpg"), Path("y.jpg"), 1),
            RenameOperation(Path("c.jpg"), Path("z.jpg"), 2),
        ]

        summary = summarize_operations(operations)

        assert summary == {1: 2, 2: 1}


class TestSummarizeResults:
    def test_all_success(self):
        results = [
            RenameResult(success=True, operation=RenameOperation(Path("a"), Path("b"), 1)),
            RenameResult(success=True, operation=RenameOperation(Path("c"), Path("d"), 1)),
        ]

        success, failure = summarize_results(results)

        assert success == 2
        assert failure == 0

    def test_mixed_results(self):
        results = [
            RenameResult(success=True, operation=RenameOperation(Path("a"), Path("b"), 1)),
            RenameResult(success=False, operation=RenameOperation(Path("c"), Path("d"), 1), error="test"),
            RenameResult(success=True, operation=RenameOperation(Path("e"), Path("f"), 1)),
        ]

        success, failure = summarize_results(results)

        assert success == 2
        assert failure == 1
