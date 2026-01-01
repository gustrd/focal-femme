"""File renaming logic for organizing photos by cluster."""

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .utils import ClusterState, format_cluster_id, generate_safe_filename, parse_cluster_prefix

logger = logging.getLogger(__name__)


@dataclass
class RenameOperation:
    """Represents a single file rename operation."""

    source: Path
    destination: Path
    cluster_id: int

    @property
    def source_name(self) -> str:
        return self.source.name

    @property
    def destination_name(self) -> str:
        return self.destination.name


@dataclass
class RenameResult:
    """Result of a rename operation."""

    success: bool
    operation: RenameOperation
    error: str | None = None


class FileRenamer:
    """Handles file renaming based on cluster assignments."""

    def __init__(self, dry_run: bool = False):
        """
        Initialize the renamer.

        Args:
            dry_run: If True, don't actually rename files
        """
        self.dry_run = dry_run

    def plan_renames(
        self,
        state: ClusterState,
        folder: Path,
        normalized_beauty_scores: dict[int, int] | None = None,
    ) -> list[RenameOperation]:
        """
        Plan rename operations based on cluster assignments.

        Args:
            state: ClusterState with cluster assignments
            folder: Target folder containing the files
            normalized_beauty_scores: Optional dict mapping cluster_id to normalized score (0-99)

        Returns:
            List of RenameOperation objects
        """
        operations: list[RenameOperation] = []

        if normalized_beauty_scores is None:
            normalized_beauty_scores = {}

        # Get set of all current filenames (for collision detection)
        existing_files = {f.name for f in folder.iterdir() if f.is_file()}

        # Track new filenames we're planning to use
        planned_names: set[str] = set()

        for file_path_str, face_data in state.faces.items():
            file_path = Path(file_path_str)

            if face_data.cluster_id is None:
                continue

            if not file_path.exists():
                logger.warning(f"File no longer exists: {file_path}")
                continue

            # Get normalized beauty score for this cluster
            beauty_score = normalized_beauty_scores.get(face_data.cluster_id, 0)

            # Check if file already has the correct prefix
            current_prefix, _ = parse_cluster_prefix(file_path.name)
            expected_prefix = format_cluster_id(face_data.cluster_id, beauty_score)

            if current_prefix == expected_prefix:
                # Already has correct prefix, skip
                continue

            # Generate new filename
            # Exclude current file from collision check, include planned names
            collision_set = (existing_files - {file_path.name}) | planned_names

            new_name = generate_safe_filename(
                file_path,
                face_data.cluster_id,
                collision_set,
                beauty_score,
            )

            new_path = file_path.parent / new_name
            planned_names.add(new_name)

            operations.append(RenameOperation(
                source=file_path,
                destination=new_path,
                cluster_id=face_data.cluster_id,
            ))

        return operations

    def execute_rename(self, operation: RenameOperation) -> RenameResult:
        """
        Execute a single rename operation.

        Args:
            operation: The rename operation to execute

        Returns:
            RenameResult indicating success or failure
        """
        if self.dry_run:
            return RenameResult(success=True, operation=operation)

        try:
            # Use shutil.move for cross-filesystem compatibility
            shutil.move(str(operation.source), str(operation.destination))
            return RenameResult(success=True, operation=operation)
        except OSError as e:
            return RenameResult(
                success=False,
                operation=operation,
                error=str(e),
            )

    def execute_all(
        self,
        operations: list[RenameOperation],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[RenameResult]:
        """
        Execute all rename operations.

        Args:
            operations: List of rename operations
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of RenameResult objects
        """
        results: list[RenameResult] = []
        total = len(operations)

        for idx, operation in enumerate(operations):
            result = self.execute_rename(operation)
            results.append(result)

            if not result.success:
                logger.error(
                    f"Failed to rename {operation.source_name}: {result.error}"
                )

            if progress_callback:
                progress_callback(idx + 1, total)

        return results

    def update_state_paths(
        self,
        state: ClusterState,
        results: list[RenameResult],
    ) -> None:
        """
        Update ClusterState with new file paths after successful renames.

        Args:
            state: ClusterState to update
            results: List of RenameResult from executed operations
        """
        for result in results:
            if not result.success:
                continue

            old_path = str(result.operation.source)
            new_path = str(result.operation.destination)

            if old_path in state.faces:
                # Move the face data to the new path key
                face_data = state.faces.pop(old_path)
                face_data.file_path = result.operation.destination
                state.faces[new_path] = face_data


def summarize_operations(operations: list[RenameOperation]) -> dict[int, int]:
    """
    Summarize rename operations by cluster.

    Args:
        operations: List of rename operations

    Returns:
        Dictionary mapping cluster_id to count of files
    """
    summary: dict[int, int] = {}
    for op in operations:
        if op.cluster_id not in summary:
            summary[op.cluster_id] = 0
        summary[op.cluster_id] += 1
    return summary


def summarize_results(results: list[RenameResult]) -> tuple[int, int]:
    """
    Summarize rename results.

    Args:
        results: List of rename results

    Returns:
        Tuple of (success_count, failure_count)
    """
    success = sum(1 for r in results if r.success)
    failure = len(results) - success
    return success, failure
