"""CLI entry point for focal-femme."""

import logging
import sys
from pathlib import Path

import click
from tqdm import tqdm

from . import __version__
from .clusterer import FaceClusterer, get_cluster_summary, get_cluster_beauty_scores, normalize_beauty_scores
from .detector import FaceDetector
from .renamer import FileRenamer, summarize_operations, summarize_results
from .utils import (
    ClusterState,
    delete_cluster_state,
    format_cluster_id,
    get_image_files,
    load_cluster_state,
    save_cluster_state,
)


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


@click.command()
@click.argument(
    "folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--eps",
    type=float,
    default=0.4,
    help="DBSCAN epsilon (cosine distance). 0.3=strict, 0.4=moderate, 0.5=loose.",
)
@click.option(
    "--min-samples",
    type=int,
    default=2,
    help="DBSCAN min_samples parameter (default: 2). Minimum images to form a cluster.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without renaming files.",
)
@click.option(
    "--reset",
    is_flag=True,
    help="Clear existing embeddings cache and start fresh.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed processing information.",
)
@click.option(
    "--female-threshold",
    type=float,
    default=0.5,
    help="Minimum confidence for female classification (0-1, default: 0.5).",
)
@click.version_option(version=__version__, prog_name="focal-femme")
def main(
    folder: Path,
    eps: float,
    min_samples: int,
    dry_run: bool,
    reset: bool,
    verbose: bool,
    female_threshold: float,
) -> None:
    """
    Organize photos by clustering faces of the primary female subject.

    FOLDER is the directory containing images to process.

    The tool detects faces in each image, filters for female faces,
    and clusters them using DBSCAN. Files are renamed with cluster-based
    prefixes (e.g., person_001_IMG_1234.jpg).
    """
    from .model_setup import setup_models
    setup_models()
    setup_logging(verbose)
    folder = folder.resolve()

    click.echo(f"Processing folder: {folder}")

    # Handle reset
    if reset:
        if delete_cluster_state(folder):
            click.echo("Cleared existing embeddings cache.")
        else:
            click.echo("No existing cache to clear.")

    # Get image files
    image_files = get_image_files(folder)
    if not image_files:
        click.echo("No supported image files found in the folder.")
        sys.exit(0)

    click.echo(f"Found {len(image_files)} image files.")

    # Load or create state
    state = load_cluster_state(folder)
    if state is None:
        state = ClusterState()
        click.echo("Starting fresh (no existing cache).")
    else:
        click.echo(f"Loaded cache with {len(state.faces)} existing faces.")

    # Determine which files need processing
    existing_paths = {str(f) for f in image_files}
    cached_paths = set(state.faces.keys())

    # Remove entries for files that no longer exist
    removed = [p for p in cached_paths if p not in existing_paths]
    for path in removed:
        del state.faces[path]
    if removed:
        click.echo(f"Removed {len(removed)} entries for deleted files.")

    # Find new files to process
    new_files = [f for f in image_files if str(f) not in state.faces]

    if new_files:
        click.echo(f"Processing {len(new_files)} new images...")

        # Initialize detector
        detector = FaceDetector(female_threshold=female_threshold)

        # Process images with progress bar
        with tqdm(total=len(new_files), desc="Detecting faces", unit="img") as pbar:
            def progress_callback(current: int, total: int) -> None:
                pbar.update(1)

            new_faces = detector.process_images(new_files, progress_callback, verbose=verbose)

        click.echo(f"Found primary female faces in {len(new_faces)} images.")

        # Merge new faces into state
        for path_str, face_data in new_faces.items():
            state.faces[path_str] = face_data
    else:
        click.echo("All images already processed.")

    # Check if we have any faces to cluster
    if not state.faces:
        click.echo("No female faces found in any images. Nothing to cluster.")
        save_cluster_state(state, folder)
        sys.exit(0)

    # Perform clustering
    click.echo("Clustering faces...")
    clusterer = FaceClusterer(eps=eps, min_samples=min_samples)
    cluster_result = clusterer.cluster(state)
    clusterer.update_state_with_clusters(state, cluster_result)

    # Show cluster summary
    cluster_summary = get_cluster_summary(state)
    beauty_scores = get_cluster_beauty_scores(state)
    normalized_scores = normalize_beauty_scores(beauty_scores)
    click.echo(f"\nClustering complete:")
    click.echo(f"  - {cluster_result.num_clusters} clusters formed")
    click.echo(f"  - {cluster_result.num_noise} images as individual clusters (noise)")
    click.echo(f"  - {len(state.faces)} total images with faces")

    # Always show cluster breakdown with beauty scores
    click.echo("\nCluster breakdown:")
    for cluster_id in sorted(cluster_summary.keys()):
        files = cluster_summary[cluster_id]
        beauty = beauty_scores.get(cluster_id, 0.0)
        norm_score = normalized_scores.get(cluster_id, 0)
        click.echo(f"  {format_cluster_id(cluster_id, norm_score)}: {len(files)} images, avg beauty: {beauty:.2f} (norm: {norm_score})")

    # Plan rename operations
    renamer = FileRenamer(dry_run=dry_run)
    operations = renamer.plan_renames(state, folder, normalized_scores)

    if not operations:
        click.echo("\nNo files need renaming (all already have correct prefixes).")
        save_cluster_state(state, folder)
        sys.exit(0)

    # Show planned operations
    op_summary = summarize_operations(operations)
    click.echo(f"\nPlanned renames: {len(operations)} files")

    if dry_run:
        click.echo("\n[DRY RUN] Would rename:")
        for op in operations:
            click.echo(f"  {op.source_name} -> {op.destination_name}")
        save_cluster_state(state, folder)
        sys.exit(0)

    # Confirm before renaming
    if not click.confirm("\nProceed with renaming?"):
        click.echo("Aborted.")
        save_cluster_state(state, folder)
        sys.exit(0)

    # Execute renames
    click.echo("Renaming files...")
    with tqdm(total=len(operations), desc="Renaming", unit="file") as pbar:
        def rename_progress(current: int, total: int) -> None:
            pbar.update(1)

        results = renamer.execute_all(operations, rename_progress)

    # Update state with new paths
    renamer.update_state_paths(state, results)

    # Show results
    success, failure = summarize_results(results)
    click.echo(f"\nRenaming complete: {success} succeeded, {failure} failed")

    if failure > 0:
        click.echo("\nFailed operations:")
        for result in results:
            if not result.success:
                click.echo(f"  {result.operation.source_name}: {result.error}")

    # Save state
    save_cluster_state(state, folder)
    click.echo(f"\nCache saved. Run again to process new images incrementally.")


if __name__ == "__main__":
    main()
