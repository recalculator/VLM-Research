"""
Configuration and path management for VLM Research experiment.

This module provides workspace-relative path resolution to ensure
the code runs from any directory.
"""

import os
import sys
from pathlib import Path


def get_workspace_root():
    """
    Get the workspace root directory (where this config.py file is located).

    Returns:
        Path: Absolute path to workspace root
    """
    return Path(__file__).parent.resolve()


def get_fastv_path():
    """Get path to FastV repository."""
    return get_workspace_root() / "FastV"


def get_prumerge_path():
    """Get path to LLaVA-PruMerge repository."""
    return get_workspace_root() / "LLaVA-PruMerge"


def get_outputs_path():
    """Get path to outputs directory (created if doesn't exist)."""
    outputs = get_workspace_root() / "outputs"
    outputs.mkdir(exist_ok=True)
    return outputs


def get_visualizations_path():
    """Get path to visualizations directory (created if doesn't exist)."""
    viz_path = get_outputs_path() / "visualizations"
    viz_path.mkdir(exist_ok=True)
    return viz_path


def setup_python_paths():
    """
    Add repository paths to Python sys.path for imports.

    This must be called before importing from FastV or PruMerge.
    """
    workspace = get_workspace_root()
    fastv_path = get_fastv_path()
    prumerge_path = get_prumerge_path()

    # Add FastV paths
    fastv_transformers = fastv_path / "src" / "transformers" / "src"
    fastv_src = fastv_path / "src" / "FastV"

    paths_to_add = [
        str(fastv_transformers),
        str(fastv_src),
        str(prumerge_path),
    ]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    return {
        "workspace": workspace,
        "fastv": fastv_path,
        "prumerge": prumerge_path,
        "fastv_transformers": fastv_transformers,
        "fastv_src": fastv_src,
    }


def check_repositories_exist():
    """
    Check if required repositories are cloned.

    Returns:
        dict: Status of each repository
    """
    fastv_exists = get_fastv_path().exists()
    prumerge_exists = get_prumerge_path().exists()

    return {
        "fastv": fastv_exists,
        "prumerge": prumerge_exists,
        "all_present": fastv_exists and prumerge_exists
    }


def check_modifications_applied():
    """
    Check if code modifications are applied to repositories.

    Returns:
        dict: Status of modifications
    """
    status = {}

    # Check PruMerge modification
    prumerge_file = get_prumerge_path() / "llava" / "model" / "multimodal_encoder" / "clip_encoder.py"
    if prumerge_file.exists():
        content = prumerge_file.read_text()
        status["prumerge_modified"] = "kept_token_indices" in content
    else:
        status["prumerge_modified"] = False

    # Check FastV modification
    fastv_file = get_fastv_path() / "src" / "transformers" / "src" / "transformers" / "models" / "llama" / "modeling_llama.py"
    if fastv_file.exists():
        content = fastv_file.read_text()
        status["fastv_modified"] = "kept_visual_token_indices" in content
    else:
        status["fastv_modified"] = False

    status["all_modified"] = status.get("prumerge_modified", False) and status.get("fastv_modified", False)

    return status


if __name__ == "__main__":
    print("Workspace Configuration")
    print("=" * 60)
    print(f"Workspace root: {get_workspace_root()}")
    print(f"FastV path: {get_fastv_path()}")
    print(f"PruMerge path: {get_prumerge_path()}")
    print(f"Outputs path: {get_outputs_path()}")
    print()

    print("Repository Status")
    print("=" * 60)
    repo_status = check_repositories_exist()
    print(f"FastV present: {repo_status['fastv']}")
    print(f"PruMerge present: {repo_status['prumerge']}")
    print()

    print("Modification Status")
    print("=" * 60)
    mod_status = check_modifications_applied()
    print(f"PruMerge modified: {mod_status.get('prumerge_modified', False)}")
    print(f"FastV modified: {mod_status.get('fastv_modified', False)}")
    print()

    if repo_status['all_present'] and mod_status.get('all_modified', False):
        print("✓ All checks passed! Ready to run experiments.")
    else:
        print("✗ Some checks failed. See above for details.")
