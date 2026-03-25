import shutil
import time
from pathlib import Path
from typing import Callable

from Bio.SeqRecord import SeqRecord
import numpy as np
import torch
from Bio.PDB.Structure import Structure
from tinwilai.convert import (
    coords_to_result_df,
    label_to_c1p_coords,
    pdb_to_c1p_coords,
    structure_to_c1p_coords,
)
from tinwilai.logger import logger
from tinwilai.tm_score import score


def score_bio(
    tmp_dir: Path,
    usalign_path: Path,
    sequences_csv_path: str,
    labels_csv_path: str,
    target_id: str,
    submission: Structure,
) -> float:
    solution_df = coords_to_result_df(
        target_id,
        *label_to_c1p_coords(
            sequences_csv_path,
            labels_csv_path,
            target_id,
        ),
    )
    submission_df = coords_to_result_df(
        target_id,
        *structure_to_c1p_coords(
            submission,
        ),
    )
    return score(
        tmp_dir,
        usalign_path,
        solution_df,
        submission_df,
        "",
    )


def remkdir(dir_path: Path) -> None:
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir()


def try_run(
    seq_record: SeqRecord,
    name: str,
    run: Callable,
    coords_list: np.ndarray,
    **kwargs,
):
    start_time = time.perf_counter()

    not_filled = np.all(np.isnan(coords_list), axis=(1, 2))
    start = np.where(not_filled)[0][0]
    try:
        logger.info("  running %s", name)
        model_paths = run(**kwargs)
        for i, model_path in enumerate(model_paths, start):
            _, coords_list[i] = pdb_to_c1p_coords(model_path, seq_record.id)
    except Exception as e:
        logger.error("    error (%s): %s", name, e)
        torch.cuda.empty_cache()

    end_time = time.perf_counter()
    logger.info("    %s done in %.3f s", name, end_time - start_time)


def stats(
    name: str,
    arr: np.ndarray,
    offset: int,
):
    offset_str = " " * offset
    logger.info(offset_str + "%s:", name)
    logger.info(offset_str + "  avg: %.3f s", arr.mean())
    logger.info(offset_str + "  min: %.3f s", arr.min())
    logger.info(offset_str + "  max: %.3f s", arr.max())


def rerange(
    old_range: tuple[float, float],
    new_range: tuple[float, float],
    old_value: int,
    limit_range: tuple[float, float] | None = None,
) -> int:
    frac = (old_value - old_range[0]) / (old_range[1] - old_range[0])
    new_value = new_range[0] + frac * (new_range[1] - new_range[0])
    if limit_range is not None:
        new_value = max(new_value, limit_range[0])
        new_value = min(new_value, limit_range[1])
    return round(new_value)
