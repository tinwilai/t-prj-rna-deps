import os
import shutil
import subprocess
from pathlib import Path

from Bio.Align import MultipleSeqAlignment
from Bio.Blast import NCBIXML
from Bio.PDB.Structure import Structure
from Bio.SeqRecord import SeqRecord
from tinwilai.convert import (
    blast_record_to_generic,
    coords_to_result_df,
    label_to_c1p_coords,
    mmseqs_output_to_generic,
    seq_records_to_fasta,
    structure_to_c1p_coords,
)
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


def blastn(
    tmp_dir: Path,
    blastn_path: Path,
    blast_db: Path,
    seq_records: list[SeqRecord],
) -> list[MultipleSeqAlignment]:
    blastn_dir = tmp_dir / "blastn"
    remkdir(blastn_dir)

    in_path = blastn_dir / "blast_query.fasta"
    out_path = blastn_dir / "blast_output.xml"
    seq_records_to_fasta(seq_records, in_path)
    subprocess.run(
        [
            blastn_path,
            "-db",
            blast_db,
            "-query",
            in_path,
            "-out",
            out_path,
            "-num_threads",
            f"{os.cpu_count()}",
            "-outfmt",
            "5",
            "-task",
            "blastn-short",
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    blast_records = NCBIXML.parse(open(out_path))
    results = []
    for query_seq_record, blast_record in zip(seq_records, blast_records):
        align = blast_record_to_generic(blast_record)
        align._records.insert(0, query_seq_record)
        results.append(align)
    return results


def remkdir(dir_path: Path) -> None:
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir()


def mmseqs(
    tmp_dir: Path,
    mmseqs_path: Path,
    target_db: Path,
    seq_records: list[SeqRecord],
) -> dict[str, MultipleSeqAlignment]:
    mmseqs_dir = tmp_dir / "mmseqs"
    remkdir(mmseqs_dir)

    mtmp_dir = mmseqs_dir / "tmp"
    fasta_path = mmseqs_dir / "query.fasta"
    result_path = mmseqs_dir / "result.m8"

    seq_records_to_fasta(seq_records, fasta_path)
    subprocess.run(
        [
            mmseqs_path,
            "easy-search",
            fasta_path,
            target_db,
            result_path,
            mtmp_dir,
            "--db-load-mode",
            "2",
            "-s",
            "7.5",
            "--search-type",
            "3",
            "--format-output",
            "query,target,taln,qlen,qstart,qend,alnlen",
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    return mmseqs_output_to_generic(result_path)
