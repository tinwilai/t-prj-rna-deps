import os
import subprocess
from pathlib import Path

from Bio.Align import MultipleSeqAlignment
from Bio.Blast import NCBIXML
from Bio.PDB.Structure import Structure
from Bio.SeqRecord import SeqRecord
from tinwilai.convert import blast_record_to_generic, coords_to_result_df, label_to_c1p_coords, seq_records_to_fasta, structure_to_c1p_coords
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
    in_path = tmp_dir / "blast_query.fasta"
    out_path = tmp_dir / "blast_output.xml"
    seq_records_to_fasta(seq_records, in_path)
    subprocess.run([
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
    ])
    blast_records = NCBIXML.parse(open(out_path))
    results = []
    for query_seq_record, blast_record in zip(seq_records, blast_records):
        align = blast_record_to_generic(blast_record)
        align._records.insert(0, query_seq_record)
        results.append(align)
    return results
