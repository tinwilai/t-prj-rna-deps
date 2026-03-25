import os
import subprocess
from pathlib import Path

from Bio.Align import MultipleSeqAlignment
from Bio.Blast import NCBIXML
from Bio.SeqRecord import SeqRecord
from tinwilai.convert import (
    blast_record_to_generic,
    mmseqs_output_to_generic,
    seq_records_to_fasta,
)
from tinwilai.utils import remkdir


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
