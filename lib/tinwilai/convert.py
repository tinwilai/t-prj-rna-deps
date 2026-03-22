from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import MultipleSeqAlignment
from Bio.Blast.NCBIXML import Blast
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import FastaWriter
from Bio.SeqRecord import SeqRecord


def structure_to_c1p_coords(structure: Structure) -> tuple[str, np.ndarray]:
    chain = next(structure.get_chains())
    residue_list = list(filter(lambda r: r.resname != "HOH", chain.get_residues()))
    sequence = ""
    coords = []
    for residue in residue_list:
        c1p_atom = next(filter(lambda a: a.id == "C1'", residue.get_atoms()))
        sequence += residue.resname
        coords.append(c1p_atom.coord)
    return sequence, np.array(coords)


def pdb_to_c1p_coords(pdb_path: str, target_id: str) -> tuple[str, np.ndarray]:
    pdb_parser = PDBParser()
    predicted_structure = pdb_parser.get_structure(
        target_id.upper(),
        pdb_path,
    )
    sequence, coords = structure_to_c1p_coords(predicted_structure)
    return sequence, coords


def label_to_c1p_coords(sequences_csv_path: str, labels_csv_path: str, target_id: str) -> tuple[str, np.ndarray]:
    sequences_df = pd.read_csv(sequences_csv_path)
    labels_df = pd.read_csv(labels_csv_path)
    sequence = sequences_df[sequences_df["target_id"] == target_id.upper()]["sequence"].iloc[0]
    coords_df = labels_df[labels_df["ID"].str.contains(target_id.upper())].sort_values("resid")
    return sequence, coords_df[["x_1", "y_1", "z_1"]].to_numpy()


def coords_to_result_df(target_id: str, sequence: str, coords_list: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame()
    df["ID"] = [f"{target_id.upper()}_{resid}" for resid in range(len(sequence))]
    df["resname"] = list(sequence)
    df["resid"] = [resid for resid in range(len(sequence))]
    for i, coords in enumerate(coords_list):
        x, y, z = coords.T
        df[f"x_{i}"] = x
        df[f"y_{i}"] = y
        df[f"z_{i}"] = z
    return df


def structure_to_sequence(structure: Structure) -> str:
    seq, _ = structure_to_c1p_coords(structure)
    return "".join(seq)


def structures_to_seq_records(structures: list[Structure]) -> list[SeqRecord]:
    records = []
    for structure in structures:
        sequence, _ = structure_to_c1p_coords(structure)
        record = SeqRecord(Seq(sequence), id=structure.id)
        records.append(record)
    return records


def seq_records_to_fasta(seq_records: list[SeqRecord], out_path: Path) -> None:
    with open(out_path, "w") as handle:
        writer = FastaWriter(handle, wrap=0, record2title=lambda r: r.id)
        writer.write_records(seq_records)


def blast_record_to_generic(blast_record: Blast) -> MultipleSeqAlignment:
    hsps = [alignment.hsps[0] for alignment in blast_record.alignments]
    seq_records = [
        SeqRecord(
            Seq(
                ("-" * (hsp.query_start - 1) + hsp.query[: hsp.align_length] + "-" * (blast_record.query_length - hsp.query_end))[
                    : blast_record.query_length
                ]
            ),
            id=f"{alignment.hit_id}({hsp.sbjct_start}-{hsp.sbjct_end}:{alignment.length})",
        )
        for hsp, alignment in zip(hsps, blast_record.alignments)
    ]
    return MultipleSeqAlignment(seq_records)
