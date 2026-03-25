# https://github.com/YangLab-SDU/trRosettaRNA2/blob/main/trRNA2/predict.py
import os
from pathlib import Path
import tempfile
from typing import Literal

from attr import dataclass
import numpy as np
import pandas as pd
import torch
from tinwilai.utils import remkdir
from trRNA2.model_3d import Folding
from trRNA2.model_ss import SSpredictor
from trRNA2.utils import obj, parse_a3m, parse_bpseq, parse_ct, read_json, ss2mat
from tinwilai.logger import logger

device = "cuda"


def predict(
    model: Folding,
    seq,
    msa,
    ss,
    config: dict,
    num_recycles: int,
    nrows: int,
    return_mid: bool,
):
    with torch.no_grad():
        L = len(seq)
        res_id = torch.arange(L, device=device).view(1, L)

        if ss is not None:
            ss = ss.squeeze().to(device)
            if not (ss.shape[-1] == msa.shape[-1] == L):
                raise ValueError(
                    f"Length mismatch: seq length {L}, ss length {ss.shape[-1]}, msa length {msa.shape[-1]}!"
                )
            ss = ss.view(1, L, L)
        msa = msa.view(1, -1, L)
        outputs_all, outputs = model(
            seq,
            msa,
            ss,
            res_id=res_id.to(device),
            num_recycle=num_recycles,
            msa_cutoff=nrows,
            return_mid=return_mid,
            config=config,
        )

    outputs_tosave_all = {}
    for c in outputs_all:
        outputs = outputs_all[c]
        outputs_tosave = {}
        for k in outputs:
            if isinstance(outputs[k], torch.Tensor):
                outputs_tosave[k] = outputs[k].cpu().detach().numpy()
            elif k == "frames":
                outputs_tosave[k] = {
                    "R": outputs[k][0].cpu().detach().numpy(),
                    "t": outputs[k][1].cpu().detach().numpy(),
                }
            elif k == "frames_allatm":
                outputs_tosave[k] = {
                    fname: {
                        "R": tup[0].cpu().detach().numpy(),
                        "t": tup[1].cpu().detach().numpy(),
                    }
                    for fname, tup in outputs[k].items()
                }
            elif k == "geoms":
                pred_dict = outputs["geoms"]["inter_labels"]
                for kk in pred_dict:
                    for kkk in pred_dict[kk]:
                        pred_dict[kk][kkk] = pred_dict[kk][kkk].cpu().detach().numpy()
                outputs_tosave["inter_labels"] = pred_dict
        outputs_tosave_all[c] = outputs_tosave

    return outputs_tosave_all, outputs_all


@dataclass
class npz2cstArgs:
    fas: str
    dcut: float
    tmpdir: str


@dataclass
class FoldAllArgs:
    fas: str
    tmpdir: str
    cpu: int
    nmodels: int


def main(
    tmp_dir: Path,
    msa_path: Path,
    ss_path: Path | None = None,
    ss_fmt: Literal[
        "dot_bracket",
        "ct",
        "bpseq",
        "prob",
    ] = "dot_bracket",
    nrows: int = 500,
    num_recycles: int = 3,
    relax_steps: int = 200,
    return_mid: bool = False,
    gpu: int = 0,
    cpu: int = 5,
    pyrosetta: bool = False,
    fas: Path | None = None,
    nmodels: int = 5,
    dcut: float = 0.45,
) -> list[Path]:
    """
    Args:
        tmp_dir (Path): temp dir
        msa_path (Path): input MSA file
        ss_path (Path | None, optional): the custom secondary structure (SS) file; using trRNA2-SS if not provided (dafault). Defaults to None.
        ss_fmt (Literal[ &quot;dot_bracket&quot;, &quot;ct&quot;, &quot;bpseq&quot;, &quot;prob&quot;, ], optional): the format of custom SS file; dot_bracket(default)/ct/bpseq/prob. Defaults to "dot_bracket".
        nrows (int, optional): maximum number of rows in the MSA repr (default: 500). Defaults to 500.
        num_recycles (int, optional): number of recycles (default: 3). Defaults to 3.
        relax_steps (int, optional): maximum steps of relaxment (default: 200). Defaults to 200.
        return_mid (bool, optional): whether return mid-cycle predictions (default: False). Defaults to False.
        gpu (int, optional): use which gpu. Defaults to 0.
        cpu (int, optional): number of CPUs to use. Defaults to 5.
        pyrosetta (bool, optional): whether run energy minimization (i.e., PyRosetta version; default: False). Defaults to False.
        fas (str | None, optional): (pyrosetta) input FASTA file. Defaults to None.
        nmodels (int, optional): (pyrosetta) number of decoys to generate. Defaults to 5.
        dcut (float, optional): cutoff (pyrosetta) of distance restraints. Defaults to 0.45.
    """

    trrna2_dir = tmp_dir / "trrna2"
    rosetta_dir = trrna2_dir / "tmp"
    remkdir(trrna2_dir)
    if pyrosetta:
        remkdir(rosetta_dir)

    rosetta_dir_str = str(rosetta_dir)
    msa_path_str = str(msa_path)
    fas_str = str(fas)

    if pyrosetta:
        assert fas_str is not None, (
            "please specify the path to the fasta file by `--fas` if PyRosetta version is needed!"
        )
        relax_steps = 0

    logger.info("    reading input msa")
    msa = parse_a3m(msa_path_str, limit=20000)
    msa = torch.from_numpy(msa).to(device)[None]

    raw_seq = (
        open(msa_path_str).readlines()[1].strip().replace("T", "U").replace("-", "")
    )

    if ss_path is None:
        logger.info("    predicting secondary structure")

        with torch.no_grad():
            ss_lst = []
            for ss_model in ss_models:
                ss = ss_model(msa, msa_cutoff=nrows).float()
                ss_lst.append(ss)
            ss = torch.mean(torch.stack(ss_lst, dim=0), dim=0)
    else:
        logger.info("    reading input secondary structure")
        ss_path_str = str(ss_path)
        if bool(config["use_ss"]):
            if ss_fmt == "dot_bracket":
                ss = ss2mat(open(ss_path_str).read().rstrip().splitlines()[-1].strip())
            elif ss_fmt == "ct":
                ss = parse_ct(ss_path_str, length=len(msa[0]))
            elif ss_fmt == "bpseq":
                ss = parse_bpseq(ss_path_str)
            elif ss_fmt == "prob":
                ss = np.loadtxt(ss_path_str)
                ss += ss.T
            else:
                raise AssertionError
            ss = torch.from_numpy(ss).float()
        else:
            ss = None

    logger.info("    predicting tertiary structure")
    outputs_tosave_all, outputs_all = predict(
        model,
        raw_seq,
        msa,
        ss,
        config,
        num_recycles,
        nrows,
        return_mid,
    )

    output_paths = []

    unrelaxed_model = os.path.abspath(trrna2_dir / "model_1_unrelaxed.pdb")
    relaxed_model = os.path.abspath(trrna2_dir / "model_1_relaxed.pdb")
    npz = os.path.abspath(trrna2_dir / "model_1_2D.npz")

    logger.info("    saving outputs")
    npz_dict = {}
    outputs_tosave = {}
    for c in outputs_tosave_all:
        outputs = outputs_all[c]
        outputs_tosave = outputs_tosave_all[c]

        node_cords_pred = outputs["cord_tns_pred"][-1].squeeze(0).permute(1, 0, 2)
        chain_id = "A"
        save_pdb = (
            unrelaxed_model.replace(".pdb", f"_c{c}.pdb")
            if c < max(outputs_tosave_all)
            else unrelaxed_model
        )

        model.structure_module.converter.export_pdb_file(
            raw_seq,
            node_cords_pred.data.cpu().numpy(),
            path=save_pdb,
            chain_id=chain_id,
            confidence=outputs["plddt"][0].data.cpu().numpy(),
        )
        npz_dict = outputs_tosave["inter_labels"]
        npz_dict["plddt"] = outputs["plddt"][-1][0].data.cpu().numpy()
        np.savez_compressed(
            npz.replace(".npz", f"_c{c}.npz") if c < config["max_recycle"] else npz,
            **npz_dict,
        )
    # output_paths.append(unrelaxed_model)

    table = pd.DataFrame()
    for i in range(len(raw_seq)):
        table.loc[i + 1, "pLDDT"] = npz_dict["plddt"][i]
    table.index.names = ["Residue_Index"]
    table.to_csv(os.path.abspath(trrna2_dir / "plddt.csv"))

    if relax_steps is not None and relax_steps > 0:
        logger.info("    relaxing")
        from trRNA2.folding.refine import refine

        refine(unrelaxed_model, relaxed_model, relax_steps)
        output_paths.append(relaxed_model)

    if pyrosetta:
        logger.info("    running pyrosetta")
        from trRNA2.folding.utils_cst import npz2cst
        from trRNA2.folding.utils_ros import fold_all

        if rosetta_dir_str is None:
            rosetta_dir_str = tempfile.TemporaryDirectory(prefix="/dev/shm/").name
        # parse npz into rosetta-format restraint files
        npz2cst(
            npz2cstArgs(fas_str, dcut, rosetta_dir_str),
            geoms=outputs_tosave["inter_labels"],
        )

        # perform energy minimization
        pyrosetta_model_path = trrna2_dir / "model_1_pyrosetta.pdb"
        pyrosetta_model_path_str = str(pyrosetta_model_path)
        fold_all(
            FoldAllArgs(fas_str, rosetta_dir_str, cpu, nmodels),
            out_pdb=pyrosetta_model_path_str,
        )
        output_paths.append(pyrosetta_model_path)

    plddt_global = npz_dict["plddt"].mean()
    logger.debug(f"    pLDDT: {plddt_global:.3f}")

    return output_paths


def load_ss_models(params_dir: Path):
    global ss_models

    logger.info("  loading ss models")

    ss_models = []
    for nm in range(1, 4):
        ss_mname = f"model_{nm}_finetune"
        config_ss = read_json(params_dir / f"config_ss/{ss_mname}.json")

        ss_model = SSpredictor(
            dim_2d=config_ss["dim_pair"],
            layers_2d=config_ss["RNAformer"]["n_block"],
            config=config_ss,
            device=device,
        ).to(device)
        model_ckpt = torch.load(
            params_dir / f"models_ss/{ss_mname}.pth.tar",
            map_location=device,
            weights_only=True,
        )
        ss_model.load_state_dict(model_ckpt)
        ss_models.append(ss_model)


def load_model(params_dir: Path, model_name: str, cpu: int):
    global model
    global config

    torch.set_num_threads(cpu)

    logger.info("  loading main model")
    config = read_json(params_dir / f"config/{model_name}.json")
    model_ckpt = torch.load(
        params_dir / f"models/{model_name}.pth.tar",
        map_location=device,
        weights_only=True,
    )
    if "to_dist.fc_2d.distance.C4'.1.weight" in model_ckpt:
        obj["inter_labels"]["distance"] = [
            "C3'",
            "P",
            "N1",
            "C4",
            "C1'",
            "CiNj",
            "PiNj",
            "C4'",
        ]
    model = Folding(
        dim_2d=config["dim_pair"],
        layers_2d=config["RNAformer"]["n_block"],
        config=config,
    ).to(device)
    model.load_state_dict(model_ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def load_all_models(params_dir: Path, model_name: str, cpu: int):
    load_model(params_dir, model_name, cpu)
    load_ss_models(params_dir)


def run(
    tmp_dir: Path,
    input_seq_path: Path | None,
    input_msa_path: Path,
    input_dbn_path: Path | None,
    relax_steps: int,
) -> list[Path]:
    return main(
        tmp_dir,
        input_msa_path,
        input_dbn_path,
        "dot_bracket",
        cpu=os.cpu_count(),
        relax_steps=relax_steps,
        pyrosetta=input_dbn_path is not None,
        fas=input_seq_path,
    )
