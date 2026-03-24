# https://github.com/WangJiuming/rhofold_protocol/blob/main/rhofold/inference.py
from pathlib import Path

import logging
import numpy as np
import torch
from rhofold.config import rhofold_config
from rhofold.relax.relax import AmberRelaxation
from rhofold.utils import save_ss2ct
from rhofold.utils.alphabet import get_features
from rhofold.rhofold import RhoFold
from tinwilai.utils import remkdir

logger = logging.getLogger("T_prj.rna")

device = "cuda"


@torch.no_grad()
def main(
    tmp_dir: Path,
    model: RhoFold,
    input_path: Path,
    msa_path: Path | None = None,
    relax_steps: int = 1000,
) -> tuple[Path, Path]:
    rhofold_dir = tmp_dir / "rhofold"
    remkdir(rhofold_dir)

    unrelaxed_model_path = rhofold_dir / "unrelaxed_model.pdb"
    relaxed_model_path = rhofold_dir / "relaxed_model.pdb"

    # Input seq, MSA
    if msa_path is None:
        msa_path = input_path
    else:
        if msa_path is None:
            raise ValueError(
                "Single sequence mode is off. Please provide the input MSA file."
            )

    data_dict = get_features(input_path, msa_path)

    logger.info("    forward pass")

    # Forward pass
    outputs = model(
        tokens=data_dict["tokens"].to(device),
        rna_fm_tokens=data_dict["rna_fm_tokens"].to(device),
        seq=data_dict["seq"],
    )

    output = outputs[-1]

    logger.info("    saving results")

    # Secondary structure, .ct format
    ss_prob_map = torch.sigmoid(output["ss"][0, 0]).data.cpu().numpy()
    ss_file = f"{rhofold_dir}/ss.ct"
    save_ss2ct(ss_prob_map, data_dict["seq"], ss_file, threshold=0.5)

    # Dist prob map & Secondary structure prob map, .npz format
    npz_file = f"{rhofold_dir}/results.npz"
    np.savez_compressed(
        npz_file,
        dist_n=torch.softmax(output["n"].squeeze(0), dim=0).data.cpu().numpy(),
        dist_p=torch.softmax(output["p"].squeeze(0), dim=0).data.cpu().numpy(),
        dist_c=torch.softmax(output["c4_"].squeeze(0), dim=0).data.cpu().numpy(),
        ss_prob_map=ss_prob_map,
        plddt=output["plddt"][0].data.cpu().numpy(),
    )

    # The last cords prediction
    node_cords_pred = output["cord_tns_pred"][-1].squeeze(0)
    model.structure_module.converter.export_pdb_file(
        data_dict["seq"],
        node_cords_pred.data.cpu().numpy(),
        path=str(unrelaxed_model_path),
        chain_id=None,
        confidence=output["plddt"][0].data.cpu().numpy(),
    )

    logger.info("    relaxing")

    # Amber relaxation
    if device == "cpu":
        use_gpu = False
    else:
        use_gpu = True

    if relax_steps is not None:
        devnull_logger = logging.Logger("devnull")
        devnull_logger.addHandler(logging.NullHandler())
        if relax_steps > 0:
            amber_relax = AmberRelaxation(
                max_iterations=relax_steps, logger=devnull_logger, use_gpu=use_gpu
            )
            amber_relax.process(str(unrelaxed_model_path), str(relaxed_model_path))

    return unrelaxed_model_path, relaxed_model_path


def load_model(model_path: Path) -> RhoFold:
    model = RhoFold(rhofold_config)

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))["model"]
    )
    model.eval()

    model.to(device)

    return model


def convert_model(model: RhoFold, device_to: str) -> RhoFold:
    global device

    device = device_to
    model = model.to(device)

    return model
