import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import DataParallel
from rhofold.config import rhofold_config
from rhofold.relax.relax import AmberRelaxation
from rhofold.utils import get_device, save_ss2ct, timing
from rhofold.utils.alphabet import get_features
from rhofold.rhofold import RhoFold

logger = logging.getLogger("RhoFold+ Inference")
logger.setLevel(level=logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

device = get_device(None)


@torch.no_grad()
def main(
    input_path: Path,
    output_dir: Path,
    model: RhoFold,
    msa_path: Path | None = None,
    relax_steps: int = 1000,
) -> None:
    # Input seq, MSA
    logger.info(f"Input FASTA {input_path}")

    if msa_path is None:
        msa_path = input_path
        logger.info("The model will use the single query sequence only. Setting the MSA path to the input fasta file.")

    else:
        if msa_path is None:
            raise ValueError("Single sequence mode is off. Please provide the input MSA file.")

        logger.info(f"Input MSA path: {msa_path}")

    with timing("RhoFold+ Inference", logger=logger):
        data_dict = get_features(input_path, msa_path)

        # Forward pass
        outputs = model(
            tokens=data_dict["tokens"].to(device),
            rna_fm_tokens=data_dict["rna_fm_tokens"].to(device),
            seq=data_dict["seq"],
        )

        output = outputs[-1]

        # Secondary structure, .ct format
        ss_prob_map = torch.sigmoid(output["ss"][0, 0]).data.cpu().numpy()
        ss_file = f"{output_dir}/ss.ct"
        save_ss2ct(ss_prob_map, data_dict["seq"], ss_file, threshold=0.5)

        # Dist prob map & Secondary structure prob map, .npz format
        npz_file = f"{output_dir}/results.npz"
        np.savez_compressed(
            npz_file,
            dist_n=torch.softmax(output["n"].squeeze(0), dim=0).data.cpu().numpy(),
            dist_p=torch.softmax(output["p"].squeeze(0), dim=0).data.cpu().numpy(),
            dist_c=torch.softmax(output["c4_"].squeeze(0), dim=0).data.cpu().numpy(),
            ss_prob_map=ss_prob_map,
            plddt=output["plddt"][0].data.cpu().numpy(),
        )

        # Save the prediction
        unrelaxed_model = f"{output_dir}/unrelaxed_model.pdb"

        # The last cords prediction
        node_cords_pred = output["cord_tns_pred"][-1].squeeze(0)
        model.structure_module.converter.export_pdb_file(
            data_dict["seq"],
            node_cords_pred.data.cpu().numpy(),
            path=unrelaxed_model,
            chain_id=None,
            confidence=output["plddt"][0].data.cpu().numpy(),
            logger=logger,
        )

    # Amber relaxation
    if device == "cpu":
        use_gpu = False
    else:
        use_gpu = True

    if relax_steps is not None:
        if relax_steps > 0:
            with timing(f"Amber Relaxation : {relax_steps} iterations", logger=logger):
                amber_relax = AmberRelaxation(max_iterations=relax_steps, logger=logger, use_gpu=use_gpu)
                relaxed_model = f"{output_dir}/relaxed_{relax_steps}_model.pdb"
                amber_relax.process(unrelaxed_model, relaxed_model)


def load_model(model_path: Path) -> RhoFold | DataParallel[RhoFold]:
    logger.info("Constructing RhoFold+")
    model = RhoFold(rhofold_config)

    logger.info(f"    loading {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["model"])
    model.eval()

    logger.info(f"    Inference using device {device}")
    model = model.to(device)

    # https://docs.pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    return model


def convert_model(model: RhoFold, device_to: str) -> RhoFold:
    global device

    device = device_to
    logger.info(f"Switching model to device {device}")
    model = model.to(device)

    return model
