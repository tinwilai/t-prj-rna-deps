# https://github.com/LinearFold/LinearFold/blob/master/linearfold
import subprocess
from pathlib import Path
from typing import Literal


def main(
    input_sequence: str,
    linearfold_path: Path,
    beamsize=100,
    is_sharpturn=False,
    is_verbose=False,
    is_eval=False,
    is_constraints=False,
    zuker_subopt=False,
    delta=5.0,
    shape_file_path="",
    is_fasta=False,
    dangles: Literal[0, 2] = 2,
) -> str:
    """
    Args:
        beamsize (int, optional): set beam size. Defaults to 100.
        use_vienna (bool, optional): use vienna parameters. Defaults to False.
        is_sharpturn (bool, optional): enable sharp turn in prediction. Defaults to False.
        is_verbose (bool, optional): print out energy of each loop in the structure. Defaults to False.
        is_eval (bool, optional): print out energy of a given structure. Defaults to False.
        is_constraints (bool, optional): print out energy of a given structure. Defaults to False.
        zuker_subopt (bool, optional): output Zuker suboptimal structures. Defaults to False.
        delta (float, optional): compute Zuker suboptimal structures with scores or energies in a certain range of the optimum. Defaults to 5.0.
        shape_file_path (str, optional): specify a file name that contains SHAPE reactivity data. Defaults to "".
        is_fasta (bool, optional): input is in fasta format. Defaults to False.
        dangles (Literal[0, 2], optional): the way to treat `dangling end' energies for bases adjacent to helices in free ends and multi-loops. Defaults to 2.

    Returns:
        str: predicted secondary structure in dot-bracket notation.
    """

    cmd = [
        linearfold_path,
        beamsize,
        int(is_sharpturn),
        int(is_verbose),
        int(is_eval),
        int(is_constraints),
        int(zuker_subopt),
        delta,
        shape_file_path,
        int(is_fasta),
        dangles,
    ]
    cmd = list(map(str, cmd))
    # https://stackoverflow.com/a/73574419
    proc = subprocess.run(cmd, input=input_sequence, capture_output=True, text=True)
    return proc.stdout.splitlines()[1].split(" ")[0]
