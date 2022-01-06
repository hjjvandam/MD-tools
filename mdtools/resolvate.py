import uuid
import shutil
import tempfile
import subprocess
from pathlib import Path
import parmed as pmd
import MDAnalysis


def top_trim(top_file: str):
    new_topfile = tempfile.mktemp()
    old_top = open(top_file, "r")
    new_top = open(new_topfile, "w")
    for line in old_top.readlines()[:-1]:
        if "WAT" in line:
            line = line.replace("WAT", "SOL")
        new_top.write(line)
    old_top.close()
    new_top.close()

    shutil.copy2(new_topfile, top_file)


def pdb_trim(pdb_file: str):
    new_pdbfile = tempfile.mktemp()
    old_pdb = open(pdb_file, "r")
    new_pdb = open(new_pdbfile, "w")
    for line in old_pdb.readlines()[:-1]:
        line = line.replace("OW", "O ")
        line = line.replace("HW1", "H1 ")
        line = line.replace("HW2", "H2 ")
        new_pdb.write(line)
    old_pdb.close()
    new_pdb.close()

    shutil.copy2(new_pdbfile, pdb_file)


def amber_to_gmx(
    amber_pdb_file: str,
    amber_top_file: str,
    gmx_pdb_file: str,
    gmx_top_file: str,
):
    # conver amber input to gmx
    amber_top = pmd.load_file(amber_top_file, xyz=amber_pdb_file)
    amber_top.save(gmx_pdb_file, overwrite=True)
    amber_top.save(gmx_top_file, overwrite=True)


def gmx_to_amber(
    gmx_pdb_file: str,
    gmx_top_file: str,
    amber_pdb_file: str,
    amber_top_file: str,
):
    # conver gmx input to amber
    gmx_top = pmd.load_file(gmx_top_file, xyz=gmx_pdb_file)
    shutil.copy2(gmx_pdb_file, amber_pdb_file)
    gmx_top.save(amber_top_file, format="amber")


def strip_water(pdb_file: str, top_file: str):
    # remove H from both pdb and top
    mda_u = MDAnalysis.Universe(pdb_file)
    no_sol = mda_u.select_atoms("not resname WAT")
    no_sol.write(pdb_file)


def define_pbc_box(pdb_file: str):
    command = f"editconf -f {pdb_file} -o {pdb_file} -c -box 11 11 11"
    process = subprocess.Popen(command, shell=True)
    process.wait()


def add_water(input_pdb_file: str, output_pdb_file: str, top_file: str):
    command = f"genbox -cp {input_pdb_file} -cs -p {top_file} -o {output_pdb_file}"
    process = subprocess.Popen(command, shell=True)
    process.wait()
    pdb_trim(output_pdb_file)


def verify(pdb_file: str, top_file: str, mdp_file: str):
    with tempfile.NamedTemporaryFile(suffix=".tpr") as tmp_tpr:
        command = f"grompp -f {mdp_file} -c {pdb_file} -p {top_file} -o {tmp_tpr.name}"
        process = subprocess.Popen(command, shell=True)
        process.wait()


def validate_system(pdb_file: str, top_file: str):
    try:
        pmd.load_file(top_file, xyz=pdb_file)
    except ValueError:
        print(f"Error: \n\t{pdb_file}\n\t{top_file}\n")


def resolvate(
    input_path: Path, output_path: Path, mdp_file: str, is_strip_water: bool = True
):
    # Assumes input data structure
    # input_path is a directory, with several sub directories
    # each containing a seperate PDB and TOP files

    # Cleans itself up
    workdir = Path(".").resolve().joinpath(f"{uuid.uuid4()}")
    workdir.mkdir()

    for system_dir in input_path.glob("*"):
        if not system_dir.is_dir():
            continue

        try:
            old_pdb_file = next(system_dir.glob("*.pdb"))
            old_top_file = next(system_dir.glob("*.prmtop"))
        except StopIteration:
            continue

        # Replicate output directory structure
        system_output_path = output_path.joinpath(system_dir.name)
        system_output_path.mkdir()
        new_pdb_file = system_output_path.joinpath(old_pdb_file.name).as_posix()
        new_top_file = system_output_path.joinpath(old_top_file.name).as_posix()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Make temp files
            tmp_pdb_file = Path(tmpdir).joinpath(old_pdb_file.name).as_posix()
            tmp_sol_pdb_file = (
                Path(tmpdir).joinpath(f"sol_{old_pdb_file.name}").as_posix()
            )
            tmp_top_file = (
                workdir.joinpath(old_top_file.name).with_suffix(".top").as_posix()
            )

            amber_to_gmx(
                old_pdb_file.as_posix(),
                old_top_file.as_posix(),
                tmp_pdb_file,
                tmp_top_file,
            )

            if is_strip_water:
                strip_water(tmp_pdb_file, tmp_top_file)

            top_trim(tmp_top_file)

            print("Step 0: Defining PBC box...")
            define_pbc_box(tmp_pdb_file)

            print("Step 1: Adding water...")
            add_water(tmp_pdb_file, tmp_sol_pdb_file, tmp_top_file)

            print("Step 3: Verifying...")
            verify(tmp_sol_pdb_file, tmp_top_file, mdp_file)

            gmx_to_amber(tmp_sol_pdb_file, tmp_top_file, new_pdb_file, new_top_file)

            # Clean up topology files.
            # Note: for some reason, gromax fails to handle topologies in the tmp directory.
            for file_ in workdir.glob("*"):
                file_.unlink()

    workdir.rmdir()


if __name__ == "__main__":

    input_path = Path("/homes/abrace/tmp/outlier_pdbs_tops/")
    output_path = Path("/homes/abrace/tmp/outlier_pdbs_tops_done_2/")
    mdp_file = "/homes/abrace/src/resolvate/ions.mdp"
    is_strip_water = False

    resolvate(input_path, output_path, mdp_file, is_strip_water)
