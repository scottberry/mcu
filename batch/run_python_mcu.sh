#! /bin/bash
#
### use 16 CPUs
#SBATCH -c 16
#
### run on hydra
#SBATCH --partition hydra
#
#SBATCH --time=8:00:00
#SBATCH --mem=3500m
#
### write both output and errors to file `run_mcu.NNN.log`
#SBATCH -o run_mcu.%j.log
#SBATCH -e run_mcu.%j.log
#

# activate virtualenv;
venv="$HOME/.virtualenvs/mcu/"
if [ -r "$venv/bin/activate" ]; then
  . "$venv/bin/activate"
else
    echo 1>&2 "Cannot activate virtualenv '$venv'"
    exit 69  # EX_UNAVAILABLE
fi

# for debugging purposes, it's a good idea to print out the venv name
# and the actual python interpreter used
echo 1>&2 "Running in virtualenv '$venv', using python interpreter $(command -v python) ..."

# run python script
python "$@"
