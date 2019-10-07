#! /bin/bash
#
### use 8 CPUs
#SBATCH -c 8
#
### run for (max) 8 hours
#SBATCH --time=8:00:00
#
### use (max) 3500 MB of memory per CPU
#SBATCH --mem-per-cpu=3500m
#
### write both output and errors to file `run_python.NNN.log`
#SBATCH -o run_python.%j.log
#SBATCH -e run_python.%j.log
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
