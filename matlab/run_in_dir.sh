#!/bin/bash
# pass a full matlab temp directory path as a first argument

echo "Current hostname: $(hostname)"

# Remember, when running on LSF, the hostname isn't "eulerXX", it's "eXXXX".
# "eulerXX" is reserved for the job management nodes.
if [[ $(hostname) =~ e[0-9]+ ]]; then
  echo 'Running on Euler. Loading matlab module.'
  module load matlab/8.2
else
  echo 'Running Matlab locally. No module management needed.'
fi

cd $1
# Note: is the 'quit' necessary on Euler?
matlab -nodisplay -nojvm -singleCompThread -r 'rungp, quit'
#matlab -nodisplay -nojvm -singleCompThread -r 'rungp'

mlRes="$?"
if [[ "$mlRes" -ne 0 ]]; then
    echo "Problem running Matlab! Exit code [$mlRes]."
    exit "$mlRes"
fi


if [[ $(hostname) =~ e[0-9]+ ]]; then
  echo 'Unloading matlab module.'
  module unload matlab
fi

