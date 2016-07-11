#!/bin/bash
# pass a full matlab temp directory path as a first argument

if [[ $(hostname) =~ "euler" ]]; then
  echo 'Running on Euler. Loading matlab module.'
  module load matlab/8.2
else
  echo 'Running Matlab locally.'
fi

cd $1
# Note: is the 'quit' necessary on Euler?
#matlab -nodisplay -nojvm -singleCompThread -r 'rungp, quit'
matlab -nodisplay -nojvm -singleCompThread -r 'rungp'


if [[ $(hostname) =~ "euler" ]]; then
  echo 'Running on Euler. Loading matlab module.'
  module unload matlab
fi

