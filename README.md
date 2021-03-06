# Document Similarity Graphs for Crowdsourcing


## Overview

Using the document data from the TREC 2011 conference (non-free dataset, sadly),
our goal is to build a graph from document similarities for every topic,
whereby an edge would exist between two documents whose similarity exceeds
a certain similarity threshold.

This similarity graph would then prove to be a valuable tool in tasks such as
improved vote aggregation (propagate human voters' votes to similar nearby
documents), as well as active learning (efficiently identifying most
informative unlabeled documents).

Most of this project and its code is based on Martin Davtyan's own
framework and thesis project. His code is [also available on GitHub][0].
[The theory behind the original project has been published in
CIKM '15.][2]


## Data

TODO(andrei): More information about getting the data, even though it's
behind a paywall or at least a bureaucracy-wall.


## Project Structure

```
/
/print_todos.sh         @Valloric's ag-powered TODO search tool.
/README.md              You're reading this right now!
/remote_output.py       Utility for stripping all output from a Jupyter
                        notebook.
/TODO.md                Very general project TODOs.
/crowd                  Contains the main project code, such as the graph
                        generation algorithms, the data classes, and the
                        experiments.
/matlab                 Gausian Process code for enhanced vote
                        aggregation. Matlab instead of Python for
                        various technical and historic reasons.
/notebooks              Contains exploratory Jupyter notebooks.
/remote                 Helper scripts for remote execution, e.g. on
                        the Euler compute cluster.
```

## Running the code

Most of the interesting stuff currently resides in the Jupyter notebooks in the
'notebooks' folder. For dependency management, Anaconda is highly recommended.

The 'compute_learning_curves.py' tool is slowly growing into the main
experiment driver. Remote deployment is handled using [Fabric3][1].

Note that all the pickle (`*.pkl`) files produced by this tool are
created using Dill, since it supports more things than the stock
`pickle` module, such as direct serialization of lambdas and more.


## Roadmap

 * Improve existing methods using more advanced techniques, such as adaptive
   weights or combinations of techniques (poor man's boosting).
 * Explore usability of graph structure for active learning tasks.


[0]:https://github.com/martinthenext/ir-crowd-thesis
[1]:https://pypi.python.org/pypi/Fabric3
[2]:https://dl.acm.org/citation.cfm?id=2806460
