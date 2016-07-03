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


## Data

TODO(andrei) More information about getting the data, even though it's behind
a paywall or at least a bureaucracy-wall.


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
/crowd/notebooks        Contains exploratory Jupyter notebooks.
```

## Running the code

Most of the interesting stuff currently resides in the Jupyter notebooks in the
'notebooks' folder. For dependency management, Anaconda is highly recommended.


## Roadmap

 * Recreate existing vote propagation methods using graph (NN,
   Merge-Enough-Votes, etc.).
 * Improve existing methods using more advanced techniques, such as adaptive
   weights or combinations of techniques (poor man's boosting).
 * Explore usability of graph structure for active learning tasks.
 * Integrate Gaussian Process aggregation into this project somehow,
   even if it confines to project to just running on Euler for the time
   being.

## Rough timeline

### Apr 5 - Apr 19
 * Improved graph structure visualization.
 * Simple framework for majority voting.

### Apr 19 - May 3
 * Reimplement Martin's sampling strategy for evaluation
 * Filter out docs with no votes and no ground truth (toggleable) in graph
 * Recreate NN and MV experiments using graph structure.

### May 3 - May 17
 * Improve existing techniques using dynamic weights.
 * Find threshold properties (low priority)
 * Aggregated plots for everything

### May 17 - August
 * Information diffusion techniques in graph.
