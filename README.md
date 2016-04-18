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
/notebooks        Contains exploratory Jupyter notebooks.
/crowd            Contains the main project code, such as the graph generation
                  algorithms, the data classes, and the experiments.
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
 * Integrate auto-reload in notebooks to improve workflow.
