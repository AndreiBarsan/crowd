# TODO (in addition to official project milestones and in-code TODOs)
## Project enhancements, Spring/Summer 2017
 [ ] Re-run simple experiment and check learning curves are sane.
 [ ] Separate experiment run from plotting/analysis. Remember, you want
     to be able to re-analyze experiment dumps as many times as you want.
 [ ] Run experiment with multiple threshold values and see which results
     are best.
 [ ] Statistical significance tests to quantify differences between
     different aggregation techniques.


## Old, general stuff
 * Set up automatic PEP 8 formatting and linting, even inside notebooks.
 * Have documentation-specific sweep over all code, ensuring that especially
   non-trivial stuff is explained right.
 * Remove all absolute file locations from code (no more '~/data').
 * Set up script(s) to download the downloadable data, and format/merge it
   accordingly.
 * Ensure you have up-to-date instructions on obtaining the non-downloadable
   data.
 * Recreate all results on a fresh machine (after everything is done).
 * Add reference to Martin's paper!!! Giving credit is very important.

# July
## Monday, July 4th
### DONE
 * Extract fully modular aggregate plot gen into own utility. Use gflags.
 * Ran entire simulation locally (~3h); got decent numbers; ETH's VPN was down
   on RDS @ home for some reason...

## Tuesday, July 5th
 * Run serious job on Euler aggregating all topics.
 * Do a little bit of investigation regarding your code speed. Remember, you
   can always scrap notebook support if it makes your code considerably faster.
   Use more detailed profiler than %prun in the notebook!
 * Take train to Bucharest.
### DONE
 n/a

## Wednesday, July 6th
 * Visa appointment
 * Busy schedule, probably no time to work
 * Short test run on Euler.
### DONE
 n/a

## Thursday, July 7th
 * More experiments on Euler. Each one should hopefully take less than 2h.
 * Prepare plots for meeting with Carsten.
### DONE
 n/a

## Friday, July 8th
 * Online meeting with Carsten.
 * Train back to Brasov.

## Saturday, July 9th
 * Try to recreate Gaussian Process learning curves using Martin's old code
   on Euler, since it definitely shouldn't be as slow as it currently is for
   you. Save these numbers and use them as a baseline when experimenting with
   e.g. a dedicated python package or something.
