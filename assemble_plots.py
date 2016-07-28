#!/usr/bin/env python
"""Quick and dirty tool for unifying data from multiple dilled experiments."""
import logging
from subprocess import call
from os import path

import click
import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

experiment_root = path.join('experiments', 'euler')
logging.basicConfig(level=logging.DEBUG)

sns.set_context("paper")
sns.set_style("whitegrid")


class Source(object):
    """Describes a source to plot from."""

    def __init__(self, name, file_name, keys):
        self.name = name
        self.file_name = file_name
        self.keys = keys


# The (dilleld) data sources to use. Set up here since using command-line args
# is probably overkill.
sources = [
    Source("Baselines",
           path.join(experiment_root, 'curves-upto-1-topic-limit--1-1469326603/result-data.pkl'),
           {'IC5-MEV', 'LV-GP', 'LV-MEV(1)', 'LV-MV'}),
    Source("New GP+IC experiment",
           path.join(experiment_root,  'curves-upto-1-topic-limit--1-1469555551/result-data.pkl'),
           {'IC5-GP'})
]

marker_map = {
    'IC5-MEV': '^',
    'IC5-GP': 'o',
    'LV-GP': 'x',
    'LV-MEV(1)': '|'
}

pretty_name_map = {
    'IC5-MEV': 'Independent Cascade Sampling + MEV',
    'IC5-GP': 'Independent Cascade Sampling + GP',
    'LV-GP': 'Random Sampling + GP',
    'LV-MEV(1)': 'Random Sampling + MEV'
}


# Boilerplate.
@click.group()
def cli():
    pass


@cli.command(help="Displays the names of the plotted configs from the"
                  " configured sources.")
def keys():
    click.echo("Experiments in available sources:")
    for source in sources:
        click.echo("Source [{0}] ({1})".format(source.name, source.file_name))
        with open(source.file_name, 'rb') as dill_file:
            data_list = dill.load(dill_file)
            for config, data in data_list:
                click.echo("\t- {0}".format(config.name))


@cli.command()
@click.option('--stats-every', default=25,
              help="How often to echo numeric values of what's being plotted."
                   "(0 - 100)")
def assemble(stats_every):
    click.echo("Assembling plots from specified configs.")

    for source in sources:
        click.echo("Source [{0}] ({1})".format(source.name, source.file_name))
        with open(source.file_name, 'rb') as dill_file:
            data_list = dill.load(dill_file)

            for config, data in data_list:
                if config.name in source.keys:
                    click.echo("Processing config [{0}]...".format(config.name))
                    # Assume X always represents votes/per doc, from 0 to 1.
                    x = np.linspace(0.0, 1.0, data.shape[0])
                    plt.plot(x, data,
                             label=pretty_name_map.get(config.name, config.name),
                             marker=marker_map.get(config.name),
                             markevery=5)

                    # Output the numeric values in case we e.g. want to put it
                    # in a table in an article/blogpost.
                    for idx, point in enumerate(data[config.name]):
                        if idx % stats_every == 0:
                            click.echo("{0}, {1}".format(idx, point))
                        # Note that the extra indirection ('config.name') is
                        # because 'data' is normally a 1-column pandas frame.
                        # click.echo("{0}: {1}".format(x, data[config.name][x]))

                    last_point = data[config.name][1.0]
                    click.echo("{0}, {1}".format(data.shape[0], last_point))

    plt.xlabel("Mean votes per document")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('/tmp/plot.png')
    plt.savefig('/tmp/plot.eps')


if __name__ == '__main__':
    cli()
