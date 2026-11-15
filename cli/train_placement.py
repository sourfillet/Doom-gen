"""Stage 4 training CLI: the tag-conditioned placement GNN.

    train-placement --maps-dir dataset/maps --tags-dir tags --out checkpoints/

Count heads use Poisson NLL; type heads use a masked soft cross-entropy over
each sector's real type distribution (sectors with no monsters/pickups don't
supervise the corresponding type head). CPU by default - small model/graphs.
"""
from __future__ import annotations

import click

from cli.training_common import graph_training_options, run_graph_training


@click.command()
@graph_training_options
def main(**options):
    run_graph_training('placement', options)


if __name__ == '__main__':
    main()
