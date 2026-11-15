"""Stage 3 training CLI: the tag-conditioned property GNN over the sector graph.

    train-properties --maps-dir dataset/maps --tags-dir tags --out checkpoints/

The model and graphs are small, so this trains fine on CPU (the default).
One graph per step; targets are per-sector flat/texture/special indices from
data.property_targets.
"""
from __future__ import annotations

import click

from cli.training_common import graph_training_options, run_graph_training


@click.command()
@graph_training_options
def main(**options):
    run_graph_training('properties', options)


if __name__ == '__main__':
    main()
