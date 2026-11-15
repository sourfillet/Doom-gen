"""Build explicit source-family train/validation/test corpus manifests."""
import json
from pathlib import Path

import click

from data.corpus import build_manifest


@click.command()
@click.option("--maps-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out", required=True, type=click.Path(dir_okay=False))
@click.option("--seed", default=0, type=int)
@click.option("--validation-fraction", default=0.1, type=click.FloatRange(0, 1, max_open=True))
@click.option("--test-fraction", default=0.1, type=click.FloatRange(0, 1, max_open=True))
@click.option("--families", default=None, type=click.Path(exists=True, dir_okay=False),
              help="JSON mapping WAD hashes to a common family ID for related releases.")
@click.option("--deduplicate/--keep-duplicates", default=True)
def main(maps_dir, out, seed, validation_fraction, test_fraction, families, deduplicate):
    try:
        manifest = build_manifest(maps_dir, seed=seed, validation_fraction=validation_fraction,
                                  test_fraction=test_fraction,
                                  families=json.loads(Path(families).read_text()) if families else None,
                                  deduplicate=deduplicate)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    counts = {split: sum(e['split'] == split for e in manifest['entries'])
              for split in ('train', 'validation', 'test')}
    click.echo(f"wrote {path}: {counts}; duplicate maps omitted: {len(manifest['duplicates'])}")
    if not counts['validation']:
        click.echo("No independent validation families; this manifest supports overfit/smoke training only.")


if __name__ == "__main__":
    main()
