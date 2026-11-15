"""Smoke-check an installed wheel outside the checkout, without torch.

    uv run --isolated --no-project --with . python -I tools/check_install.py
"""
import importlib
import importlib.abc
import importlib.metadata
from pathlib import Path
import sys


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch deliberately unavailable during installed-package smoke check')


def main():
    sys.meta_path.insert(0, NoTorch())
    from click.testing import CliRunner
    entries = [entry for entry in importlib.metadata.distribution('doom-gen').entry_points
               if entry.group == 'console_scripts']
    assert len(entries) == 12, 'expected all 12 installed console commands'
    for entry in entries:
        result = CliRunner().invoke(entry.load(), ['--help'])
        assert result.exit_code == 0, (entry.name, result.output, result.exception)
    checkout = Path(__file__).resolve().parents[1]
    for name in ('cli', 'roomgraph', 'generation.pipeline', 'wad_parser.gl_nodes',
                 'wad_parser.progression', 'wad_parser.portal_contacts', 'data.source_endings',
                 'data.spatial_portals', 'data.spatial_targets', 'data.spatial_geometry', 'data.spatial_prediction',
                 'roomgraph.spatial', 'roomgraph.assemble_spatial',
                 'generation.objectives', 'generation.asset_extractor'):
        module = importlib.import_module(name)
        mod_path = Path(module.__file__).resolve()
        assert checkout not in mod_path.parents or 'site-packages' in mod_path.parts, f'{name} imported from checkout'
    assert 'torch' not in sys.modules
    from cli.overfit_spatial import main as overfit_spatial
    result = CliRunner().invoke(overfit_spatial, ['--help'])
    assert result.exit_code == 0, (result.output, result.exception)
    assert 'torch' not in sys.modules
    print('All 12 installed commands and generation/parser imports pass without torch.')


if __name__ == '__main__':
    main()
