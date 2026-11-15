import json

import pytest
from click.testing import CliRunner

torch = pytest.importorskip('torch')

from cli.overfit_roomgen import main
from tests.geometry_fixtures import two_rooms


def test_overfit_preserves_nonempty_directory_without_checkpoint(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text('{"scope": "previous diagnostic"}\n')
    preserved = report.read_bytes()
    result = CliRunner().invoke(main, ['--out', str(tmp_path), '--steps', '1', '--samples', '1'])
    assert result.exit_code != 0 and 'must be empty' in result.output
    assert list(tmp_path.iterdir()) == [report]
    assert report.read_bytes() == preserved


def test_fragment_overfit_censors_endings_unless_artificial_terminal_is_requested(tmp_path):
    from models.roomgen.model import RoomGraphRNN
    from models.training import rng_state, restore_rng
    state = rng_state()
    try:
        path = tmp_path / 'fragment.json'
        path.write_text(json.dumps(two_rooms().to_dict()))
        torch.manual_seed(7)
        initial = RoomGraphRNN(hidden=8).h_stop.bias.detach().clone()
        common = ['--source-json', str(path), '--steps', '1', '--hidden', '8', '--samples', '1', '--seed', '7']
        censored = tmp_path / 'censored'
        result = CliRunner().invoke(main, common + ['--out', str(censored)])
        assert result.exit_code == 0, (result.output, result.exception)
        report = json.loads((censored / 'report.json').read_text())
        checkpoint = torch.load(censored / 'roomgen_000001.pt', weights_only=False)
        assert not report['source_ending']['complete'] and not report['terminal_target']
        assert report['exact_binned_graph_matches'] is None
        assert torch.equal(checkpoint['model']['h_stop.bias'], initial)
        forced = tmp_path / 'forced'
        result = CliRunner().invoke(main, common + ['--out', str(forced), '--force-terminal'])
        assert result.exit_code == 0, (result.output, result.exception)
        report = json.loads((forced / 'report.json').read_text())
        checkpoint = torch.load(forced / 'roomgen_000001.pt', weights_only=False)
        assert report['terminal_target'] and report['forced_terminal']
        assert not report['source_ending']['complete']
        assert isinstance(report['exact_binned_graph_matches'], int)
        assert not torch.equal(checkpoint['model']['h_stop.bias'], initial)
        assert (forced / 'overfit_roomgen.py').exists()
    finally:
        restore_rng(state)
