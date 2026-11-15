"""Common reproducible training, validation, and exact-resume checkpoints."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import platform
import ast
from pathlib import Path

import numpy as np


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        items = [item.cpu() for item in state['cuda']]
        count = torch.cuda.device_count()
        if len(items) == count:
            torch.cuda.set_rng_state_all(items)
        else:
            for dev_idx, item in enumerate(items[:count]):
                torch.cuda.set_rng_state(item, device=dev_idx)


def load_resume(path, device='cpu'):
    import torch
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('training_version') != 1 or 'optimizer' not in checkpoint or 'rng' not in checkpoint:
        raise ValueError("checkpoint has no complete optimizer/RNG state; use it for inference, not exact resume")
    return checkpoint


def tag_fingerprint(tags_dir):
    digest = hashlib.sha256()
    if tags_dir:
        root = Path(tags_dir)
        for path in sorted(root.rglob('*.json')):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def code_fingerprint(kind=None, *, root=None):
    """Hash training's local import closure, or the full code tree for audits.

    Generation/evaluation-only edits should not invalidate exact training
    resume. Old checkpoints retain their conservative full-tree hash scope.
    """
    digest = hashlib.sha256()
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    paths = set()
    scopes = {name: [f'cli.train_{name}', 'models.training']
              for name in ('layout', 'roomgen', 'properties', 'placement')}
    scopes['roomgraph_preprocessing'] = ['data.source_endings', 'data.roomgraph_targets',
        'data.corpus', 'roomgraph.segment', 'wad_parser.map_extractor', 'wad_parser.types']
    scopes['raster_preprocessing'] = ['data.dataset', 'data.augmentation', 'data.rasterizer']
    if kind in scopes:
        queue, seen = list(scopes[kind]), set()
        while queue:
            module = queue.pop()
            if module in seen:
                continue
            seen.add(module)
            path = root.joinpath(*module.split('.')).with_suffix('.py')
            if not path.is_file():
                path = root.joinpath(*module.split('.'), '__init__.py')
            if not path.is_file():
                continue  # installed dependencies are checked separately
            paths.add(path)
            parts = module.split('.')
            package = parts if path.name == '__init__.py' else parts[:-1]
            queue.extend('.'.join(parts[:index]) for index in range(1, len(parts)))
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    queue.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    prefix = package[:max(len(package) - node.level + 1, 0)] if node.level else []
                    base = '.'.join(prefix + (node.module.split('.') if node.module else []))
                    if base:
                        queue.append(base)
                        queue.extend(f'{base}.{alias.name}' for alias in node.names if alias.name != '*')
                elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                      and isinstance(node.args[0].value, str)
                      and (isinstance(node.func, ast.Name) and node.func.id == '__import__'
                           or isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module')):
                    queue.append(node.args[0].value)
    else:
        for directory in ('data', 'models', 'cli', 'wad_parser', 'wad_writer', 'generation', 'roomgraph', 'tagging'):
            paths.update((root / directory).rglob('*.py'))
    if not paths:
        raise ValueError('no source files found for code fingerprint')
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_preprocessing_fingerprint(scope='roomgraph_preprocessing_v1', *, root=None):
    """Source-cache semantics exclude unrelated model/generation/CLI edits."""
    if scope == 'repository_v1':
        return code_fingerprint(root=root)
    if scope != 'roomgraph_preprocessing_v1':
        raise ValueError(f'unsupported source preprocessing fingerprint scope: {scope}')
    return code_fingerprint('roomgraph_preprocessing', root=root)


def runtime_versions(*packages: str) -> dict[str, str]:
    versions = {'python': platform.python_version()}
    for name in packages:
        versions[name] = importlib.metadata.version(name)
    return versions


def dependency_versions(device=None):
    import shapely
    import torch
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'click', 'shapely', 'networkx')}
    versions.update(python=platform.python_version(), geos=shapely.geos_version_string)
    versions.update(machine=platform.machine(), cuda_runtime=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), cpu_threads=torch.get_num_threads(),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32)
    if device is not None:
        device = torch.device(device)
        versions['training_device'] = str(device)
        if device.type == 'cuda':
            versions['cuda_device_name'] = torch.cuda.get_device_name(device)
            versions['cuda_capability'] = list(torch.cuda.get_device_capability(device))
    return versions


def fit(model, conditioner, optimizer, train_pool, validation_pool, loss_fn, *,
        config, settings, manifest, out_dir, steps, save_every=1000,
        validate_every=1000, resume=None, tags_dir=None):
    """loss_fn(sample, training) returns a differentiable scalar or (loss, metrics).

    Validation uses a fixed RNG stream and restores training RNG afterwards.
    `steps` is the total desired optimizer step, including resumed work.
    """
    import click
    import torch

    if len(train_pool) == 0:
        raise ValueError("no usable training samples after preprocessing")
    if steps < 1 or save_every < 1 or validate_every < 1:
        raise ValueError("steps/save-every/validate-every must be positive")
    signature = tag_fingerprint(tags_dir)
    code_scope = (resume.get('code_fingerprint_scope', 'repository_v1') if resume else
                  'training_dependencies_v1' if config.get('kind') in {'layout', 'roomgen', 'properties', 'placement'}
                  else 'repository_v1')
    code_signature = code_fingerprint(config['kind'] if code_scope == 'training_dependencies_v1' else None)
    versions = dependency_versions(next(model.parameters()).device)
    first, best = 1, float('inf')
    if resume:
        if resume['config'] != config or resume['settings'] != settings:
            raise ValueError("resume architecture/training settings differ from the checkpoint")
        if resume['corpus']['fingerprint'] != manifest['fingerprint'] or resume['tag_fingerprint'] != signature:
            raise ValueError("resume corpus or tag store changed; start a new experiment")
        if resume.get('code_fingerprint') != code_signature:
            raise ValueError('training/preprocessing code changed; start a new experiment rather than claiming exact resume')
        if resume.get('versions') != versions:
            raise ValueError('training dependencies changed; start a new experiment rather than claiming exact resume')
        model.load_state_dict(resume['model'])
        conditioner.load_state_dict(resume['conditioner'])
        conditioner.tag_to_index = resume['tag_to_index']
        optimizer.load_state_dict(resume['optimizer'])
        restore_rng(resume['rng'])
        first, best = resume['step'] + 1, resume.get('best_validation_loss', float('inf'))
    if steps < first:
        raise ValueError(f"--steps must exceed checkpoint step {first - 1}")
    out = Path(out_dir)
    if not resume and ((out / 'metrics.jsonl').exists() or any(out.glob('*.pt'))):
        raise ValueError('output already contains a training run; use --resume or choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    manifest_out = out / 'corpus-manifest.json'
    manifest_tmp = manifest_out.with_suffix('.json.tmp')
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest_tmp.replace(manifest_out)
    metrics_path = out / 'metrics.jsonl'
    if resume and metrics_path.exists():
        kept = []
        for raw_line in metrics_path.read_text(encoding='utf-8').splitlines():
            if not raw_line.strip():
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if entry.get('step', 0) <= resume['step']:
                kept.append(raw_line)
        metrics_tmp = metrics_path.with_suffix('.jsonl.tmp')
        metrics_tmp.write_text(('\n'.join(kept) + '\n') if kept else '', encoding='utf-8')
        metrics_tmp.replace(metrics_path)
    def save(path, step):
        payload = {'training_version': 1, 'model': model.state_dict(),
                   'conditioner': conditioner.state_dict(), 'tag_to_index': conditioner.tag_to_index,
                   'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'config': config,
                   'settings': settings, 'corpus': manifest, 'tag_fingerprint': signature,
                   'code_fingerprint': code_signature,
                   'code_fingerprint_scope': code_scope,
                   'step': step, 'best_validation_loss': best, 'versions': versions}
        if hasattr(model, 'guidance_convention'):
            payload['guidance_convention'] = model.guidance_convention
        # Property/placement inference expects vocabularies at top level.
        if 'vocabs' in config:
            payload['vocabs'] = config['vocabs']
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
    def unpack(value):
        return value if isinstance(value, tuple) else (value, {})
    losses = []
    for step in range(first, steps + 1):
        model.train()
        conditioner.train()
        sample = train_pool[int(np.random.randint(len(train_pool)))]
        loss, metrics = unpack(loss_fn(sample, True))
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite training loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(conditioner.parameters()), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        record = {'step': step, 'train_loss': float(loss.detach()), **{k: float(v) for k, v in metrics.items()}}
        if validation_pool and (step % validate_every == 0 or step == steps):
            state = rng_state()
            try:
                seed_all(0)
                model.eval()
                conditioner.eval()
                values, by_head = [], {}
                with torch.no_grad():
                    for sample in validation_pool:
                        val_loss, val_metrics = unpack(loss_fn(sample, False))
                        values.append(float(val_loss))
                        for key, value in val_metrics.items():
                            by_head.setdefault(key, []).append(float(value))
                record['validation_loss'] = float(np.mean(values))
                record.update({f'validation_{key}': float(np.mean(value)) for key, value in by_head.items()})
            finally:
                restore_rng(state)
            if not np.isfinite(record['validation_loss']):
                raise ValueError(f"non-finite validation loss at step {step}")
            if record['validation_loss'] < best:
                best = record['validation_loss']
                save(out / 'best.pt', step)
        with (out / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if step % 50 == 0 or step == steps:
            click.echo(f"step {step}/{steps} loss {np.mean(losses[-50:]):.4f}"
                       + (f" validation {record['validation_loss']:.4f}" if 'validation_loss' in record else ''))
        if step % save_every == 0 or step == steps:
            save(out / f"{config['kind']}_{step:06d}.pt", step)
    return {'step': steps, 'train_loss': losses[-1], 'best_validation_loss': best if validation_pool else None}
