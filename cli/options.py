"""Shared generation argument parsing and user-facing failures."""
from functools import wraps
import math

import click


def parse_tags(spec):
    tags = {}
    for part in (spec or '').split(','):
        if not part.strip():
            continue
        name, separator, weight = part.strip().rpartition(':')
        if not separator:
            name, weight = part.strip(), '1'
        try:
            value = float(weight)
        except ValueError as exc:
            raise click.BadParameter('tag weights must be numbers', param_hint='--tags') from exc
        if not name.strip() or not math.isfinite(value) or value < 0:
            raise click.BadParameter('tags need a name and finite nonnegative weight', param_hint='--tags')
        tags[name.strip()] = value
    return tags


def generation_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, RuntimeError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
    return wrapped
