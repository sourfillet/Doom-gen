"""Versioned evidence for complete source examples used to learn stop labels."""
from __future__ import annotations

SOURCE_ENDINGS_VERSION = 'physical_exit_v3'
SUPPORTED_ENDINGS_VERSIONS = frozenset({'reachable_exit_v1', 'physical_exit_v2', SOURCE_ENDINGS_VERSION})


def assess_source_ending(doc, graph, *, version=SOURCE_ENDINGS_VERSION):
    """Keep historical checkpoint reference sets reproducible.

    Version 1 used exit-room roles from optimistic component traversal. Version
    2 additionally needs a physical approach and a location-aware potential
    door/key history. Version 3 consumes supported vanilla one-shot teleports
    and explicitly censors unsupported event histories. None certifies combat,
    timed puzzles or all source-port mechanics.
    """
    if version not in SUPPORTED_ENDINGS_VERSIONS:
        raise ValueError(f'unsupported source-ending assessment version: {version}')
    def result(complete, reason, history=None):
        return {'complete': complete, 'reason': reason, 'version': version, 'history': history}
    if not any(room.role == 'exit' or 'exit' in room.roles for room in graph.rooms):
        return result(False, 'no supported reachable exit role')
    if version == 'reachable_exit_v1':
        return result(True, 'historical optimistic exit-room role')
    from generation.progression import EXIT_SPECIALS
    from wad_parser.navigation import PlayerNavigation
    from wad_parser.progression import door_targets, search_exit_history, requires_action_history
    from wad_parser.types import HEXEN_UDMF_NAMESPACES
    if doc.meta.format == 'hexen' or str(doc.extra.get('udmf_namespace', '')).lower() in HEXEN_UDMF_NAMESPACES:
        return result(False, 'Hexen/script source endings are unsupported')
    if sum(thing.type == 1 for thing in doc.things) != 1:
        return result(False, 'expected exactly one player-one start')
    navigation = PlayerNavigation(doc)
    if not any(navigation.reaches_line(line) for line in doc.linedefs if line.special in EXIT_SPECIALS):
        return result(False, 'no physical player approach to a supported exit')
    if (door_targets(doc)[0] if version == 'physical_exit_v2' else requires_action_history(doc)):
        history = search_exit_history(doc, consume_one_shot_teleports=version != 'physical_exit_v2')
        return result(history['status'] == 'success', history['reason'], history)
    return result(True, 'physical player approach to a supported exit')
