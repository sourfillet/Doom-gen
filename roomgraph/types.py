"""The room-graph representation: a Doom map as a set
of rooms plus typed connections, the abstraction the graph-native generator
works in. A Room can carry concrete geometry (its source sector polygons) for
faithful round-tripping, and/or the abstract attributes a generator predicts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Room:
    id: int
    sectors: list[int]                      # source sector ids this room merges
    floor_h: int
    ceil_h: int
    area: float
    centroid: tuple[float, float]
    bbox: tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    polygons: list[list[tuple[float, float]]] = field(default_factory=list)  # concrete geometry
    role: str = "room"                       # room | start | exit | corridor | ...
    roles: list[str] = field(default_factory=list)
    holes: list[list[tuple[float, float]]] = field(default_factory=list)  # holes of the representative shell
    components: list[tuple[int, int]] = field(default_factory=list)

    @property
    def size(self) -> tuple[float, float]:
        return (self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


@dataclass
class Connection:
    a: int                                   # room id
    b: int                                   # room id
    kind: str                                # door | open | corridor | lift | teleport
    width: float                             # total shared opening length (0 for teleport)


@dataclass
class RoomGraph:
    rooms: list[Room] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    map_name: str = "MAP01"
    excluded_sectors: list[int] = field(default_factory=list)
    collapsed_door_sectors: list[int] = field(default_factory=list)
    excluded_components: list[tuple[int, int]] = field(default_factory=list)
    collapsed_door_components: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> RoomGraph:
        """Restore an exact geometry input, without re-binning a DFS sequence."""
        rooms = []
        for raw in payload['rooms']:
            room = dict(raw)
            for key in ('centroid', 'bbox'):
                room[key] = tuple(room[key])
            for key in ('polygons', 'holes'):
                if key in room:
                    room[key] = [[tuple(point) for point in ring] for ring in room[key]]
            if 'components' in room:
                room['components'] = [tuple(pair) for pair in room['components']]
            rooms.append(Room(**room))
        fields = {**payload, 'rooms': rooms,
                  'connections': [Connection(**edge) for edge in payload['connections']]}
        for key in ('excluded_components', 'collapsed_door_components'):
            if key in fields:
                fields[key] = [tuple(pair) for pair in fields[key]]
        return cls(**fields)

    def neighbors(self) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = {r.id: set() for r in self.rooms}
        for c in self.connections:
            adj[c.a].add(c.b)
            adj[c.b].add(c.a)
        return adj

    def summary(self) -> dict:
        from collections import Counter
        kinds = Counter(c.kind for c in self.connections)
        return {
            "rooms": len(self.rooms),
            "connections": len(self.connections),
            "kinds": dict(kinds),
            "collapsed_door_sectors": len(self.collapsed_door_sectors),
            "excluded_components": len(self.excluded_components),
            "avg_sectors_per_room": (
                sum(len(r.sectors) for r in self.rooms) / max(len(self.rooms), 1)),
        }
