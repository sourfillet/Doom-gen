"""The room-graph representation (plan/graph-native.md): a Doom map as a set
of rooms plus typed connections, the abstraction the graph-native generator
works in. A Room can carry concrete geometry (its source sector polygons) for
faithful round-tripping, and/or the abstract attributes a generator predicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
            "avg_sectors_per_room": (
                sum(len(r.sectors) for r in self.rooms) / max(len(self.rooms), 1)),
        }
