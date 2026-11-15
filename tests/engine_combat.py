"""Ordinary-input combat/pickup controller for the introductory demo.

Uses engine actor positions and authored geometry to choose goals; never grants
weapons, ammo, invulnerability, kills or movement through console commands.
Controller success is fixture evidence, not a human balance assessment.
"""
from collections import deque
import math

from tests.engine_navigation import _EpisodeClock, _inside_path, navigate_exit


class CombatController:
    def __init__(self, game, doc, *, max_tics=60000):
        import vizdoom as v
        from wad_parser.components import ComponentIndex
        from wad_parser.geometry import sector_geometry
        from wad_parser.traversal import potential_state, movement_neighbors
        from wad_parser.clearance import MapClearance
        self.v, self.game, self.doc, self.max_tics = v, _EpisodeClock(game), doc, max_tics
        self.positions = {vertex.id: (vertex.x, vertex.y) for vertex in doc.vertices}
        self.locator = ComponentIndex(self.positions, sector_geometry(doc))
        records = []
        state = potential_state(doc)
        self.clearance = MapClearance(doc.vertices, state['sectors'], doc.linedefs, doc.sidedefs)
        movement_neighbors(doc.vertices, state['sectors'], doc.linedefs, doc.sidedefs,
                           [e for e in doc.derived.sector_graph['edges'] if e.get('kind') == 'teleport'],
                           allow_doors=False, edge_records=records)
        self.adjacency = {}
        for source, destination, line, teleport in records:
            self.adjacency.setdefault(source, []).append((destination, line, teleport))
        self.lines = {line.id: line for line in doc.linedefs}
        self.minimum_health = 100
        self.attack_tics = 0
        self.pickups = []
        self.allow_exit_use = False
        self.capture = None
        self.captured_combat = False
        self.combat_enabled = False
        self.locked_lines = set()
        self.key_acquired = False
        self.key_gate_sector = None
        self.crossed_key_gate = False
        self.captured_key_gate = False
        self.key_gate_opening = None

    def enemies(self):
        state = self.game.get_state()
        return [obj for obj in state.objects if obj.category.lower() == 'monster'] if state else []

    def variable(self, name):
        return float(self.game.get_game_variable(getattr(self.v.GameVariable, name)))

    def walk(self, game, target, *, tolerance=20, max_tics=2000, arrival=None, stop_on_teleport=False):
        previous, stuck = None, 0
        for tick in range(max_tics):
            if (game.is_episode_finished() or game.is_player_dead()
                    or self.game.get_episode_time() >= self.max_tics):
                return False
            state = game.get_state()
            x, y, z, angle = state.game_variables[:4]
            if self.key_acquired and self.locator.owner(x, y) == self.key_gate_sector:
                self.crossed_key_gate = True
                if state.sectors is not None and len(state.sectors) > self.key_gate_sector:
                    sector = state.sectors[self.key_gate_sector]
                    self.key_gate_opening = float(sector.ceiling_height - sector.floor_height)
            if (self.key_acquired and self.capture and not self.captured_key_gate
                    and any(self._line_distance(self.lines[line], x, y) < 96 for line in self.locked_lines)):
                self.capture('gate', state)
                self.captured_key_gate = True
            self.minimum_health = min(self.minimum_health, self.variable('HEALTH'))
            if arrival is not None and math.dist((x, y), arrival) <= tolerance:
                return True
            if stop_on_teleport and previous is not None and math.dist((x, y), previous) > 128:
                return False
            visible = [label for label in state.labels if label.object_category.lower() == 'monster']
            enemy = min(visible, key=lambda label: math.dist((x, y),
                        (label.object_position_x, label.object_position_y)), default=None)
            fighting = self.combat_enabled and enemy is not None and math.dist((x, y),
                (enemy.object_position_x, enemy.object_position_y)) < 640
            if not fighting and arrival is None and math.dist((x, y), target) <= tolerance:
                return True
            aim = (enemy.object_position_x, enemy.object_position_y) if fighting else target
            desired = math.degrees(math.atan2(aim[1] - y, aim[0] - x))
            error = (desired - angle + 180) % 360 - 180
            if fighting:
                if self.capture and not self.captured_combat:
                    self.capture('combat', state)
                    self.captured_combat = True
                forward = int(math.dist((x, y), aim) > 224 and abs(error) < 12)
                attack = int(abs(error) < 6)
                # Short lateral motion reduces exposure to imp projectiles.
                right, left = int(tick % 160 < 20), int(80 <= tick % 160 < 100)
            else:
                forward, attack, right, left = int(abs(error) < 15), 0, 0, 0
            shotgun = self.variable('WEAPON3') > 0 and self.variable('AMMO3') > 0
            near_door = any(line.special in {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}
                and self._line_distance(line, x, y) < 80 for line in self.doc.linedefs)
            game.make_action([forward, max(-10, min(10, -error)),
                              int(tick % 128 == 0 and (self.allow_exit_use or near_door)),
                              attack, int(shotgun), int(not shotgun), right, left], 1)
            self.attack_tics += attack
            if previous is not None and math.dist((x, y), previous) < .1 and forward and not fighting:
                stuck += 1
                if stuck > 160:
                    return False
            else:
                stuck = 0
            previous = x, y
        return False

    def _line_distance(self, line, x, y):
        a, b = self.positions[line.v1], self.positions[line.v2]
        dx, dy = b[0] - a[0], b[1] - a[1]
        fraction = max(0, min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy)))
        return math.dist((x, y), (a[0] + fraction * dx, a[1] + fraction * dy))

    def goto(self, target, *, tolerance=20, max_replans=32):
        for _ in range(max_replans):
            if self.game.is_episode_finished() or self.game.is_player_dead():
                return False
            current_xy = tuple(self.game.get_state().game_variables[:2])
            current, goal = self.locator.owner(*current_xy), self.locator.owner(*target)
            if current is None or goal is None:
                return False
            previous, queue = {current: None}, deque([current])
            while queue and goal not in previous:
                sid = queue.popleft()
                for edge in self.adjacency.get(sid, []):
                    if edge[1] in self.locked_lines and not self.key_acquired:
                        continue
                    if edge[0] not in previous:
                        previous[edge[0]] = sid, edge
                        queue.append(edge[0])
            if goal not in previous:
                return False
            path, sid = [], goal
            while previous[sid] is not None:
                source, edge = previous[sid]
                path.append((source, edge))
                sid = source
            interrupted = False
            for source, (_, line_id, teleport) in reversed(path):
                line = self.lines[line_id]
                a, b = self.positions[line.v1], self.positions[line.v2]
                length = math.dist(a, b)
                sign = 1 if self.doc.sidedefs[line.right_sidedef].sector == source else -1
                dx, dy = -(b[1] - a[1]) / length * sign, (b[0] - a[0]) / length * sign
                middle = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                approach = middle[0] - 20 * dx, middle[1] - 20 * dy
                crossing = middle[0] + 20 * dx, middle[1] + 20 * dy
                start = tuple(self.game.get_state().game_variables[:2])
                for waypoint in _inside_path(self.doc, source, start, approach):
                    if not self.walk(self.game, waypoint, tolerance=4, stop_on_teleport=True):
                        interrupted = True
                        break
                if interrupted:
                    break
                arrival = tuple(teleport['destination_point']) if teleport else None
                if not self.walk(self.game, crossing, arrival=arrival, tolerance=4, stop_on_teleport=True):
                    interrupted = True
                    break
            if interrupted:
                continue
            start = tuple(self.game.get_state().game_variables[:2])
            for waypoint in _inside_path(self.doc, goal, start, target):
                if not self.walk(self.game, waypoint, tolerance=tolerance, stop_on_teleport=True):
                    interrupted = True
                    break
            if not interrupted:
                return True
        return False

    def complete(self, plan):
        progression = plan.get('progression', {})
        if progression.get('profile') == 'blue-key':
            self.locked_lines = set(progression['door_lines'])
            self.key_gate_sector = progression['door_sector']
        for record in plan['things']:
            if not record['role'].startswith('opening_'):
                continue
            thing = self.doc.things[record['thing']]
            if not self.goto((thing.x, thing.y), tolerance=4):
                return {'completed': False, 'reason': f"controller could not reach {record['role']}"}
            self.pickups.append({'role': record['role'], 'type': thing.type})
        opening = {'weapon_owned': self.variable('WEAPON3'), 'shells': self.variable('AMMO3'),
                   'armor': self.variable('ARMOR')}
        if not opening['weapon_owned'] or opening['shells'] < plan['shells_supplied_before_combat']:
            return {'completed': False, 'reason': 'opening equipment not acquired', 'opening': opening}
        opening['attack_tics'] = self.attack_tics
        self.combat_enabled = True
        key_result = None
        if self.locked_lines:
            key = self.doc.things[progression['key_thing']]
            def key_present():
                return any(obj.category.lower() == 'key' and obj.name == 'BlueCard'
                           and math.dist((obj.position_x, obj.position_y), (key.x, key.y)) < 4
                           for obj in self.game.get_state().objects)
            if not key_present():
                return {'completed': False, 'reason': 'planned blue key did not spawn', 'opening': opening}
            if not self.goto((key.x, key.y), tolerance=4) or key_present():
                return {'completed': False, 'reason': 'controller did not collect the blue key', 'opening': opening}
            self.key_acquired = True
            key_result = {'key_seen_before': True, 'key_collected': True, 'door_sector': self.key_gate_sector}
            if self.capture:
                self.capture('key', self.game.get_state())
        loops = 0
        while self.enemies() and loops < 96 and self.game.get_episode_time() < self.max_tics:
            loops += 1
            position = tuple(self.game.get_state().game_variables[:2])
            if self.variable('HEALTH') < 65:
                health = [obj for obj in self.game.get_state().objects if obj.category.lower() == 'health']
                if health:
                    supply = min(health, key=lambda obj: math.dist(position, (obj.position_x, obj.position_y)))
                    self.goto((supply.position_x, supply.position_y), tolerance=4)
            enemies = self.enemies()
            if not enemies:
                break
            enemy = min(enemies, key=lambda obj: math.dist(position, (obj.position_x, obj.position_y)))
            self.goto((enemy.position_x, enemy.position_y), tolerance=48)
            if self.game.is_player_dead():
                return {'completed': False, 'reason': 'player died during combat', 'opening': opening,
                        'kills': self.variable('KILLCOUNT')}
        if self.enemies():
            return {'completed': False, 'reason': 'controller exhausted combat budget', 'opening': opening,
                    'remaining': len(self.enemies()), 'kills': self.variable('KILLCOUNT')}
        kills = self.variable('KILLCOUNT')
        if kills < plan['monsters']:
            return {'completed': False, 'reason': 'episode ended before planned combat completed',
                    'kills': kills, 'opening': opening}
        interior_walks = []
        for index, terrace in enumerate(plan.get('interiors', {}).get('terraces', [])):
            cx, cy = terrace['center']
            distance = terrace['steps'][0]['width'] / 2 + 80
            viewpoints = [(cx + dx * distance, cy + dy * distance)
                          for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]]
            viewpoint = next((point for point in viewpoints
                              if self.locator.owner(*point) == terrace['base_sector']
                              and self.clearance.problem(terrace['base_sector'], *point, radius=20) is None
                              and self.goto(point, tolerance=4, max_replans=4)), None)
            if viewpoint is None:
                return {'completed': False, 'reason': 'controller could not approach interior terrace',
                        'interior_walks': interior_walks, 'opening': opening, 'kills': kills}
            for _ in range(40):
                state = self.game.get_state()
                x, y, _, angle = state.game_variables[:4]
                desired = math.degrees(math.atan2(cy - y, cx - x))
                error = (desired - angle + 180) % 360 - 180
                if abs(error) < 2:
                    break
                self.game.make_action([0, max(-10, min(10, -error)), 0, 0, 0, 0, 0, 0], 1)
            if self.capture:
                self.capture(f'terrace-{index + 1}', self.game.get_state())
            expected = terrace['steps'][-1]['floor_height']
            if not self.goto((cx, cy), tolerance=4):
                return {'completed': False, 'reason': 'controller could not ascend interior terrace',
                        'interior_walks': interior_walks, 'opening': opening, 'kills': kills}
            observed = float(self.game.get_state().game_variables[2])
            if not self.goto(viewpoint, tolerance=4):
                return {'completed': False, 'reason': 'controller could not descend interior terrace',
                        'interior_walks': interior_walks, 'opening': opening, 'kills': kills}
            returned = float(self.game.get_state().game_variables[2])
            walk = {'base_sector': terrace['base_sector'], 'top_sector': terrace['steps'][-1]['sector'],
                    'expected_floor': expected, 'observed_floor': observed,
                    'expected_return_floor': terrace['base_floor_height'], 'returned_floor': returned}
            interior_walks.append(walk)
            if abs(observed - expected) > .01 or abs(returned - terrace['base_floor_height']) > .01:
                return {'completed': False, 'reason': 'interior terrace floor differs in engine',
                        'interior_walks': interior_walks, 'opening': opening, 'kills': kills}
        self.allow_exit_use = True
        exit_result = navigate_exit(self.game, self.doc, self.walk, max_tics=self.max_tics)
        if key_result is not None:
            key_result['gate_crossed'] = self.crossed_key_gate
            key_result['gate_opening_observed'] = self.key_gate_opening
            if not self.crossed_key_gate:
                exit_result.update(completed=False, reason='required blue gate crossing was not observed')
        return {**exit_result, 'opening': opening, 'kills': kills, 'planned_monsters': plan['monsters'],
                'minimum_health': self.minimum_health, 'attack_tics': self.attack_tics,
                'interior_walks': interior_walks, 'progression': key_result}


def open_game(path, map_name='MAP01', *, iwad=None, seed=7, config_path=None, sectors_info=False):
    import vizdoom as v
    from pathlib import Path
    game = v.DoomGame()
    game.set_doom_game_path(str(iwad or Path(v.__file__).parent / 'freedoom2.wad'))
    game.set_doom_config_path(str(config_path or Path(path).with_suffix('.combat.ini')))
    game.set_doom_scenario_path(str(path))
    game.set_doom_map(map_name)
    game.set_doom_skill(3)
    game.set_seed(seed)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_episode_timeout(90000)
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(sectors_info)
    game.set_screen_format(v.ScreenFormat.RGB24)
    game.set_screen_resolution(v.ScreenResolution.RES_320X240)
    game.set_available_buttons([v.Button.MOVE_FORWARD, v.Button.TURN_LEFT_RIGHT_DELTA, v.Button.USE,
        v.Button.ATTACK, v.Button.SELECT_WEAPON3, v.Button.SELECT_WEAPON2, v.Button.MOVE_RIGHT, v.Button.MOVE_LEFT])
    game.set_available_game_variables([v.GameVariable.POSITION_X, v.GameVariable.POSITION_Y,
        v.GameVariable.POSITION_Z, v.GameVariable.ANGLE])
    game.init()
    return game
