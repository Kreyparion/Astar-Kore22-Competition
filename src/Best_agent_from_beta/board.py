import itertools
import numpy as np
from typing import Dict, List, Union, Optional, Generator
from collections import defaultdict
from kaggle_environments.envs.kore_fleets.helpers import Configuration


# <--->
from .basic import (
    Obj,
    collection_rate_for_ship_count,
    max_ships_to_spawn,
    cached_property,
    create_spawn_ships_command,
    create_launch_fleet_command,
)
from .geometry import (
    Field,
    Action,
    Point,
    North,
    South,
    Convert,
    PlanPath,
    PlanRoute,
    GAME_ID_TO_ACTION,
)
from .logger import logger

# <--->


class _ShipyardAction:
    def to_str(self):
        raise NotImplementedError

    def __repr__(self):
        return self.to_str()


class Spawn(_ShipyardAction):
    def __init__(self, ship_count: int):
        self.ship_count = ship_count

    def to_str(self):
        return create_spawn_ships_command(self.ship_count)


class Launch(_ShipyardAction):
    def __init__(self, ship_count: int, route: "BoardRoute"):
        self.ship_count = ship_count
        self.route = route

    def to_str(self):
        return create_launch_fleet_command(self.ship_count, self.route.plan.to_str())


class DoNothing(_ShipyardAction):
    def __repr__(self):
        return "Do nothing"

    def to_str(self):
        raise NotImplementedError


class BoardPath:
    max_length = 32

    def __init__(self, start: "Point", plan: PlanPath):
        assert plan.num_steps > 0 or plan.direction == Convert

        self._plan = plan

        field = start.field
        x, y = start.x, start.y
        if np.isfinite(plan.num_steps):
            n = plan.num_steps + 1
        else:
            n = self.max_length
        action = plan.direction

        if plan.direction == Convert:
            self._track = []
            self._start = start
            self._end = start
            self._build_shipyard = True
            return

        if action in (North, South):
            track = field.get_column(x, start=y, size=n * action.dy)
        else:
            track = field.get_row(y, start=x, size=n * action.dx)

        self._track = track[1:]
        self._start = start
        self._end = track[-1]
        self._build_shipyard = False

    def __repr__(self):
        start, end = self.start, self.end
        return f"({start.x}, {start.y}) -> ({end.x}, {end.y})"

    def __len__(self):
        return len(self._track)

    @property
    def plan(self):
        return self._plan

    @property
    def points(self):
        return self._track

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end


class BoardRoute:
    def __init__(self, start: "Point", plan: "PlanRoute"):
        paths = []
        for p in plan.paths:
            path = BoardPath(start, p)
            start = path.end
            paths.append(path)

        self._plan = plan
        self._paths = paths
        self._start = paths[0].start
        self._end = paths[-1].end

    def __repr__(self):
        points = []
        for p in self._paths:
            points.append(p.start)
        points.append(self.end)
        return " -> ".join([f"({p.x}, {p.y})" for p in points])

    def __iter__(self) -> Generator["Point", None, None]:
        for p in self._paths:
            yield from p.points

    def __len__(self):
        return sum(len(x) for x in self._paths)

    def points(self) -> List["Point"]:
        points = []
        for p in self._paths:
            points += p.points
        return points

    @property
    def plan(self) -> PlanRoute:
        return self._plan

    def command(self) -> str:
        return self.plan.to_str()

    @property
    def paths(self) -> List[BoardPath]:
        return self._paths

    @property
    def start(self) -> "Point":
        return self._start

    @property
    def end(self) -> "Point":
        return self._end

    def command_length(self) -> int:
        return len(self.command())

    def last_action(self):
        return self.paths[-1].plan.direction

    def expected_kore(self, board: "Board", ship_count: int):
        rate = collection_rate_for_ship_count(ship_count)
        if rate <= 0:
            return 0

        point_to_time = {}
        point_to_kore = {}
        for t, p in enumerate(self):
            point_to_time[p] = t
            point_to_kore[p] = p.kore

        for f in board.fleets:
            for t, p in enumerate(f.route):
                if p in point_to_time and t < point_to_time[p]:
                    point_to_kore[p] *= f.collection_rate

        return sum([kore * rate for kore in point_to_kore.values()])


class PositionObj(Obj):
    def __init__(self, *args, point: Point, player_id: int, board: "Board", **kwargs):
        super().__init__(*args, **kwargs)
        self._point = point
        self._player_id = player_id
        self._board = board

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id}, position={self._point}, player={self._player_id})"

    def dirs_to(self, obj: Union["PositionObj", Point]):
        if isinstance(obj, Point):
            return self._point.dirs_to(obj)
        return self._point.dirs_to(obj.point)

    def distance_from(self, obj: Union["PositionObj", Point]) -> int:
        if isinstance(obj, Point):
            return self._point.distance_from(obj)
        return self._point.distance_from(obj.point)

    @property
    def board(self) -> "Board":
        return self._board

    @property
    def point(self) -> Point:
        return self._point

    @property
    def player_id(self):
        return self._player_id

    @property
    def player(self) -> "Player":
        return self.board.get_player(self.player_id)


class Shipyard(PositionObj):
    def __init__(self, *args, ship_count: int, turns_controlled: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._ship_count = ship_count
        self._turns_controlled = turns_controlled
        self._guard_ship_count = 0
        self.action: Optional[_ShipyardAction] = None

    @property
    def turns_controlled(self):
        return self._turns_controlled

    @property
    def max_ships_to_spawn(self) -> int:
        return max_ships_to_spawn(self._turns_controlled)

    @property
    def ship_count(self):
        return self._ship_count

    @property
    def available_ship_count(self):
        return self._ship_count - self._guard_ship_count

    @property
    def guard_ship_count(self):
        return self._guard_ship_count

    def set_guard_ship_count(self, ship_count):
        assert ship_count <= self._ship_count
        self._guard_ship_count = ship_count

    @cached_property
    def incoming_allied_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id == self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets

    @cached_property
    def incoming_hostile_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id != self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets


class Fleet(PositionObj):
    def __init__(
        self,
        *args,
        ship_count: int,
        kore: int,
        route: BoardRoute,
        direction: Action,
        **kwargs,
    ):
        assert ship_count > 0
        assert kore >= 0

        super().__init__(*args, **kwargs)

        self._ship_count = ship_count
        self._kore = kore
        self._direction = direction
        self._route = route

    def __gt__(self, other):
        if self.ship_count != other.ship_count:
            return self.ship_count > other.ship_count
        if self.kore != other.kore:
            return self.kore > other.kore
        return self.direction.game_id > other.direction.game_id

    def __lt__(self, other):
        return other.__gt__(self)

    @property
    def ship_count(self):
        return self._ship_count

    @property
    def kore(self):
        return self._kore

    @property
    def route(self):
        return self._route

    @property
    def eta(self):
        return len(self._route)

    def set_route(self, route: BoardRoute):
        self._route = route

    @property
    def direction(self):
        return self._direction

    @property
    def collection_rate(self) -> float:
        return collection_rate_for_ship_count(self._ship_count)

    def expected_kore(self):
        return self._kore + self._route.expected_kore(self._board, self._ship_count)

    def cost(self):
        return self.board.spawn_cost * self.ship_count

    def value(self):
        return self.kore / self.cost()

    def expected_value(self):
        return self.expected_kore() / self.cost()


class FleetPointer:
    def __init__(self, fleet: Fleet):
        self.obj = fleet
        self.point = fleet.point
        self.is_active = True
        self._paths = []
        self._points = self.points()

    def points(self):
        for path in self.obj.route.paths:
            self._paths.append([path.plan.direction, 0])
            for point in path.points:
                self._paths[-1][1] += 1
                yield point

    def update(self):
        if not self.is_active:
            self.point = None
            return
        try:
            self.point = next(self._points)
        except StopIteration:
            self.point = None
            self.is_active = False

    def current_route(self):
        plan = PlanRoute([PlanPath(d, n) for d, n in self._paths])
        return BoardRoute(self.obj.point, plan)


class Player(Obj):
    def __init__(self, *args, kore: float, board: "Board", **kwargs):
        super().__init__(*args, **kwargs)
        self._kore = kore
        self._board = board

    @property
    def kore(self):
        return self._kore

    def fleet_kore(self):
        return sum(x.kore for x in self.fleets)

    def fleet_expected_kore(self):
        return sum(x.expected_kore() for x in self.fleets)

    def is_active(self):
        return len(self.fleets) > 0 or len(self.shipyards) > 0

    @property
    def board(self):
        return self._board

    def _get_objects(self, name):
        d = []
        for x in self._board.__getattribute__(name):
            if x.player_id == self.game_id:
                d.append(x)
        return d

    @cached_property
    def fleets(self) -> List[Fleet]:
        return self._get_objects("fleets")

    @cached_property
    def shipyards(self) -> List[Shipyard]:
        return self._get_objects("shipyards")

    @cached_property
    def ship_count(self) -> int:
        return sum(x.ship_count for x in itertools.chain(self.fleets, self.shipyards))

    @cached_property
    def opponents(self) -> List["Player"]:
        return [x for x in self.board.players if x != self]

    @cached_property
    def expected_fleets_positions(self) -> Dict[int, Dict[Point, int]]:
        """
        time -> point -> fleet
        """
        time_to_fleet_positions = defaultdict(dict)
        for f in self.fleets:
            for time, point in enumerate(f.route):
                time_to_fleet_positions[time][point] = f
        return time_to_fleet_positions

    @cached_property
    def expected_dmg_positions(self) -> Dict[int, Dict[Point, int]]:
        """
        time -> point -> dmg
        """
        time_to_dmg_positions = defaultdict(dict)
        for f in self.fleets:
            for time, point in enumerate(f.route):
                for adjacent_point in point.adjacent_points:
                    point_to_dmg = time_to_dmg_positions[time]
                    if adjacent_point not in point_to_dmg:
                        point_to_dmg[adjacent_point] = 0
                    point_to_dmg[adjacent_point] += f.ship_count
        return time_to_dmg_positions

    def actions(self):
        if self.available_kore() < 0:
            logger.warning("Negative balance. Some ships will not spawn.")

        shipyard_id_to_action = {}
        for sy in self.shipyards:
            if not sy.action or isinstance(sy.action, DoNothing):
                continue

            shipyard_id_to_action[sy.game_id] = sy.action.to_str()
        return shipyard_id_to_action

    def spawn_ship_count(self):
        return sum(
            x.action.ship_count for x in self.shipyards if isinstance(x.action, Spawn)
        )

    def need_kore_for_spawn(self):
        return self.board.spawn_cost * self.spawn_ship_count()

    def available_kore(self):
        return self._kore - self.need_kore_for_spawn()


_FIELD = None


class Board:
    def __init__(self, obs, conf):
        self._conf = Configuration(conf)
        self._step = obs["step"]

        global _FIELD
        if _FIELD is None or self._step == 0:
            _FIELD = Field(self._conf.size)
        else:
            assert _FIELD.size == self._conf.size

        self._field: Field = _FIELD

        id_to_point = {x.game_id: x for x in self._field}

        for point_id, kore in enumerate(obs["kore"]):
            point = id_to_point[point_id]
            point.set_kore(kore)

        self._players = []
        self._fleets = []
        self._shipyards = []
        for player_id, player_data in enumerate(obs["players"]):
            player_kore, player_shipyards, player_fleets = player_data
            player = Player(game_id=player_id, kore=player_kore, board=self)
            self._players.append(player)

            for fleet_id, fleet_data in player_fleets.items():
                point_id, kore, ship_count, direction, flight_plan = fleet_data
                position = id_to_point[point_id]
                direction = GAME_ID_TO_ACTION[direction]
                if ship_count < self.shipyard_cost and Convert.command in flight_plan:
                    # can't convert
                    flight_plan = "".join(
                        [x for x in flight_plan if x != Convert.command]
                    )
                plan = PlanRoute.from_str(flight_plan, direction)
                route = BoardRoute(position, plan)
                fleet = Fleet(
                    game_id=fleet_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    kore=kore,
                    route=route,
                    direction=direction,
                    board=self,
                )
                self._fleets.append(fleet)

            for shipyard_id, shipyard_data in player_shipyards.items():
                point_id, ship_count, turns_controlled = shipyard_data
                position = id_to_point[point_id]
                shipyard = Shipyard(
                    game_id=shipyard_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    turns_controlled=turns_controlled,
                    board=self,
                )
                self._shipyards.append(shipyard)

        self._players = [x for x in self._players if x.is_active()]

        self._update_fleets_destination()

    def __getitem__(self, item):
        return self._field[item]

    def __iter__(self):
        return self._field.__iter__()

    @property
    def field(self):
        return self._field

    @property
    def size(self):
        return self._field.size

    @property
    def step(self):
        return self._step

    @property
    def steps_left(self):
        return self._conf.episode_steps - self._step - 1

    @property
    def shipyard_cost(self):
        return self._conf.convert_cost

    @property
    def spawn_cost(self):
        return self._conf.spawn_cost

    @property
    def regen_rate(self):
        return self._conf.regen_rate

    @property
    def max_cell_kore(self):
        return self._conf.max_cell_kore

    @property
    def players(self) -> List[Player]:
        return self._players

    @property
    def fleets(self) -> List[Fleet]:
        return self._fleets

    @property
    def shipyards(self) -> List[Shipyard]:
        return self._shipyards

    def get_player(self, game_id) -> Player:
        for p in self._players:
            if p.game_id == game_id:
                return p
        raise KeyError(f"Player `{game_id}` doas not exists.")

    def get_obj_at_point(self, point: Point) -> Optional[Union[Fleet, Shipyard]]:
        for x in itertools.chain(self.fleets, self.shipyards):
            if x.point == point:
                return x

    def _update_fleets_destination(self):
        """
        trying to predict future positions
        very inaccurate
        """

        shipyard_positions = {x.point for x in self.shipyards}

        fleets = [FleetPointer(f) for f in self.fleets]

        while any(x.is_active for x in fleets):
            for f in fleets:
                f.update()

            # fleet to shipyard
            for f in fleets:
                if f.point in shipyard_positions:
                    f.is_active = False

            # allied fleets
            for player in self.players:
                point_to_fleets = defaultdict(list)
                for f in fleets:
                    if f.is_active and f.obj.player_id == player.game_id:
                        point_to_fleets[f.point].append(f)
                for point_fleets in point_to_fleets.values():
                    if len(point_fleets) > 1:
                        for f in sorted(point_fleets, key=lambda x: x.obj)[:-1]:
                            f.is_active = False

            # fleet to fleet
            point_to_fleets = defaultdict(list)
            for f in fleets:
                if f.is_active:
                    point_to_fleets[f.point].append(f)
            for point_fleets in point_to_fleets.values():
                if len(point_fleets) > 1:
                    for f in sorted(point_fleets, key=lambda x: x.obj)[:-1]:
                        f.is_active = False

            # adjacent damage
            point_to_fleet = {}
            for f in fleets:
                if f.is_active:
                    point_to_fleet[f.point] = f

            point_to_dmg = defaultdict(int)
            for point, fleet in point_to_fleet.items():
                for p in point.adjacent_points:
                    if p in point_to_fleet:
                        adjacent_fleet = point_to_fleet[p]
                        if adjacent_fleet.obj.player_id != fleet.obj.player_id:
                            point_to_dmg[p] += fleet.obj.ship_count

            for point, fleet in point_to_fleet.items():
                dmg = point_to_dmg[point]
                if fleet.obj.ship_count <= dmg:
                    fleet.is_active = False

        for f in fleets:
            f.obj.set_route(f.current_route())
