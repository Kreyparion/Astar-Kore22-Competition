import numpy as np
from typing import Tuple, List, Generator

# <--->
from .basic import Obj, cached_call, cached_property, min_ship_count_for_flight_plan_len

# <--->


class Action(Obj):
    def __init__(self, dx, dy, game_id, command):
        super().__init__(game_id)
        self._dx = dx
        self._dy = dy
        self._command = command

    def __repr__(self):
        return self._command

    @property
    def dx(self) -> int:
        return self._dx

    @property
    def dy(self) -> int:
        return self._dy

    @property
    def command(self) -> str:
        return self._command


North = Action(
    dx=0,
    dy=1,
    game_id=0,
    command="N",
)
East = Action(
    dx=1,
    dy=0,
    game_id=1,
    command="E",
)
South = Action(
    dx=0,
    dy=-1,
    command="S",
    game_id=2,
)
West = Action(
    dx=-1,
    dy=0,
    command="W",
    game_id=3,
)
Convert = Action(
    dx=0,
    dy=0,
    command="C",
    game_id=-1,
)


ALL_DIRECTIONS = {North, East, South, West}
ALL_ACTIONS = {North, East, South, West, Convert}
GAME_ID_TO_ACTION = {x.game_id: x for x in ALL_ACTIONS}
COMMAND_TO_ACTION = {x.command: x for x in ALL_ACTIONS}
ACTION_TO_OPPOSITE_ACTION = {
    North: South,
    East: West,
    South: North,
    West: East,
}


def get_opposite_action(action):
    return ACTION_TO_OPPOSITE_ACTION.get(action, action)


class Point(Obj):
    def __init__(self, x: int, y: int, kore: float, field: "Field"):
        super().__init__(game_id=(field.size - y - 1) * field.size + x)
        self._x = x
        self._y = y
        self._kore = kore
        self._field = field

    def __repr__(self):
        return f"Point({self._x}, {self._y})"

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    def to_tuple(self) -> Tuple[int, int]:
        return self._x, self._y

    @property
    def kore(self) -> float:
        return self._kore

    def set_kore(self, kore: float):
        self._kore = kore

    @property
    def field(self) -> "Field":
        return self._field

    def apply(self, action: Action) -> "Point":
        return self._field[(self.x + action.dx, self.y + action.dy)]

    @cached_call
    def distance_from(self, point: "Point") -> int:
        return sum(p.num_steps for p in self.dirs_to(point))

    @cached_property
    def adjacent_points(self) -> List["Point"]:
        return [self.apply(a) for a in ALL_DIRECTIONS]

    @cached_property
    def row(self) -> List["Point"]:
        return list(self._field.points[:, self.y])

    @cached_property
    def column(self) -> List["Point"]:
        return list(self._field.points[self.x, :])

    @cached_call
    def nearby_points(self, r: int) -> List["Point"]:
        if r > 1:
            points = []
            for p in self._field:
                distance = self.distance_from(p)
                if 0 < distance <= r:
                    points.append(p)
            return points
        elif r == 1:
            return self.adjacent_points

        raise ValueError("Radius must be more or equal then 1")

    @cached_call
    def dirs_to(self, point: "Point") -> List["PlanPath"]:
        dx, dy = self._field.swap(self._x - point.x, self._y - point.y)
        ret = []
        if dx:
            ret.append(PlanPath(West, dx))
        if dy:
            ret.append(PlanPath(South, dy))
        return ret


class Field:
    def __init__(self, size: int):
        self._size = size
        self._points = self.create_array(size)

    def __iter__(self) -> Generator[Point, None, None]:
        for row in self._points:
            yield from row

    def create_array(self, size: int) -> np.ndarray:
        ar = np.zeros((size, size), dtype=Point)
        for x in range(size):
            for y in range(size):
                point = Point(x, y, kore=0, field=self)
                ar[x, y] = point
        return ar

    @property
    def points(self) -> np.ndarray:
        return self._points

    def get_row(self, y: int, start: int, size: int) -> List[Point]:
        if size < 0:
            return self.get_row(y, start=start + size + 1, size=-size)[::-1]

        ps = self._points
        start %= self._size
        out = []
        while size > 0:
            d = list(ps[slice(start, start + size), y])
            size -= len(d)
            start = 0
            out += d
        return out

    def get_column(self, x: int, start: int, size: int) -> List[Point]:
        if size < 0:
            return self.get_column(x, start=start + size + 1, size=-size)[::-1]

        ps = self._points
        start %= self._size
        out = []
        while size > 0:
            d = list(ps[x, slice(start, start + size)])
            size -= len(d)
            start = 0
            out += d
        return out

    @property
    def size(self) -> int:
        return self._size

    def swap(self, dx, dy):
        size = self._size
        if abs(dx) > size / 2:
            dx -= np.sign(dx) * size
        if abs(dy) > size / 2:
            dy -= np.sign(dy) * size
        return dx, dy

    def __getitem__(self, item) -> Point:
        x, y = item
        return self._points[x % self._size, y % self._size]


class PlanPath:
    def __init__(self, direction: Action, num_steps: int = 0):
        if direction == Convert:
            self._direction = direction
            self._num_steps = 0
        elif num_steps > 0:
            self._direction = direction
            self._num_steps = num_steps
        else:
            self._direction = get_opposite_action(direction)
            self._num_steps = -num_steps

    def __repr__(self):
        return self.to_str()

    @property
    def direction(self):
        return self._direction

    @property
    def num_steps(self):
        return self._num_steps

    def to_str(self):
        if self.direction == Convert:
            return Convert.command
        elif self.num_steps == 0:
            return ""
        elif self.num_steps == 1:
            return self.direction.command
        else:
            return self.direction.command + str(self.num_steps - 1)

    def reverse(self) -> "PlanPath":
        return PlanPath(self.direction, -self.num_steps)


class PlanRoute:
    def __init__(self, paths: List[PlanPath]):
        self._paths = self.simplify(paths)

    def __repr__(self):
        return self.to_str()

    def __add__(self, other: "PlanRoute") -> "PlanRoute":
        return PlanRoute(self.paths + other.paths)

    def __bool__(self):
        return bool(self._paths)

    @property
    def paths(self):
        return self._paths

    @property
    def num_steps(self):
        return sum(x.num_steps for x in self._paths)

    @classmethod
    def simplify(cls, paths: List[PlanPath]):
        if not paths:
            return paths

        new_paths = []
        last_path = None
        for p in paths:
            if last_path and p.direction == last_path.direction:
                new_paths[-1] = PlanPath(p.direction, p.num_steps + last_path.num_steps)
            else:
                last_path = p
                new_paths.append(p)
        return new_paths

    def command_length(self):
        return len(self.to_str())

    def min_fleet_size(self):
        return min_ship_count_for_flight_plan_len(self.command_length())

    def reverse(self) -> "PlanRoute":
        return PlanRoute([x.reverse() for x in self.paths])

    @property
    def actions(self):
        actions = []
        for p in self.paths:
            actions += [p.direction for _ in range(p.num_steps)]
        return actions

    @classmethod
    def from_str(cls, str_plan: str, current_direction: Action) -> "PlanRoute":
        if current_direction not in ALL_DIRECTIONS:
            raise ValueError(f"Unknown direction `{current_direction}`")

        if not str_plan:
            return PlanRoute([PlanPath(current_direction, np.inf)])

        commands = []
        for x in str_plan:
            if x in COMMAND_TO_ACTION:
                commands.append([])
                commands[-1].append(x)
            elif x.isdigit():
                if not commands:
                    commands = [[]]
                commands[-1].append(x)
            else:
                raise ValueError(f"Unknown command `{x}`.")

        paths = []
        for i, p in enumerate(commands):
            if i == 0 and p[0].isdigit():
                action = current_direction
                num_steps = int("".join(p))
                if num_steps == 0:
                    continue
            else:
                action = COMMAND_TO_ACTION[p[0]]
                if len(p) == 1:
                    num_steps = 1
                else:
                    num_steps = int("".join(p[1:])) + 1

            paths.append(PlanPath(direction=action, num_steps=num_steps))
            if action == Convert:
                break

        if not paths:
            return PlanRoute([PlanPath(current_direction, np.inf)])

        last_direction = paths[-1].direction
        if last_direction != Convert:
            paths[-1] = PlanPath(direction=last_direction, num_steps=np.inf)

        return PlanRoute(paths)

    def to_str(self) -> str:
        s = ""
        for a in self.paths[:-1]:
            s += a.to_str()
        s += self.paths[-1].direction.command
        return s
