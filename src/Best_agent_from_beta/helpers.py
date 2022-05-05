import random
from typing import List


# <--->
from .geometry import Point
from .board import Board, Player, BoardRoute, PlanRoute

# <--->


def is_intercept_route(
    route: BoardRoute, player: Player, safety=True, allow_shipyard_intercept=False
):
    board = player.board

    if not allow_shipyard_intercept:
        shipyard_points = {x.point for x in board.shipyards}
    else:
        shipyard_points = {}

    for time, point in enumerate(route.points()[:-1]):
        if point in shipyard_points:
            return True

        for pl in board.players:
            is_enemy = pl != player

            if point in pl.expected_fleets_positions[time]:
                return True

            if safety and is_enemy:
                if point in pl.expected_dmg_positions[time]:
                    return True

    return False


def find_shortcut_routes(
    board: Board,
    start: Point,
    end: Point,
    player: Player,
    num_ships: int,
    safety: bool = True,
    allow_shipyard_intercept=False,
    route_distance=None
) -> List[BoardRoute]:
    if route_distance is None:
        route_distance = start.distance_from(end)
    routes = []
    for p in board:
        distance = start.distance_from(p) + p.distance_from(end)
        if distance != route_distance:
            continue

        path1 = start.dirs_to(p)
        path2 = p.dirs_to(end)
        random.shuffle(path1)
        random.shuffle(path2)

        plan = PlanRoute(path1 + path2)

        if num_ships < plan.min_fleet_size():
            continue

        route = BoardRoute(start, plan)

        if is_intercept_route(
            route,
            player,
            safety=safety,
            allow_shipyard_intercept=allow_shipyard_intercept,
        ):
            continue

        routes.append(route)

    return routes


def is_invitable_victory(player: Player):
    if not player.opponents:
        return True

    board = player.board
    if board.steps_left > 100:
        return False

    board_kore = sum(x.kore for x in board) * (1 + board.regen_rate) ** board.steps_left

    player_kore = player.kore + player.fleet_expected_kore()
    opponent_kore = max(x.kore + x.fleet_expected_kore() for x in player.opponents)
    return player_kore > opponent_kore + board_kore
