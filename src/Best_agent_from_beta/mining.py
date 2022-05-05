import random
import numpy as np
from typing import List
from collections import defaultdict

# <--->
from .geometry import PlanRoute
from .board import Player, BoardRoute, Launch, Shipyard
from .helpers import is_intercept_route

# <--->


def mine(agent: Player):
    board = agent.board
    if not agent.opponents:
        return

    safety = False
    my_ship_count = agent.ship_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    if my_ship_count < 2 * op_ship_count:
        safety = True

    op_ship_count = []
    for op in agent.opponents:
        for fleet in op.fleets:
            op_ship_count.append(fleet.ship_count)

    if not op_ship_count:
        mean_fleet_size = 0
        max_fleet_size = np.inf
    else:
        mean_fleet_size = np.percentile(op_ship_count, 75)
        max_fleet_size = int(max(op_ship_count) * 1.1)

    point_to_score = estimate_board_risk(agent)

    shipyard_count = len(agent.shipyards)
    if shipyard_count < 10:
        max_distance = 15
    elif shipyard_count < 20:
        max_distance = 12
    else:
        max_distance = 8

    max_distance = min(int(board.steps_left // 2), max_distance)

    for sy in agent.shipyards:
        if sy.action:
            continue

        free_ships = sy.available_ship_count

        if free_ships <= 2:
            continue

        routes = find_shipyard_mining_routes(
            sy, safety=safety, max_distance=max_distance
        )

        route_to_score = {}
        for route in routes:
            route_points = route.points()

            if all(point_to_score[x] > 0 for x in route_points):
                num_ships_to_launch = free_ships
            else:
                if free_ships < mean_fleet_size:
                    continue
                num_ships_to_launch = min(free_ships, max_fleet_size)

            score = route.expected_kore(board, num_ships_to_launch) / len(route)
            route_to_score[route] = score

        if not route_to_score:
            continue

        routes = sorted(route_to_score, key=lambda x: -route_to_score[x])
        for route in routes:
            if all(point_to_score[x] >= 1 for x in route):
                num_ships_to_launch = free_ships
            else:
                num_ships_to_launch = min(free_ships, 199)
            if num_ships_to_launch < route.plan.min_fleet_size():
                continue
            else:
                sy.action = Launch(num_ships_to_launch, route)
                break


def estimate_board_risk(player: Player):
    board = player.board

    shipyard_to_area = defaultdict(list)
    for p in board:
        closest_shipyard = None
        min_distance = board.size
        for sh in board.shipyards:
            distance = sh.point.distance_from(p)
            if distance < min_distance:
                closest_shipyard = sh
                min_distance = distance

        shipyard_to_area[closest_shipyard].append(p)

    point_to_score = {}
    for sy, points in shipyard_to_area.items():
        if sy.player_id == player.game_id:
            for p in points:
                point_to_score[p] = 1
        else:
            for p in points:
                point_to_score[p] = -1

    return point_to_score


def find_shipyard_mining_routes(
    sy: Shipyard, safety=True, max_distance: int = 15
) -> List[BoardRoute]:
    if max_distance < 1:
        return []

    departure = sy.point
    player = sy.player

    destinations = set()
    for shipyard in sy.player.shipyards:
        siege = sum(x.ship_count for x in shipyard.incoming_hostile_fleets)
        if siege >= shipyard.ship_count:
            continue
        destinations.add(shipyard.point)

    if not destinations:
        return []

    routes = []
    for c in sy.point.nearby_points(max_distance):
        if c == departure or c in destinations:
            continue

        paths = departure.dirs_to(c)
        random.shuffle(paths)
        plan = PlanRoute(paths)
        destination = sorted(destinations, key=lambda x: c.distance_from(x))[0]
        if destination == departure:
            plan += plan.reverse()
        else:
            paths = c.dirs_to(destination)
            random.shuffle(paths)
            plan += PlanRoute(paths)

        route = BoardRoute(departure, plan)

        if is_intercept_route(route, player, safety):
            continue

        routes.append(BoardRoute(departure, plan))

    return routes
