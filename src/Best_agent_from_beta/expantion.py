import random
from typing import List
from collections import defaultdict

# <--->
from .basic import min_ship_count_for_flight_plan_len
from .geometry import Point, Convert, PlanRoute, PlanPath
from .board import Player, BoardRoute, Launch

# <--->


def expand(player: Player):
    board = player.board
    num_shipyards_to_create = need_more_shipyards(player)
    if not num_shipyards_to_create:
        return

    shipyard_positions = {x.point for x in board.shipyards}

    shipyard_to_point = find_best_position_for_shipyards(player)

    shipyard_count = 0
    for shipyard, target in shipyard_to_point.items():
        if shipyard_count >= num_shipyards_to_create:
            break

        if shipyard.available_ship_count < board.shipyard_cost or shipyard.action:
            continue

        incoming_hostile_fleets = shipyard.incoming_hostile_fleets
        if incoming_hostile_fleets:
            continue

        target_distance = shipyard.distance_from(target)

        routes = []
        for p in board:
            if p in shipyard_positions:
                continue

            distance = shipyard.distance_from(p) + p.distance_from(target)
            if distance > target_distance:
                continue

            plan = PlanRoute(shipyard.dirs_to(p) + p.dirs_to(target))
            route = BoardRoute(shipyard.point, plan)

            if shipyard.available_ship_count < min_ship_count_for_flight_plan_len(
                len(route.plan.to_str()) + 1
            ):
                continue

            route_points = route.points()
            if any(x in shipyard_positions for x in route_points):
                continue

            if not is_safety_route_to_convert(route_points, player):
                continue

            routes.append(route)

        if routes:
            route = random.choice(routes)
            route = BoardRoute(
                shipyard.point, route.plan + PlanRoute([PlanPath(Convert)])
            )
            shipyard.action = Launch(shipyard.available_ship_count, route)
            shipyard_count += 1


def find_best_position_for_shipyards(player: Player):
    board = player.board
    shipyards = board.shipyards

    shipyard_to_scores = defaultdict(list)
    for p in board:
        if p.kore > 50:
            continue

        closed_shipyard = None
        min_distance = board.size
        for shipyard in shipyards:
            distance = shipyard.point.distance_from(p)
            if shipyard.player_id != player.game_id:
                distance -= 1

            if distance < min_distance:
                closed_shipyard = shipyard
                min_distance = distance

        if (
            not closed_shipyard
            or closed_shipyard.player_id != player.game_id
            or min_distance < 3
            or min_distance > 5
        ):
            continue

        nearby_kore = sum(x.kore for x in p.nearby_points(10))
        nearby_shipyards = sum(1 for x in board.shipyards if x.distance_from(p) < 5)
        score = nearby_kore - 1000 * nearby_shipyards - 1000 * min_distance
        shipyard_to_scores[closed_shipyard].append({"score": score, "point": p})

    shipyard_to_point = {}
    for shipyard, scores in shipyard_to_scores.items():
        if scores:
            scores = sorted(scores, key=lambda x: x["score"])
            point = scores[-1]["point"]
            shipyard_to_point[shipyard] = point

    return shipyard_to_point


def need_more_shipyards(player: Player) -> int:
    board = player.board

    if player.ship_count < 100:
        return 0

    fleet_distance = []
    for sy in player.shipyards:
        for f in sy.incoming_allied_fleets:
            fleet_distance.append(len(f.route))

    if not fleet_distance:
        return 0

    mean_fleet_distance = sum(fleet_distance) / len(fleet_distance)

    shipyard_production_capacity = sum(x.max_ships_to_spawn for x in player.shipyards)

    steps_left = board.steps_left
    if steps_left > 100:
        scale = 3
    elif steps_left > 50:
        scale = 4
    elif steps_left > 10:
        scale = 100
    else:
        scale = 1000

    needed = player.kore > scale * shipyard_production_capacity * mean_fleet_distance
    if not needed:
        return 0

    current_shipyard_count = len(player.shipyards)

    op_shipyard_positions = {
        x.point for x in board.shipyards if x.player_id != player.game_id
    }
    expected_shipyard_count = current_shipyard_count + sum(
        1
        for x in player.fleets
        if x.route.last_action() == Convert or x.route.end in op_shipyard_positions
    )

    opponent_shipyard_count = max(len(x.shipyards) for x in player.opponents)
    opponent_ship_count = max(x.ship_count for x in player.opponents)
    if (
        expected_shipyard_count > opponent_shipyard_count
        and player.ship_count < opponent_ship_count
    ):
        return 0

    if current_shipyard_count < 10:
        if expected_shipyard_count > current_shipyard_count:
            return 0
        else:
            return 1

    return max(0, 5 - (expected_shipyard_count - current_shipyard_count))


def is_safety_route_to_convert(route_points: List[Point], player: Player):
    board = player.board

    target_point = route_points[-1]
    target_time = len(route_points)
    for pl in board.players:
        if pl != player:
            for t, positions in pl.expected_fleets_positions.items():
                if t >= target_time and target_point in positions:
                    return False

    shipyard_positions = {x.point for x in board.shipyards}

    for time, point in enumerate(route_points):
        for pl in board.players:
            if point in shipyard_positions:
                return False

            is_enemy = pl != player

            if point in pl.expected_fleets_positions[time]:
                return False

            if is_enemy:
                if point in pl.expected_dmg_positions[time]:
                    return False

    return True
