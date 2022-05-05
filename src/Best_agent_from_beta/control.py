import random

# <--->
from .geometry import PlanRoute
from .board import Player, Launch, Spawn, Fleet, FleetPointer, BoardRoute
from .helpers import is_invitable_victory, find_shortcut_routes
from .logger import logger

# <--->


def direct_attack(agent: Player, max_distance: int = 10):
    board = agent.board

    max_distance = min(board.steps_left, max_distance)

    targets = []
    for x in agent.opponents:
        for sy in x.shipyards:
            for fleet in sy.incoming_allied_fleets:
                if fleet.expected_value() > 0.5:
                    targets.append(fleet)

    if not targets:
        return

    shipyards = [
        x for x in agent.shipyards if x.available_ship_count > 0 and not x.action
    ]
    if not shipyards:
        return

    point_to_closest_shipyard = {}
    for p in board:
        closest_shipyard = None
        min_distance = board.size
        for sy in agent.shipyards:
            distance = sy.point.distance_from(p)
            if distance < min_distance:
                min_distance = distance
                closest_shipyard = sy
        point_to_closest_shipyard[p] = closest_shipyard.point

    opponent_shipyard_points = {x.point for x in board.shipyards if x.player_id != agent.game_id}
    for t in targets:
        min_ships_to_send = int(t.ship_count * 1.2)
        attacked = False

        for sy in shipyards:
            if sy.action or sy.available_ship_count < min_ships_to_send:
                continue

            num_ships_to_launch = sy.available_ship_count

            for target_time, target_point in enumerate(t.route, 1):
                if target_time > max_distance:
                    continue

                if sy.point.distance_from(target_point) != target_time:
                    continue

                paths = sy.point.dirs_to(target_point)
                random.shuffle(paths)
                plan = PlanRoute(paths)
                destination = point_to_closest_shipyard[target_point]

                paths = target_point.dirs_to(destination)
                random.shuffle(paths)
                plan += PlanRoute(paths)
                if num_ships_to_launch < plan.min_fleet_size():
                    continue

                route = BoardRoute(sy.point, plan)

                if any(x in opponent_shipyard_points for x in route.points()):
                    continue

                if is_intercept_direct_attack_route(route, agent, direct_attack_fleet=t):
                    continue

                logger.info(
                    f"Direct attack {sy.point}->{target_point}, distance={target_time}"
                )
                sy.action = Launch(num_ships_to_launch, route)
                attacked = True
                break

            if attacked:
                break


def is_intercept_direct_attack_route(
    route: BoardRoute, player: Player, direct_attack_fleet: Fleet
):
    board = player.board

    fleets = [FleetPointer(f) for f in board.fleets if f != direct_attack_fleet]

    for point in route.points()[:-1]:
        for fleet in fleets:
            fleet.update()

            if fleet.point is None:
                continue

            if fleet.point == point:
                return True

            if fleet.obj.player_id != player.game_id:
                for p in fleet.point.adjacent_points:
                    if p == point:
                        return True

    return False


def adjacent_attack(agent: Player, max_distance: int = 10):
    board = agent.board

    max_distance = min(board.steps_left, max_distance)

    targets = _find_adjacent_targets(agent, max_distance)
    if not targets:
        return

    shipyards = [
        x for x in agent.shipyards if x.available_ship_count > 0 and not x.action
    ]
    if not shipyards:
        return

    fleets_to_be_attacked = set()
    for t in sorted(targets, key=lambda x: (-len(x["fleets"]), x["time"])):
        target_point = t["point"]
        target_time = t["time"]
        target_fleets = t["fleets"]
        if any(x in fleets_to_be_attacked for x in target_fleets):
            continue

        for sy in shipyards:
            if sy.action:
                continue

            distance = sy.distance_from(target_point)
            if distance > target_time:
                continue
            min_ship_count = min(x.ship_count for x in target_fleets)
            num_ships_to_send = min(sy.available_ship_count, min_ship_count)

            routes = find_shortcut_routes(
                board,
                sy.point,
                target_point,
                agent,
                num_ships_to_send,
                route_distance=target_time,
            )
            if not routes:
                continue

            route = random.choice(routes)
            logger.info(
                f"Adjacent attack {sy.point}->{target_point}, distance={distance}, target_time={target_time}"
            )
            sy.action = Launch(num_ships_to_send, route)
            for fleet in target_fleets:
                fleets_to_be_attacked.add(fleet)
            break


def _find_adjacent_targets(agent: Player, max_distance: int = 5):
    board = agent.board
    shipyards_points = {x.point for x in board.shipyards}
    fleets = [FleetPointer(f) for f in board.fleets]
    if len(fleets) < 2:
        return []

    time = 0
    targets = []
    while any(x.is_active for x in fleets) and time <= max_distance:
        time += 1

        for f in fleets:
            f.update()

        point_to_fleet = {
            x.point: x.obj
            for x in fleets
            if x.is_active and x.point not in shipyards_points
        }

        for point in board:
            if point in point_to_fleet or point in shipyards_points:
                continue

            adjacent_fleets = [
                point_to_fleet[x] for x in point.adjacent_points if x in point_to_fleet
            ]
            if len(adjacent_fleets) < 2:
                continue

            if any(x.player_id == agent.game_id for x in adjacent_fleets):
                continue

            targets.append({"point": point, "time": time, "fleets": adjacent_fleets})

    return targets


def _need_more_ships(agent: Player, ship_count: int):
    board = agent.board
    if board.steps_left < 10:
        return False
    if ship_count > _max_ships_to_control(agent):
        return False
    if board.steps_left < 50 and is_invitable_victory(agent):
        return False
    return True


def _max_ships_to_control(agent: Player):
    return max(100, 3 * sum(x.ship_count for x in agent.opponents))


def greedy_spawn(agent: Player):
    board = agent.board

    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    max_ship_count = _max_ships_to_control(agent)
    for shipyard in agent.shipyards:
        if shipyard.action:
            continue

        if shipyard.ship_count > agent.ship_count * 0.2 / len(agent.shipyards):
            continue

        num_ships_to_spawn = shipyard.max_ships_to_spawn
        if int(agent.available_kore() // board.spawn_cost) >= num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)

        ship_count += num_ships_to_spawn
        if ship_count > max_ship_count:
            return


def spawn(agent: Player):
    board = agent.board

    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    max_ship_count = _max_ships_to_control(agent)
    for shipyard in agent.shipyards:
        if shipyard.action:
            continue
        num_ships_to_spawn = min(
            int(agent.available_kore() // board.spawn_cost),
            shipyard.max_ships_to_spawn,
        )
        if num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)
            ship_count += num_ships_to_spawn
            if ship_count > max_ship_count:
                return
