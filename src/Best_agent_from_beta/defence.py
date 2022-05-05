import random

# <--->
from .board import Spawn, Player, Launch
from .helpers import find_shortcut_routes
from .logger import logger

# <--->


def defend_shipyards(agent: Player):
    board = agent.board

    need_help_shipyards = []
    for sy in agent.shipyards:
        if sy.action:
            continue

        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_allied_fleets = sy.incoming_allied_fleets

        if not incoming_hostile_fleets:
            continue

        incoming_hostile_power = sum(x.ship_count for x in incoming_hostile_fleets)
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)
        incoming_allied_power = sum(
            x.ship_count
            for x in incoming_allied_fleets
            if x.eta < incoming_hostile_time
        )

        ships_needed = incoming_hostile_power - incoming_allied_power
        if sy.ship_count > ships_needed:
            sy.set_guard_ship_count(min(sy.ship_count, int(ships_needed * 1.1)))
            continue

        # spawn as much as possible
        num_ships_to_spawn = min(
            int(agent.available_kore() // board.spawn_cost), sy.max_ships_to_spawn
        )
        if num_ships_to_spawn:
            logger.debug(f"Spawn ships to protect shipyard {sy.point}")
            sy.action = Spawn(num_ships_to_spawn)

        need_help_shipyards.append(sy)

    for sy in need_help_shipyards:
        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)

        for other_sy in agent.shipyards:
            if other_sy == sy or other_sy.action or not other_sy.available_ship_count:
                continue

            distance = other_sy.distance_from(sy)
            if distance == incoming_hostile_time - 1:
                routes = find_shortcut_routes(
                    board, other_sy.point, sy.point, agent, other_sy.ship_count
                )
                if routes:
                    logger.info(f"Send reinforcements {other_sy.point}->{sy.point}")
                    other_sy.action = Launch(
                        other_sy.available_ship_count, random.choice(routes)
                    )
            elif distance < incoming_hostile_time - 1:
                other_sy.set_guard_ship_count(other_sy.ship_count)
