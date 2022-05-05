import random

import numpy as np
from collections import defaultdict

# <--->
from .basic import max_ships_to_spawn
from .board import Player, Shipyard, Launch
from .helpers import find_shortcut_routes
from .logger import logger

# <--->


class _ShipyardTarget:
    def __init__(self, shipyard: Shipyard):
        self.shipyard = shipyard
        self.point = shipyard.point
        self.expected_profit = self._estimate_profit()
        self.reinforcement_distance = self._get_reinforcement_distance()
        self._future_ship_count = self._estimate_future_ship_count()
        self.total_incoming_power = self._get_total_incoming_power()

    def __repr__(self):
        return f"Target {self.shipyard}"

    def estimate_shipyard_power(self, time):
        return self._future_ship_count[time]

    def _get_total_incoming_power(self):
        return sum(x.ship_count for x in self.shipyard.incoming_allied_fleets)

    def _get_reinforcement_distance(self):
        incoming_allied_fleets = self.shipyard.incoming_allied_fleets
        if not incoming_allied_fleets:
            return np.inf
        return min(x.eta for x in incoming_allied_fleets)

    def _estimate_profit(self):
        board = self.shipyard.board
        spawn_cost = board.spawn_cost
        profit = sum(
            2 * x.expected_kore() - x.ship_count * spawn_cost
            for x in self.shipyard.incoming_allied_fleets
        )
        profit += spawn_cost * board.shipyard_cost
        return profit

    def _estimate_future_ship_count(self):
        shipyard = self.shipyard
        player = shipyard.player
        board = shipyard.board

        time_to_fleet_kore = defaultdict(int)
        for sh in player.shipyards:
            for f in sh.incoming_allied_fleets:
                time_to_fleet_kore[len(f.route)] += f.expected_kore()

        shipyard_reinforcements = defaultdict(int)
        for f in shipyard.incoming_allied_fleets:
            shipyard_reinforcements[len(f.route)] += f.ship_count

        spawn_cost = board.spawn_cost
        player_kore = player.kore
        ship_count = shipyard.ship_count
        future_ship_count = [ship_count]
        for t in range(1, board.size + 1):
            ship_count += shipyard_reinforcements[t]
            player_kore += time_to_fleet_kore[t]

            can_spawn = max_ships_to_spawn(shipyard.turns_controlled + t)
            spawn_count = min(int(player_kore // spawn_cost), can_spawn)
            player_kore -= spawn_count * spawn_cost
            ship_count += spawn_count
            future_ship_count.append(ship_count)

        return future_ship_count


def capture_shipyards(agent: Player, max_attack_distance=10):
    board = agent.board
    agent_shipyards = [
        x for x in agent.shipyards if x.available_ship_count >= 3 and not x.action
    ]
    if not agent_shipyards:
        return

    targets = []
    for op_sy in board.shipyards:
        if op_sy.player_id == agent.game_id or op_sy.incoming_hostile_fleets:
            continue
        target = _ShipyardTarget(op_sy)
        # if target.expected_profit > 0:
        targets.append(target)

    if not targets:
        return

    for t in targets:
        shipyards = sorted(
            agent_shipyards, key=lambda x: t.point.distance_from(x.point)
        )

        for sy in shipyards:
            if sy.action:
                continue

            distance = sy.point.distance_from(t.point)
            if distance > max_attack_distance:
                continue

            power = t.estimate_shipyard_power(distance)

            if sy.available_ship_count <= power:
                continue

            num_ships_to_launch = min(sy.available_ship_count, int(power * 1.2))

            routes = find_shortcut_routes(
                board,
                sy.point,
                t.point,
                agent,
                num_ships_to_launch,
            )
            if routes:
                route = random.choice(routes)
                logger.info(
                    f"Attack shipyard {sy.point}->{t.point}"
                )
                sy.action = Launch(num_ships_to_launch, route)
                break
