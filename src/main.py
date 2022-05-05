from hashlib import new
from xmlrpc.client import Boolean
from kore_fleets import balanced_agent, random_agent
from logger import logger, init_logger
from helpers import *
from helpers import Observation, Point, Direction, ShipyardAction, Configuration, Player, Shipyard, ShipyardId, Board, Fleet
from random import gauss, randint
import math
import time
from copy import deepcopy
from typing import List, Dict, Tuple, Union


class Timer:
    def __init__(self, func):
        self.function = func
        self.inc = 0
        self.timer = 0

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.function(*args, **kwargs)
        end_time = time.time()
        self.inc += 1
        self.timer += end_time-start_time
        if self.inc == 10000:
            print("10000 Executions of {} took {} seconds".format(self.function.__name__,self.timer))
            self.inc = 0
            self.timer = 0
        
        return result





def gauss_int(mini,maxi,mu, sigma):
    return min(max(round(random.gauss(mu, sigma)), mini),maxi)

# return a tuple with the distance to and the closest point of the liste
def distance_to_closest(point : Point, liste : List[Point],size : int):
    res : Point
    distance : int
    distance = 10000
    res = point
    for p in liste:
        if point.distance_to(p,size) < distance:
            distance = point.distance_to(p,size)
            res = p
    return (distance,res)


def list_ops_shipyards_position(board: Board):
    ops = board.opponents
    ops_shipyards = []
    for op in ops:
        for op_shipyard in op.shipyards:
            ops_shipyards.append(op_shipyard.position)
    return ops_shipyards

# return possible ways to intercept a fleet (the 3 fastests) < 8 turns before interception
def intercept_points(me : Player, boards : List[Board], fleet: Fleet, shipyard: Shipyard):
    res : List[Tuple[int,Point]]
    fleetId = fleet.id
    board = boards[0]
    if not fleetId in board.fleets:
        return []
    fleetPoint = board.fleets[fleetId].position
    size = board.configuration.size
    ops_shipyards = list_ops_shipyards_position(board)
    i = 0
    res = []
    while i < 9:
        while (fleetPoint.distance_to(shipyard.position,size) != i or distance_to_closest(fleetPoint, ops_shipyards, size)[0] <= i) and i < 9:
            i += 1
            board = boards[i]
            if not fleetId in board.fleets:
                return res
            new_fleet = board.fleets[fleetId]
            fleetPoint = new_fleet.position
        if i >= 9:
            return res[:2]
        res.append((i,fleetPoint))
        i += 1
        board = boards[i]
        if not fleetId in board.fleets:
            return res[:2]
        new_fleet = board.fleets[fleetId]
        fleetPoint = new_fleet.position
    return res[:2]

def reward_fleet_opti_slow(board: Board, shipyard_id : int, duration : int, action_plan : str):
    new_board = deepcopy(board)
    obs = new_board.observation
    config = new_board.configuration
    size = config.size
    me = new_board.current_player
    action = ShipyardAction.launch_fleet_with_flight_plan(21, action_plan)
    if shipyard_id not in me.shipyard_ids:
        return 0
    shipyard = board.shipyards[shipyard_id]
    s = shipyard.position
    shipyard.next_action = action
    boards = predicts_next_boards(obs,config,n=duration-1,my_first_acctions=me.next_actions)
    
    first_dir = Direction.from_char(action_plan[0])
    fleet_point = s.translate(first_dir.to_point(),size)
    fleet = boards[1].get_fleet_at_point(fleet_point)
    if fleet == None:
        return 0
    fleet_number = fleet.ship_count
    fleet_id = fleet.id
    if fleet_id not in boards[-1].fleets:
        return 0
    new_fleet = boards[-1].fleets[fleet_id]
    
    return new_fleet.kore-config.spawn_cost*(fleet_number-fleet.ship_count)

def reward_fleet(board: Board, shipyard_id : int, duration : int, action_plan : str):
    me = board.current_player
    if shipyard_id not in me.shipyard_ids:
        return 0
    pos = board.shipyards[shipyard_id].position
    pseudo_fleet = Pseudo_Fleet(action_plan,None,pos)
    size = board.configuration.size
    next_pos = next_n_positions(pseudo_fleet,size,duration)
    return fast_kore_calcul(next_pos, board)

def translate_n(p : Point, dir: Direction, times: int, size: int) -> Point:
    if times > 0:
        new_p = p
        for i in range(times):
            new_p = new_p.translate(dir.to_point(),size)
        return new_p
    elif times < 0:
        return translate_n(p, dir.opposite(),-times,size)
    return p

def translate_n_m(p : Point, dir1: Direction, n: int, dir2: Direction, m: int, size: int) -> Point:
    new_p = translate_n(p,dir1,n,size)
    return translate_n(new_p,dir2,m,size)
    


def best_action2(liste_action):
    if liste_action == []:
        return (0,"")
    else:
        return max(liste_action)

def best_action(liste_action):
    if liste_action == []:
        return (0,"")
    else:
        return liste_action[0]

def best_action_multiple_fleet(board: Board, dict_action : Dict[FleetId,List[Tuple[float,str]]])-> Tuple[str,int]:
    dict_plan : Dict[Tuple[str,int]]
    def maximum(dictionnaire):
        res = (0,0)
        best_plan = ""
        for p,x in dictionnaire.items():
            if x > res:
                res = x
                best_plan = p
        return best_plan,res[1]
    dict_plan = dict()
    for (fleetid,actions) in dict_action.items():
        dict_plan_fleet : Dict[Tuple[str,int]]
        dict_plan_fleet = dict()
        for action in actions:
            (reward,plan) = action
            dict_plan_fleet[plan] = (reward,board.fleets[fleetid].ship_count)
        for (plan,detail) in dict_plan_fleet.items():
            if plan in dict_plan:
                (reward,nb_ships) = detail
                (last_reward,last_nb_ships) = dict_plan[plan]
                dict_plan[plan] = (reward+ last_reward, nb_ships + last_nb_ships)
            else:
                dict_plan[plan] = detail
    if dict_plan == dict():
        return ("",0)
    return maximum(dict_plan)
            

def plan_secure_route_home_21(me : Player, boards : List[Board], fleet: Fleet, shipyard: Shipyard, intercept_p : Tuple[int,Point]) -> List[Tuple[float,str]]:
    plans = plan_route_home_21(me,boards,fleet,shipyard,intercept_p)
    i,p = intercept_p
    board = boards[0]
    size = board.configuration.size
    pos = shipyard.position
    spawn_cost = board.configuration.spawn_cost
    nb_ships = fleet.ship_count + 1
    res = []
    for plan in plans:
        plan2 = decompose(plan)
        reward,length = evaluate_plan(boards,plan2,pos,nb_ships,spawn_cost,size)
        if reward > 0:
            if length >= i:
                res.append((reward,plan))
    return res

def plan_route_home_21(me : Player, boards : List[Board], fleet: Fleet, shipyard: Shipyard, intercept_p : Tuple[int,Point]) -> List[str]:
    (iter,p) = intercept_p
    s = shipyard.position
    board = boards[0]
    size = board.configuration.size
    
    def plan_square(dir : Direction,dir2 : Direction,travel_distance : int,travel_distance2 : int) -> str:
        if travel_distance2 != 0 and travel_distance != 0:
            return dir.to_char()+str(travel_distance-1)+dir2.to_char()+str(travel_distance2-1)+dir.opposite().to_char()+str(travel_distance-1)+dir2.opposite().to_char()
        if travel_distance == 0 and travel_distance2 != 0:
            return dir2.to_char()+str(travel_distance2-1)+dir2.opposite().to_char()
        if travel_distance2 == 0 and travel_distance != 0:
            return dir.to_char()+str(travel_distance-1)+dir.opposite().to_char()
        return ""
    
    def North_South(p: Point,s: Point,size) -> Tuple[int,Direction]:
        if p.y == s.y:
            return (0,Direction.SOUTH)
        if (p.y-s.y)%size<(s.y-p.y)%size:
            return ((p.y-s.y)%size,Direction.NORTH)
        return ((s.y-p.y)%size,Direction.SOUTH)
            
    def West_East(p,s,size):
        if p.x == s.x:
            return (0,Direction.WEST)
        if (p.x-s.x)%size<(s.x-p.x)%size:
            return ((p.x-s.x)%size,Direction.EAST)
        return ((s.x-p.x)%size,Direction.WEST)

    # First North-South
    travel_distance,dir = North_South(p,s,size)
    travel_distance2,dir2 = West_East(p,s,size)    
    next_pos = next_n_positions(fleet,size,travel_distance+travel_distance2)
    # augmenter progressivement la distance qui reste à parcourir derrière en calculant la reward avec predict et checker si on est pas proche du terrain enemie
    if travel_distance != 0 and travel_distance2 != 0:
        # rectangles 
        plan_to_try = []
        # first North South
        for _ in range(2):
            i = 0
            while travel_distance2+i < 10 and i < 7:
                travelled_distance = travel_distance+travel_distance2+i
                new_point = translate_n_m(s,dir,travel_distance,dir2,travel_distance2+i,size)
                if distance_to_closest(new_point, list_ops_shipyards_position(board), size)[0] > travelled_distance:
                    plan = plan_square(dir,dir2,travel_distance,travel_distance2+i)
                    plan_to_try.append(plan)
                else:
                    break
                i += 1
            dir,dir2,travel_distance,travel_distance2=dir2,dir,travel_distance2,travel_distance
        return plan_to_try
        """
        # first East West
        new_point = translate_n(p,dir2,travel_distance2)
        new_point = translate_n(new_point,dir,travel_distance)
        for i in range(8):
            travelled_distance = travel_distance+travel_distance2+i
            new_point = translate_n(new_point,dir,1)
            plan = dir2.to_char()+str(travel_distance2)+dir.to_char()+str(travel_distance+i)+dir2.opposite().to_char()+str(travel_distance2)+dir.opposite().to_char()
            reward = reward_fleet(board, shipyard.id, travel_distance*2+travel_distance2*2+2*i,plan)
            plan_to_try.append((reward,plan))
        """
    else:
        if travel_distance2 == 0:
            dir2,travel_distance2=dir,travel_distance
        if travel_distance2 == 0:
            return []
        plan_to_try = []
        i = 0
        while travel_distance2+i < 10 and i < 6:
            travelled_distance = travel_distance2+i
            new_point = translate_n(s,dir2,travel_distance2+i,size)
            if distance_to_closest(new_point, list_ops_shipyards_position(board), size)[0] > travelled_distance:
                dir = dir2.rotate_left()
                for k in range(2):
                    new_point2 = new_point
                    j = 0
                    while j < 8:
                        travelled_distance2 = travel_distance2+i+j
                        new_point = translate_n_m(s,dir,j,dir2,travel_distance2+i,size)
                        if distance_to_closest(new_point, list_ops_shipyards_position(board), size)[0] > travelled_distance2:
                            plan = plan_square(dir2,dir,travel_distance2+i,j)
                            plan_to_try.append(plan)
                            j += 1
                        else:
                            break
                    dir = dir.opposite()
            else:
                break
            i += 1
        return plan_to_try
        


def predicts_next_boards(obs,config,n=23,my_first_acctions=None,my_agent=balanced_agent,op_agent=balanced_agent):
    board = Board(obs, config)
    me = board.current_player
    new_observation = deepcopy(obs)
    boards = [board]
    ops = board.opponents
    for i in range(n):
        new_actions: List[Dict[str,str]]
        new_actions = [dict()]*len(new_observation['players'])
        if i == 0:
            if my_first_acctions != None:
                new_actions[me.id] = my_first_acctions
            else:
                new_actions[me.id] = my_agent(new_observation,config)
        for op in ops:
            new_obs = new_observation
            new_obs['player'] = op.id
            op_next_actions = op_agent(new_obs,config)
            new_actions[op.id] = op_next_actions
        
        new_observation = Board(new_observation, config, new_actions).next().observation
        boards.append(Board(new_observation, config))
    return boards

def decompose(fligtPlan : str) -> List[Union[str,int]]:
    res = []
    suite = ""
    for x in fligtPlan:
        if x in "SNWEC":
            if suite != "":
                if suite != "0":
                    res.append(int(suite))
                    suite = ""
            res.append(x)
        else:
            suite = suite + x
    return res

def compose(plan :List[Union[str,int]]) -> str:
    res = ""
    for i in range(len(plan)):
        res = res + str(plan[i])
    return res

def maximum_flight_plan_length(num_ships):
    return math.floor(2 * math.log(num_ships)) + 1

def minimum_ships_num(length_fligth_plan):
    dict_relation = {1:1,2:2,3:3,4:5,6:13,7:21,8:34,9:55,10:91,11:149,12:245,13:404}
    if 0<length_fligth_plan<14:
        return dict_relation[length_fligth_plan]
    return 10000

def next_position(fleet : Fleet, size):
    plan = decompose(fleet.flight_plan)
    dir = None
    if len(plan) == 0:
        dir = fleet.direction
    else:
        next_action = plan[0]
        if type(next_action) == int:
            dir = fleet.direction
        elif next_action != "C":
            dir = Direction.from_char(next_action)
        elif next_action == "C":
            return fleet.position
    new_pos = translate_n(fleet.position,dir,1,size)
    return new_pos

        

def endangered(boards: List[Board], fleet: Fleet):
    fleet_id = fleet.id
    size = boards[0].configuration.size
    i = 0
    while i < len(boards)-1 and fleet_id in boards[i].fleets:
        i += 1
    if i == 0:
        return False # The fleet died so it's not in danger
    if i >= len(boards)-1:
        return True
    p = next_position(boards[i-1].fleets[fleet_id],size)
    if boards[0].get_shipyard_at_point(p) != None:
        if boards[i].get_shipyard_at_point(p).player_id == boards[0].current_player_id:
            return False
    if boards[i].get_fleet_at_point(p) != None:
        if boards[i].get_fleet_at_point(p).player_id == boards[0].current_player_id:
            return endangered(boards,boards[i].get_fleet_at_point(p))
    return True

# ------------------------------ Next Pos prediction ------------------------------------------

class Pseudo_Fleet:
    def __init__(self,flight_plan:str,direction:Union[Direction,None],position: Point) -> None:
        self.flight_plan = flight_plan
        self.direction = direction
        self.position = position


def next_n_positions(fleet: Union[Fleet,Pseudo_Fleet], size: int, n: int) -> List[Point]:
    plan = decompose(fleet.flight_plan)
    dir = fleet.direction
    pos = fleet.position
    return next_n_positions_rec(pos,dir,plan,n,size,[pos])

def next_n_positions_rec(pos: Point,dir: Direction, plan: List[Union[str,int]], n: int, size, a_return):
    direction = dir
    if n == 0:
        return a_return
    if len(plan) == 0:
        pass
    else:
        next_action = plan[0]
        if type(next_action) == int:
            if next_action == 0:
                plan.pop(0)
                next_n_positions_rec(pos,dir,plan,n,size,a_return)
            elif next_action == 1:
                plan.pop(0)
            else:
                plan[0] -= 1
        elif next_action != "C":
            direction = Direction.from_char(next_action)
            plan.pop(0)
        else:
            return a_return
    new_pos = translate_n(pos,direction,1,size)
    return next_n_positions_rec(new_pos,direction,plan,n-1,size,a_return + [new_pos])

def add_to_all(liste: List[List],item) -> List[List]:
    res = []
    for i in range(len(liste)):
        res.append(liste[i] + [item])
    return res

# 0. ------- Generate all paths

def generate_all_looped_plans(nb_move : int, plan: List[List[Union[str,int]]]=[[]], last_dir : Union[str,None]=None, last_move : Boolean=True) -> List[List[Union[str,int]]]:
    current_dir : Direction
    if nb_move == 0:
        return plan
    res = []
    if last_move == False and nb_move != 1:
        for i in range(9):
            res += generate_all_looped_plans(nb_move-1, add_to_all(plan,i+1), last_dir, last_move=True)
    current_dir = Direction.NORTH
    if last_dir == None:
        res += generate_all_looped_plans(nb_move-1, add_to_all(plan,current_dir.to_char()), last_dir=current_dir.to_char(), last_move=False)
    else:
        current_dir = Direction.from_char(last_dir)
    for i in range(3):
        current_dir = current_dir.rotate_right()
        res += generate_all_looped_plans(nb_move-1, add_to_all(plan,current_dir.to_char()), last_dir=current_dir.to_char(), last_move=False)
        
    return res

def generate_all_plans_A_to_B(boards: List[Board],a: Point,b: Point,nb_moves: int):
    board = boards[0]
    size = board.configuration.size
    def North_South(p: Point,s: Point,size) -> Tuple[int,Direction]:
        if p.y == s.y:
            return (0,Direction.SOUTH)
        if (p.y-s.y)%size<(s.y-p.y)%size:
            return ((p.y-s.y)%size,Direction.NORTH)
        return ((s.y-p.y)%size,Direction.SOUTH)
            
    def West_East(p,s,size):
        if p.x == s.x:
            return (0,Direction.WEST)
        if (p.x-s.x)%size<(s.x-p.x)%size:
            return ((p.x-s.x)%size,Direction.EAST)
        return ((s.x-p.x)%size,Direction.WEST)

    # First North-South
    travel_distance,dir = North_South(a,b,size)
    travel_distance2,dir2 = West_East(a,b,size)
     
    
# 1. -------- Kore Calcul

def fast_kore_calcul(next_pos: List[Point],board : Board):
    all_cells = board.cells
    tot_kore = 0
    for i in range(0,len(next_pos)-1):
        current_pos = next_pos[i+1]
        current_cell = all_cells[current_pos]
        tot_kore += current_cell.kore*(1.02**i)
    return tot_kore


def fast_precise_kore_calcul(next_pos: List[Point],boards : List[Board]):
    tot_kore = 0
    for i in range(0,len(next_pos)-1):
        current_pos = next_pos[i+1]
        id_boards = i+1
        if i+1>= len(boards):
            id_boards = -1
        current_cell = boards[id_boards].get_cell_at_point(current_pos)
        tot_kore += current_cell.kore
    return tot_kore

# 2. --------- Danger level


def danger_level(boards : List[Board], next_pos: List[Point], nb_ship : int):
    res = 0
    board = boards[0]
    size = board.configuration.size
    for i in range(0,len(next_pos)-1):
        if i+1 >= len(boards):
            return res
        current_pos = next_pos[i+1]
        liste_ops_shipyards_pos = list_ops_shipyards_position(board)
        distance_to_closest_shipyard,op_shipyard_pos = distance_to_closest(current_pos, liste_ops_shipyards_pos, size)
        while distance_to_closest_shipyard < i:
            shipyard_power = boards[i-distance_to_closest_shipyard].get_shipyard_at_point(op_shipyard_pos).ship_count
            if shipyard_power > nb_ship:
                return 1
            liste_ops_shipyards_pos.remove(op_shipyard_pos)
            distance_to_closest_shipyard,op_shipyard_pos = distance_to_closest(current_pos, liste_ops_shipyards_pos, size)
            res += 0.2
    return res
        

# 4. Easy to retrieve ?


def retrieve_cost(boards : List[Board], next_pos: List[Point], nb_ship: int):
    me = boards[0].current_player
    size = boards[0].configuration.size
    if len(next_pos) <= len(boards):
        last_board = boards[len(next_pos)-1]
        last_pos = next_pos[len(next_pos)-1]
        fleet = last_board.get_fleet_at_point(last_pos)
        if fleet != None and me.id == fleet.player_id and fleet.ship_count != nb_ship:
            return (0,len(next_pos))
        shipyard = last_board.get_shipyard_at_point(last_pos)
        if shipyard != None and me.id == shipyard.player_id:
            return (0,len(next_pos))
    for i in range(len(next_pos)):
        dist = distance_to_closest(next_pos[len(next_pos)-1-i], [shipyard.position for shipyard in me.shipyards], size)[0]
        if dist <= len(next_pos)-1-i:
            return (nb_ship + 1 + dist//2,len(next_pos)-1-i)
    


# 3. --------- Length


def duration_till_stop(boards : List[Board], next_pos: List[Point],nb_ship: int):
    me = boards[0].current_player
    size = boards[0].configuration.size
    nb_ships_left = nb_ship
    for i in range(0,len(next_pos)-1):
        if i+1 >= len(boards):
            return i+1,nb_ships_left
        current_pos = next_pos[i+1]
        shipyard = boards[i+1].get_shipyard_at_point(current_pos)
        if shipyard != None:
            if shipyard.player_id == me.id:
                return i+1,nb_ships_left
            else:
                return i+1,nb_ships_left-shipyard.ship_count
        fleet = boards[i+1].get_fleet_at_point(current_pos)
        if fleet != None:
            if fleet.player_id == me.id:
                if fleet.ship_count > nb_ships_left:
                    return i+1,nb_ships_left
                else:
                    nb_ships_left += fleet.ship_count
            else:
                if fleet.ship_count > nb_ships_left:
                    return i+1,nb_ships_left-fleet.ship_count
                else:
                    nb_ships_left = nb_ships_left-fleet.ship_count
        """
        for i in range(4):
            p = translate_n(current_pos,Direction.from_index(i),1,size)
            fleet = boards[i+1].get_fleet_at_point(p)
            if fleet != None and fleet.player_id != me.id and fleet.ship_count >= nb_ship:
                return i+1
        """
    return len(next_pos)-1,nb_ships_left

# ------------------- Global result

def evaluate_plan(boards: List[Board],plan:List[Union[Direction,int]],pos,nb_ships,spawn_cost,size):
    action_plan = compose(plan)
    pseudo_fleet = Pseudo_Fleet(action_plan,None,pos)
    next_pos = next_n_positions(pseudo_fleet,size,30)
    duration,nb_ships_left = duration_till_stop(boards, next_pos,nb_ships)
    next_pos = next_pos[:duration+1]
    retrieve_costs,real_duration = retrieve_cost(boards, next_pos,nb_ships)
    next_pos = next_pos[:real_duration+1]
    kore = fast_precise_kore_calcul(next_pos,boards)* math.log(nb_ships) / 20
    danger = danger_level(boards,next_pos,nb_ships)
    return kore-danger*danger*500-retrieve_costs*spawn_cost*0.2-real_duration-(nb_ships_left<=0)*10000,real_duration

def best_fleet_overall(boards : List[Board], shipyard_id : ShipyardId):
    board = boards[0]
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    me = board.current_player
    if shipyard_id not in me.shipyard_ids:
        return 0
    pos = board.shipyards[shipyard_id].position
    all_possible_plans = generate_all_looped_plans(5)
    nb_ships = 8
    best_plan = []
    best_global_reward = 0
    for plan in all_possible_plans:
        global_reward = evaluate_plan(boards,plan,pos,nb_ships,spawn_cost,size)[0]
        if global_reward > best_global_reward:
            best_plan = plan
            best_global_reward = global_reward
    logger.info(f"{best_global_reward,best_plan}")
    return best_plan


def safe_plan_to_pos(boards: List[Board],shipyard_id: ShipyardId,pos: Point,ship_number_to_send: int):
    board = boards[0]
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    me = board.current_player
    if shipyard_id not in me.shipyard_ids:
        return 0
    shipyard_pos = board.shipyards[shipyard_id].position
    nb_moves = maximum_flight_plan_length(ship_number_to_send)
    all_possible_plans = generate_all_plans_A_to_B(boards,shipyard_pos,pos,nb_moves-1)
    nb_ships = 8
    best_plan = []
    best_global_reward = 0
    for plan in all_possible_plans:
        global_reward = evaluate_plan(boards,plan,pos,nb_ships,spawn_cost,size)[0]
        if global_reward > best_global_reward:
            best_plan = plan
            best_global_reward = global_reward
    logger.info(f"{best_global_reward,best_plan}")


def concat(a : List[List[Any]]) -> List[Any]:
    res = []
    for x in a:
        for y in x:
            res.append(y)
    return res

# Find best way for the 8 or below : check for n pos if it is dangerous + cell have opposant ships/shipyards on them
# A good way can have already a 21 intercepting it
# see if the 21 or below can intercept 2-3-4 small fleets

def get_shortest_flight_path_between(position_a, position_b, size, trailing_digits=False):
    mag_x = 1 if position_b.x > position_a.x else -1
    abs_x = abs(position_b.x - position_a.x)
    dir_x = mag_x if abs_x < size/2 else -mag_x
    mag_y = 1 if position_b.y > position_a.y else -1
    abs_y = abs(position_b.y - position_a.y)
    dir_y = mag_y if abs_y < size/2 else -mag_y
    flight_path_x = ""
    if abs_x > 0:
        flight_path_x += "E" if dir_x == 1 else "W"
        flight_path_x += str(abs_x - 1) if (abs_x - 1) > 0 else ""
    flight_path_y = ""
    if abs_y > 0:
        flight_path_y += "N" if dir_y == 1 else "S"
        flight_path_y += str(abs_y - 1) if (abs_y - 1) > 0 else ""
    if not len(flight_path_x) == len(flight_path_y):
        if len(flight_path_x) < len(flight_path_y):
            return flight_path_x + (flight_path_y if trailing_digits else flight_path_y[0])
        else:
            return flight_path_y + (flight_path_x if trailing_digits else flight_path_x[0])
    return flight_path_y + (flight_path_x if trailing_digits or not flight_path_x else flight_path_x[0]) if random() < .5 else flight_path_x + (flight_path_y if trailing_digits or not flight_path_y else flight_path_y[0])

def check_location(board, loc, me):
    if board.cells.get(loc).shipyard and board.cells.get(loc).shipyard.player.id == me.id:
        return 0
    kore = 0
    for i in range(-3, 4):
        for j in range(-3, 4):
            pos = loc.translate(Point(i, j), board.configuration.size)
            kore += board.cells.get(pos).kore or 0
    return kore

# Go protect the shipyard if in danger
def indanger_shipyards(boards : List[Board])-> Tuple[Shipyard,int]:
    pass

def matrix_kore(board : Board,matrix,p,size):
    middle = (len(matrix)//2,len(matrix[0])//2)
    res = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            val = matrix[i][j]
            if val != 0:
                newp = translate_n_m(p, Direction.North,i-middle[0],Direction.WEST,j-middle[1],size)
                res += val*board.get_cell_at_point(newp).kore
    return res


# find a good place for the shipyard
def best_pos_shipyard(boards: List[Board], shipyard_id: ShipyardId) -> Union[Point,None]:
    matrix = [[0,0,0,4,0,0,0],
              [0,0,3,6,3,0,0],
              [0,3,5,10,5,3,0],
              [4,6,10,-20,10,6,4],
              [0,3,5,10,5,3,0],
              [0,0,3,6,3,0,0],
              [0,0,0,4,0,0,0]]

    board = boards[0]
    size = board.configuration.size
    if not shipyard_id in board.shipyards:
        return None
    s = board.shipyards[shipyard_id].position
    liste_ops_shipyards_pos = list_ops_shipyards_position(board)
    best_pos = None
    best_kore = 0
    for i in range(9):
        for j in range(9):
            if 14 >= i+j >= 3:
                for k in range(4):
                    time_board = boards[-1]
                    if i+j < len(boards):
                        time_board = boards[i+j]
                    possible_pos = translate_n_m(s,Direction.from_index(k),i,Direction.from_index(k).rotate_right(),j,size)
                    distance_to_closest_ops_shipyard,_ = distance_to_closest(possible_pos, liste_ops_shipyards_pos, size)
                    if distance_to_closest_ops_shipyard <= i+j:
                        potential_kore = matrix_kore(time_board,matrix,possible_pos,size)
                        if potential_kore > best_kore:
                            best_kore = potential_kore
                            best_pos = possible_pos
    return best_pos
    
    

def agent(obs: Observation, config: Configuration):
    new_observation : Observation
    if obs.step == 0:
        init_logger(logger)

    board = Board(obs, config)
    step = board.step
    my_id = obs["player"]
    remaining_time = obs["remainingOverageTime"]
    logger.info(f"<step_{step + 1}>, remaining_time={remaining_time:.1f}")
    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore
    ops = board.opponents
    size = board.configuration.size
    
    # --------------------------------- Predict the next Boards ---------------------------------
    boards = predicts_next_boards(obs,config)
    logger.info("boards generated")
    # --------------------------------- Choose a move ---------------------------------
    
    #i_shipyards,ship_power = indanger_shipyards(boards,)
    
    for shipyard in me.shipyards:
        action = None
        nb_ships = shipyard.ship_count
        """
        if turn == 6:
            if nb_ships>=8:
                if random.random() < 0.5:
                    action = ShipyardAction.launch_fleet_with_flight_plan(8, "E3S4W")
                else:
                    action = ShipyardAction.launch_fleet_with_flight_plan(8, "N3E4S")
        """
        
        no_endangered = True
        if turn >= 8:
            if nb_ships >= 21:
                possible_actions = dict()
                nb_ships_to_send = 21
                for fleet in me.fleets:
                    if endangered(boards,fleet):
                        no_endangered = False
                        if fleet.ship_count < nb_ships:
                            liste_intercept = intercept_points(me,boards,fleet,shipyard)
                            possible_actions[fleet.id] = []
                            possible_actions2 = []
                            for (i,p) in liste_intercept:
                                possible_actions[fleet.id] += plan_secure_route_home_21(me,boards,fleet,shipyard,(i,p))
                            liste_intercept2 = intercept_points(me,boards[2:],fleet,shipyard)
                            for (i,p) in liste_intercept2:
                                possible_actions2 += plan_secure_route_home_21(me,boards[2:],fleet,shipyard,(i,p))
                            if possible_actions[fleet.id] != [] and possible_actions2 != []:
                                difference = max(possible_actions2)[0] - max(possible_actions[fleet.id])[0]
                                possible_actions[fleet.id] = [(rewards-difference,plans) for (rewards,plans) in possible_actions[fleet.id]]
                opponent_fleets = concat([op.fleets for op in ops])
                for fleet in opponent_fleets:
                    if fleet.ship_count < nb_ships:
                        liste_intercept = intercept_points(me,boards,fleet,shipyard)
                        possible_actions[fleet.id] = []
                        possible_actions2 = []
                        for (i,p) in liste_intercept:
                            possible_actions[fleet.id] += plan_secure_route_home_21(me,boards,fleet,shipyard,(i,p))
                        liste_intercept2 = intercept_points(me,boards[2:],fleet,shipyard)
                        for (i,p) in liste_intercept2:
                            possible_actions2 += plan_secure_route_home_21(me,boards[2:],fleet,shipyard,(i,p))
                        if possible_actions[fleet.id] != [] and possible_actions2 != []:
                            difference = max(possible_actions2)[0] - max(possible_actions[fleet.id])[0]
                            possible_actions[fleet.id] = [(rewards-difference,plans) for (rewards,plans) in possible_actions[fleet.id]]
                plan,nb_ships_mini = best_action_multiple_fleet(board,possible_actions)
                        
                    
                
                if plan != "":
                    logger.info(plan)
                    nb_ships_to_send = max(minimum_ships_num(len(plan)),nb_ships_mini+1)
                    action = ShipyardAction.launch_fleet_with_flight_plan(nb_ships_to_send, plan)
        if action == {}:
            action = None
        # ----- builder
        remaining_kore = me.kore
        convert_cost = board.configuration.convert_cost
        if action == None and remaining_kore > 300 and shipyard.max_spawn > 5:
            if shipyard.ship_count >= convert_cost + 20:
                pos = best_pos_shipyard(boards,shipyard.id)
                ship_number_to_send = max(convert_cost + 20, int(shipyard.ship_count/2))
                plan = safe_plan_to_pos(boards,shipyard.id,pos,ship_number_to_send) + "C"
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(ship_number_to_send, plan)
                action = shipyard.next_action
        # ---- invader
        if action == {}:
            action = None
        if action == None and shipyard.ship_count >= 100 and remaining_kore > 500:
            dist,p_op_shipyard = distance_to_closest(shipyard.position, list_ops_shipyards_position(board), size)
            if dist < len(boards):
                if board.get_shipyard_at_point(p_op_shipyard).ship_count < 100:
                    path = get_shortest_flight_path_between(shipyard.position,p_op_shipyard,size)
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(board.get_shipyard_at_point(p_op_shipyard).ship_count + 23,path)
                    action = shipyard.next_action
        if action == {}:
            action = None
        if action == None:
            if nb_ships >= 29: #+ 21*(1-no_endangered):
                plan = compose(best_fleet_overall(boards,shipyard.id))
                if plan != "":
                    action = ShipyardAction.launch_fleet_with_flight_plan(8, plan)
        if action == None:
            if kore_left >= spawn_cost:
                action = ShipyardAction.spawn_ships(int(min(shipyard.max_spawn,kore_left/spawn_cost)))
        shipyard.next_action = action
        """
        if turn < 20:
            pass
        elif turn % period == 1 and nb_ships >= 34:
            action = ShipyardAction.launch_fleet_with_flight_plan(21, "E4N6W4S")
        elif turn % period == 3 and board.fleets.values: 
            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3N")
        elif turn % period == 5: 
            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2N")
        elif turn % period == 7: 
            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E1N")
        elif turn % period == 9: 
            action = ShipyardAction.launch_fleet_with_flight_plan(2, "EN")
        elif turn % period == 11: 
            action = ShipyardAction.launch_fleet_with_flight_plan(2, "N")
        
        """
    logger.info(me.next_actions)
    return me.next_actions



from kaggle_environments import make
from Best_agent_from_beta.main import agent2

env = make("kore_fleets", debug=True)
print(env.name, env.version)

env.run([agent,balanced_agent])
