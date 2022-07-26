from xmlrpc.client import Boolean
from kore_fleets import balanced_agent, random_agent, do_nothing_agent
from logger import logger, init_logger
from helpers import *
from helpers import Observation, Point, Direction, ShipyardAction, Configuration, Player, Shipyard, ShipyardId, Fleet, FleetId, Board
import random
import math
import time
from copy import deepcopy
from typing import List, Dict, Tuple, Union, Any
import matplotlib.pyplot as plt

external_timer = time.time()
turn = 0
try_number = 0
boards = []
saved_gravity2 = [[[0 for x in range(21)] for y in range(21)] for i in range(40)]
building_ships_send : List['Point4d']
building_ships_send = []
size = 0
perso_kore = 500

class Timer:
    def __init__(self, func):
        self.function = func
        self.inc = 0
        self.timer = 0
        self.a_print = False

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.function(*args, **kwargs)
        end_time = time.time()
        self.inc += 1
        self.timer += end_time-start_time
        """
        if self.inc == 10000:
            print("10000 Executions of {} took {} seconds".format(self.function.__name__,self.timer))
            self.inc = 0
            self.timer = 0
        """
        if 40 <= int(external_timer-time.time())%50:
            self.a_print = True

        if self.a_print == True and int(external_timer-time.time())%50<10:
            if self.timer > 0.0001:
                print(f"Purcentage of use of {self.function.__name__} : {self.timer:.3g} %")
            self.a_print = False
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

def tot_ship_count(player: Player,board: Board) -> int:
    sum = 0
    for fleet in player.fleets:
        sum += fleet.ship_count
        plan = fleet.flight_plan
        if plan != "" and plan[-1] == "C":
            sum -= 50
    for shipyard in player.shipyards:
        sum += shipyard.ship_count
    return sum

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
    

def tab_kore(board: Board):
    size = board.configuration.size
    res = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            pos = Point(size - 1 - j, i)
            res[i][j] = board.get_cell_at_point(pos).kore

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
        reward,length,_ = evaluate_plan(boards,plan2,pos,nb_ships,spawn_cost,size)
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
    
    def North_South(s: Point,p: Point,size) -> Tuple[int,Direction]:
        if p.y == s.y:
            return (0,Direction.SOUTH)
        if (p.y-s.y)%size<(s.y-p.y)%size:
            return ((p.y-s.y)%size,Direction.NORTH)
        return ((s.y-p.y)%size,Direction.SOUTH)
            
    def West_East(s: Point,p: Point,size): # /!\ potential swap between s and p
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
        
def spawn_agent(obs: Observation,config: Configuration):
    board = Board(obs,config)
    me = board.current_player
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore
    shipyards = me.shipyards
    for shipyard in shipyards:
        action = None
        if kore_left >= spawn_cost:
                action = ShipyardAction.spawn_ships(int(min(shipyard.max_spawn,kore_left/spawn_cost)))
        shipyard.next_action = action
    return me.next_actions

def spawn_agent2(obs: Observation,config: Configuration):
    board = Board(obs,config)
    me = board.current_player
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore
    shipyards = me.shipyards
    for shipyard in shipyards:
        action = None
        if kore_left >= spawn_cost+10000:
                action = ShipyardAction.spawn_ships(int(min(shipyard.max_spawn,(kore_left-10000)/spawn_cost)))
        shipyard.next_action = action
    return me.next_actions

@Timer
def predicts_next_boards(obs: Observation,config: Observation,n=30,my_first_acctions=None,my_agent=spawn_agent2,op_agent=spawn_agent):
    board = Board(obs, config)
    me = board.current_player
    boards = [board]
    ops = board.opponents
    new_observation = deepcopy(obs)
    for player in new_observation.players:
        player[0] = 10000
    new_configuration = deepcopy(config)
    for i in range(n):
        new_actions: List[Dict[str,str]]
        new_actions = [dict()]*len(new_observation['players'])
        new_obs = deepcopy(new_observation)
        new_config = deepcopy(new_configuration)
        if i == 0:
            if my_first_acctions != None:
                new_actions[me.id] = my_first_acctions
            else:
                new_actions[me.id] = my_agent(new_obs,new_config)
        else:
            new_obs['player'] = me.id
            new_actions[me.id] = my_agent(new_obs,new_config)
        for op in ops:
            new_obs['player'] = op.id
            op_next_actions = op_agent(new_obs,new_config)
            new_actions[op.id] = op_next_actions
        
        new_board = Board(new_observation, new_configuration, new_actions).next()
        new_observation = new_board.observation
        new_configuration = new_board.configuration
        boards.append(new_board)
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
    if suite != "":
        if suite != "0":
            res.append(int(suite))
    return res

def compose(plan :List[Union[str,int]]) -> str:
    res = ""
    for i in range(len(plan)):
        res = res + str(plan[i])
    return res

def maximum_flight_plan_length(num_ships):
    return math.floor(2 * math.log(num_ships)) + 1

def minimum_ships_num(length_fligth_plan):
    dict_relation = {1:1,2:2,3:3,4:5,5:8,6:13,7:21,8:34,9:55,10:91,11:149,12:245,13:404}
    if 0<length_fligth_plan<14:
        return dict_relation[length_fligth_plan]
    return 10000

def ratio_kore_ship_count(num_ships):
    dict_relation = {1:0*0.01, 2:3*0.01, 3:5*0.01, 5:8*0.01, 8:10*0.01, 13:13*0.01, 21:15*0.01, 34:18*0.01, 55:20*0.01, 91:23*0.01, 149:25*0.01, 245:28*0.01, 404:30*0.01}
    num = minimum_ships_num(maximum_flight_plan_length(num_ships))
    if num != 10000:
        return dict_relation[num]
    if num_ships > 0:
        return 0.3
    return 0
    

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


# ------------------------------ Next Pos prediction ------------------------------------------

class Pseudo_Fleet:
    def __init__(self,flight_plan:str,direction:Union[Direction,None],position: Point) -> None:
        self.flight_plan = flight_plan
        self.direction = direction
        self.position = position


@Timer
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

@Timer
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

def generate_all_plans_A_to_B(boards: List[Board],a: Point,b: Point,nb_moves: int, need_precise_end=False):
    board = boards[0]
    size = board.configuration.size
    def North_South(s: Point,p: Point,size) -> Tuple[int,Direction]:
        if p.y == s.y:
            return (0,Direction.SOUTH)
        if (p.y-s.y)%size<(s.y-p.y)%size:
            return ((p.y-s.y)%size,Direction.NORTH)
        return ((s.y-p.y)%size,Direction.SOUTH)
            
    def West_East(s: Point,p: Point,size):
        if p.x == s.x:
            return (0,Direction.WEST)
        if (p.x-s.x)%size<(s.x-p.x)%size:
            return ((p.x-s.x)%size,Direction.EAST)
        return ((s.x-p.x)%size,Direction.WEST)

    def generate_all_plans_in_2_directions(nb_moves: int, t1 : int, dir1: Direction, t2: int, dir2: Direction, plan: List[List[Union[str,int]]]=[[]], last_move : Boolean=True) -> List[List[Union[str,int]]]:
        if t1 == 0 and  t2 == 0:
            return plan
        if nb_moves == 0:
            return []
        res = []
        if last_move == False:
            for i in range(9):
                if t1>=i+1:
                    res += generate_all_plans_in_2_directions(nb_moves-1, t1-i-1,dir1,t2,dir2, add_to_all(plan,i+1), last_move=True)
        res += generate_all_plans_in_2_directions(nb_moves-1, t2-1,dir2,t1,dir1, add_to_all(plan,dir2.to_char()), last_move=False)            
        return res
    
    # First North-South
    travel_distance,dir = North_South(a,b,size)
    travel_distance2,dir2 = West_East(a,b,size)
    return generate_all_plans_in_2_directions(nb_moves,travel_distance,dir,travel_distance2,dir2) + generate_all_plans_in_2_directions(nb_moves,travel_distance2,dir2,travel_distance,dir)


# 1. -------- Kore Calcul

def fast_kore_calcul(next_pos: List[Point],board : Board):
    all_cells = board.cells
    tot_kore = 0
    for i in range(0,len(next_pos)-1):
        current_pos = next_pos[i+1]
        current_cell = all_cells[current_pos]
        tot_kore += current_cell.kore*(1.02**i)
    return tot_kore


@Timer
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


@Timer
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


@Timer
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
    return (50,len(next_pos))
    


# 3. --------- Length


@Timer
def duration_till_stop(boards : List[Board], next_pos: List[Point],nb_ship: int):
    me = boards[0].current_player
    size = boards[0].configuration.size
    nb_ships_left = nb_ship
    for i in range(0,len(next_pos)-1):
        if i+1 >= len(boards):
            return i+1,nb_ships_left
        current_pos = next_pos[i]
        shipyard = boards[i+1].get_shipyard_at_point(current_pos)
        if shipyard != None:
            if shipyard.player_id == me.id:
                return i+1,nb_ships_left
            else:
                return i+1,nb_ships_left
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
        for j in range(4):
            p = translate_n(current_pos,Direction.from_index(j),1,size)
            fleet = boards[i+1].get_fleet_at_point(p)
            if fleet != None and fleet.player_id != me.id:
                if fleet.ship_count >= nb_ship:
                    return i,nb_ship-fleet.ship_count
                else:
                    nb_ships_left -= fleet.ship_count

    return len(next_pos)-1,nb_ships_left

# ------------------- Global result

@Timer
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
    return kore-danger*danger*500-retrieve_costs*spawn_cost*0.2-real_duration-(nb_ships_left<=0)*10000,real_duration,nb_ships_left


def safe_plan_to_pos(boards: List[Board],shipyard_pos: Point,pos: Point,ship_number_to_send: int,minimum_to_arrive: int):
    board = boards[0]
    size = board.configuration.size
    convert_cost = board.configuration.convert_cost
    spawn_cost = board.configuration.spawn_cost
    me = board.current_player
    nb_moves = maximum_flight_plan_length(ship_number_to_send)-(minimum_to_arrive>=50)
    all_possible_plans = generate_all_plans_A_to_B(boards,shipyard_pos,pos,nb_moves)
    nb_ships = ship_number_to_send
    best_plan = []
    best_global_reward = -1000000
    max_to_arrive = minimum_to_arrive
    for plan in all_possible_plans:
        global_reward,_,nb_ships_left = evaluate_plan(boards,plan,pos,nb_ships,spawn_cost,size)
        if nb_ships_left >= max_to_arrive and global_reward > best_global_reward:
            best_plan = plan
            max_to_arrive = nb_ships_left
            best_global_reward = global_reward
            logger.info(f"found {best_plan} with {best_global_reward} points")
    return best_plan


def concat(a : List[List[Any]]) -> List[Any]:
    res = []
    for x in a:
        for y in x:
            res.append(y)
    return res

def matrix_kore(board : Board,matrix,p,size):
    middle = (len(matrix)//2,len(matrix[0])//2)
    res = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            val = matrix[i][j]
            if val != 0:
                newp = translate_n_m(p, Direction.NORTH,i-middle[0],Direction.WEST,j-middle[1],size)
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
    me = board.current_player
    if not shipyard_id in board.shipyards:
        return None
    s = board.shipyards[shipyard_id].position
    best_pos = None
    best_kore = 0
    for i in range(2,7):
        for j in range(2,7):
            if 10 >= i+j >= 3:
                for k in range(4):
                    time_board = boards[-1]
                    if i+j < len(boards):
                        time_board = boards[i+j]
                    liste_ops_shipyards_pos = list_ops_shipyards_position(time_board)
                    time_me = time_board.current_player
                    liste_my_shipyards_pos = [shipyard.position for shipyard in time_me.shipyards]
                    possible_pos = translate_n_m(s,Direction.from_index(k),i,Direction.from_index(k).rotate_right(),j,size)
                    distance_to_closest_ops_shipyard,_ = distance_to_closest(possible_pos, liste_ops_shipyards_pos, size)
                    distance_to_my_closest_shipyard,_ = distance_to_closest(possible_pos,liste_my_shipyards_pos,size)
                    if distance_to_closest_ops_shipyard >= i+j and 5>= distance_to_my_closest_shipyard >= 3 and distance_to_my_closest_shipyard<distance_to_closest_ops_shipyard-2:
                        potential_kore = matrix_kore(time_board,matrix,possible_pos,size)
                        if potential_kore > best_kore:
                            best_kore = potential_kore
                            best_pos = possible_pos
    return best_pos

class Slice:
    def __init__(self, board: Board) -> None:
        self.gravity : Dict[Point,float]
        self.avg_gravity : float
        self.gravity = dict()
        self.danger_zone : Dict[Point,int]
        self.danger_zone = dict()
        self.board = board
        self.me = board.current_player
        self.size = board.configuration.size
        self.danger_zone2 : List[List[int]]
        self.danger_zone2 = [[0 for x in range(self.size)] for y in range(self.size)]
        self.avg_gravity = 0
        self.kore: Dict[Point,float]
        self.kore = dict()
        self.kore2: List[List[float]]
        self.kore2 = [[0 for x in range(self.size)] for y in range(self.size)]
        self.gravity2: List[List[float]]
        self.gravity2 = [[0 for x in range(self.size)] for y in range(self.size)]
        

    def add_gravity(self: 'Slice', position: Point, val: float) -> None:
        if position in self.gravity:
            self.gravity[position] += val
        else:
            self.gravity[position] = val

    def add_gravity2(self: 'Slice', position: Point, val: float) -> None:
        self.gravity2[position.x][position.y] += val
    
    def set_slice_gravity(self: 'Slice',slice_gravity:List[List[float]]):
        self.gravity2 = slice_gravity

    def multiply_gravity(self: 'Slice', position: Point, val: float) -> None:
        if position in self.gravity:
            self.gravity[position] *= val

    def multiply_gravity2(self: 'Slice', position: Point, val: float) -> None:
        self.gravity2[position.x][position.y] *= val

    def add_danger_level_at_point(self: 'Slice', position: Point, val : int) -> None:
        if position in self.danger_zone:
            res = self.danger_zone[position] + val
            if res > 0:
                self.danger_zone[position] = res
            else:
                self.danger_zone[position] = 0
        else:
            if val > 0:
                self.danger_zone[position] = val
    
    def add_danger_level2_at_point(self: 'Slice', position: Point, val : int) -> None:
        self.danger_zone2[position.x][position.y] += val
        
    def get_gravity_at_point(self: 'Slice', position: Point) -> float:
        if position in self.gravity:
            return self.gravity[position]
        else:
            return 0
    
    def get_gravity2_at_point(self: 'Slice', position: Point) -> float:
        return self.gravity2[position.x][position.y]
    
    def get_danger_level_at_point(self: 'Slice', position: Point) -> int:
        if position in self.danger_zone:
            return self.danger_zone[position]
        else:
            return 0
    
    def get_danger_level2_at_point(self: 'Slice', position: Point) -> int:
        return self.danger_zone2[position.x][position.y]
        
    def get_cell_at_point(self, position: Point):
        return self.board.get_cell_at_point(position)
    
    def get_fleet_at_point(self,position : Point):
        return self.board.get_fleet_at_point(position)
    
    def get_shipyard_at_point(self, position: Point):
        return self.board.get_shipyard_at_point(position)
    
    def get_potential_at_point(self,position: Point,nb_op_shipyard:int) -> float:
        cell = self.get_cell_at_point(position)
        if cell == None:
            return 0
        else:
            fleet_kore = 0
            fleet = cell.fleet
            if fleet != None and fleet.player_id != self.me.id:
                fleet_kore = reward_for_taking_an_ennemy_fleet(fleet.kore,fleet.ship_count)
            reward_shipyard = 0
            shipyard = cell.shipyard
            if shipyard != None and shipyard.player_id != self.me.id:
                reward_shipyard = reward_for_taking_a_shipyard(shipyard.max_spawn)*0.98**nb_op_shipyard
            cell_kore = cell.kore 
            if cell_kore >= 500:
                cell_kore += 100
            if 499 >= cell_kore >= 400:
                cell_kore = 400-(cell_kore-400)*2
            return cell_kore + fleet_kore + reward_shipyard
    
    def get_real_kore_at_point(self,position: Point):
        cell = self.get_cell_at_point(position)
        if cell == None:
            return 0
        else:
            return cell.kore
    
    def set_kore_at_point(self,position: Point,kore: float):
        self.kore[position] = kore
    
    def get_kore_at_point(self,position: Point):
        if position in self.kore:
            return self.kore[position]
        else:
            return 0
    
    def set_kore2_at_point(self,position: Point,kore: float):
        self.kore2[position.x][position.y] = kore
    
    def get_kore2_at_point(self,position: Point):
        return self.kore2[position.x][position.y]
    
    def calculate_avg_gravity(self,size):
        tot_sum = 0
        for i in range(size):
            for j in range(size):
                pos = Point(i,j)
                tot_sum += self.get_gravity_at_point(pos)
        self.avg_gravity = tot_sum / (size*size)
    


class Point4d:
    def __init__(self,point: Point, t : int, d: Union[Direction,None]) -> None:
        self._point = point
        self._t = t
        self._d = d
    
    @property
    def point(self):
        return self._point
    
    @property
    def t(self):
        return self._t
    
    @property
    def d(self):
        return self._d
    
    def move(self,d: Direction,size: int) -> 'Point4d':
        new_point = self._point.translate(d.to_point(),size)
        return Point4d(new_point,self._t+1,d)
    
    def wait(self) -> 'Point4d':
        return Point4d(self._point,self._t+1,self._d)
    
    def move_n_times(self, d: Direction, size: int, times: int):
        new_point = translate_n(self._point,d,times,size)
        return Point4d(new_point,self._t+times,d)
    
    def translate(self,d: Direction,size: int) -> 'Point4d':
        new_point = self._point.translate(d.to_point(),size)
        return Point4d(new_point,self._t,d)
    
    def next_pos(self, flightplan: str, size: int) -> 'Point4d':
        plan = decompose(flightplan)
        dir = None
        if len(plan) == 0:
            dir = self._d
        else:
            next_action = plan[0]
            if type(next_action) == int:
                dir = self._d
            elif next_action != "C":
                dir = Direction.from_char(next_action)
            elif next_action == "C":
                return self
        new_pos = self.move(dir,size)
        return new_pos
    
    def next_move_and_update(self, flightplan: str, size: int) -> 'Point4d':
        plan = decompose(flightplan)
        dir = None
        if len(plan) == 0:
            dir = self._d
        else:
            next_action = plan[0]
            if type(next_action) == int:
                dir = self._d
                if next_action == 1:
                    plan.pop(0)
            elif next_action != "C":
                dir = Direction.from_char(next_action)
                plan.pop(0)
            elif next_action == "C":
                return self,""
        new_pos = self.move(dir,size)
        return new_pos,compose(plan)
        
    


class Space:
    def __init__(self,boards: List[Board]) -> None:
        self.space : List[Slice]
        highway_points : List[Point4d]
        self.space = []
        for i in range(len(boards)):
            self.space.append(Slice(boards[i]))
        self.ghost_fleets : Dict[Point4d,Fleet]
        self.ghost_fleets = dict()
        self.highway_points = dict()
        self.variance = 1

    def __get__(self) -> List[Slice]:
        return self.space
    
    def __len__(self) -> int:
        return len(self.space)

    def __getitem__(self, ii) -> Union['Space',Slice]:
        """Get a list item"""
        if isinstance(ii, slice):
            return self.__class__(self.space[ii])
        else:
            return self.space[ii]

    def __delitem__(self, ii):
        """Delete an item"""
        del self.space[ii]

    def __setitem__(self, ii, val):
        # optional: self._acl_check(val)
        self._list[ii] = val

    def __str__(self):
        return str(self.space)

    def insert(self, ii, val):
        # optional: self._acl_check(val)
        self.space.insert(ii, val)

    def append(self, val):
        self.insert(len(self._list), val)
    def get_cell_at_point(self, point4d: Point4d):
        return self.space[point4d.t].get_cell_at_point(point4d.point)
    
    def get_fleet_at_point(self,point4d: Point4d):
        if point4d in self.ghost_fleets:
            return self.ghost_fleets[point4d]
        return self.space[point4d.t].get_fleet_at_point(point4d.point)

    def get_real_kore_at_point(self, point4d: Point4d):
        return self.space[point4d.t].get_real_kore_at_point(point4d.point)
        
    def get_shipyard_at_point(self, point4d: Point4d):
        return self.space[point4d.t].get_shipyard_at_point(point4d.point)
    
    def get_kore_at_point(self, point4d: Point4d):
        return self.space[point4d.t].get_kore_at_point(point4d.point)
    
    def set_kore_at_point(self, point4d: Point4d,kore: float):
        return self.space[point4d.t].set_kore_at_point(point4d.point,kore)
    
    def get_kore2_at_point(self, point4d: Point4d):
        return self.space[point4d.t].get_kore2_at_point(point4d.point)
    
    def set_kore2_at_point(self, point4d: Point4d,kore: float):
        return self.space[point4d.t].set_kore2_at_point(point4d.point,kore)
    
    def get_potential_at_point(self, point4d: Point4d, nb_op_shipyard: int):
        return self.space[point4d.t].get_potential_at_point(point4d.point,nb_op_shipyard)
    
    def get_gravity_at_point(self,point4d: Point4d):
        return self.space[point4d.t].get_gravity_at_point(point4d.point)

    def add_gravity(self,point4d: Point4d, val: float):
        return self.space[point4d.t].add_gravity(point4d.point, val)
    
    def multiply_gravity(self,point4d: Point4d, val: float):
        return self.space[point4d.t].multiply_gravity(point4d.point, val)
    
    def get_gravity2_at_point(self,point4d: Point4d):
        return self.space[point4d.t].get_gravity2_at_point(point4d.point)

    def add_gravity2(self,point4d: Point4d, val: float):
        return self.space[point4d.t].add_gravity2(point4d.point, val)
    
    def multiply_gravity2(self,point4d: Point4d, val: float):
        return self.space[point4d.t].multiply_gravity2(point4d.point, val)
    
    def calculate_avg_gravity(self,size):
        for i in range(len(self.space)):
            self.space[i].calculate_avg_gravity(size)
    
    
    def show_gravity(self,size):
        res1 = [[0 for _ in range(size)] for _ in range(size)]
        res2 = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                pos = Point(size - 1 - j, i)
                point = Point4d(pos,0,None)
                res1[i][j] = self.get_gravity2_at_point(point)
                res2[i][j] = self.get_kore2_at_point(point)
        plt.matshow(res1)
        plt.matshow(res2)
        plt.show()
        logger.info(f"{res2}")
    
    def get_danger_level_at_point(self,point4d: Point4d):
        return self.space[point4d.t].get_danger_level_at_point(point4d.point)

    def add_danger_level_at_point(self,point4d: Point4d, val: int):
        return self.space[point4d.t].add_danger_level_at_point(point4d.point, val)
    
    def get_danger_level2_at_point(self,point4d: Point4d):
        return self.space[point4d.t].get_danger_level2_at_point(point4d.point)

    def add_danger_level2_at_point(self,point4d: Point4d, val: int):
        return self.space[point4d.t].add_danger_level2_at_point(point4d.point, val)
    
    def show_danger_level(self,size):
        res = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                pos = Point(size - 1 - j, i)
                point = Point4d(pos,35,None)
                res[i][j] = self.get_danger_level2_at_point(point)
        plt.matshow(res)
        plt.show()
    
    def set_highway(self,pos: Point,d: Direction,distance: int, max_spawn: int) -> None:
        self.highway_points[pos] = (d,distance,max_spawn)
    
    def is_highway(self,pos: Point) -> Union[Tuple[Direction,int,int],None]:
        if pos in self.highway_points:
            return self.highway_points[pos]
        return None

    def show_highways(self,size):
        res = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                pos = Point(size - 1 - j, i)
                res[i][j] = (self.is_highway(pos) != None)
        plt.matshow(res)
        plt.show()

    def send_ghost_fleet(self, start_position: Point4d, ms : int, vplan: str, player_ID: PlayerId):
        plan = vplan
        if plan != "":
            if plan[0] in "NSWE":
                t = start_position.t
                ghost_fleet = Fleet(FleetId("ghost"),ms,Direction.from_char(plan[0]),start_position.point,0,plan,player_ID,self.space[t].board)
                size = self.space[0].board.configuration.size
                nb_ships_left = ms
                i = 0
                pos = start_position
                while 1:
                    if i+1 >= len(self.space):
                        break
                    if plan != "" and plan[0] == "C":
                        break
                    pos,plan = pos.next_move_and_update(plan,size)
                    shipyard = self.get_shipyard_at_point(pos)
                    if shipyard != None:
                        break

                    fleet = self.get_fleet_at_point(pos)
                    if fleet != None:
                        if fleet.player_id == player_ID:
                            if fleet.ship_count > nb_ships_left:
                                break
                            else:
                                nb_ships_left += fleet.ship_count
                        else:
                            if fleet.ship_count >= nb_ships_left:
                                break
                            else:
                                nb_ships_left = nb_ships_left-fleet.ship_count

                    for k in range(4):
                        p = pos.translate(Direction.from_index(k),size)
                        fleet = self.get_fleet_at_point(p)
                        if fleet != None and fleet.player_id != player_ID and fleet.ship_count >= nb_ships_left:
                            return

                    new_kore = ghost_fleet.kore + self.get_kore2_at_point(pos)*ratio_kore_ship_count(nb_ships_left)
                    ghost_fleet = Fleet(FleetId("ghost"),nb_ships_left,pos.d,pos.point,new_kore,plan,player_ID,self.space[pos.t].board)
                    self.ghost_fleets[pos] = ghost_fleet

                    i += 1

@Timer
def ApplyDangerLevel(space: Space):
    size = space[0].board.configuration.size
    for t in range(len(space)):
        if t==0:
            continue
        slice = space[t]
        ops = slice.board.opponents
        for op in ops:
            nb_influenced_cases = max(10-len(op.shipyards)//6,6)
            for op_shipyard in op.shipyards:
                pos = op_shipyard.position
                danger_lvl = op_shipyard.ship_count
                if danger_lvl >= 5:
                    space.add_danger_level2_at_point(Point4d(pos,t,None),danger_lvl)
                    for nt in range(1,nb_influenced_cases):
                        if t+nt<=len(space)-1:
                            for l in range(4):
                                d = Direction.from_index(l)
                                dp = d.rotate_left()
                                new_pos = translate_n_m(pos,d,0,dp,nt,size)
                                for k in range(nt):
                                    new_pos = translate_n_m(new_pos,d,1,dp,-1,size)
                                    space.add_danger_level2_at_point(Point4d(new_pos,t+nt,None),danger_lvl)

        """
        me = slice.board.current_player
        for shipyard in me.shipyards:
            pos = shipyard.position
            danger_lvl = shipyard.ship_count
            if danger_lvl > 2:
                for nt in range(10):
                    for l in range(4):
                        d = Direction.from_index(l)
                        dp = d.rotate_left()
                        for k in range(nt):
                            new_pos = translate_n_m(pos,d,k,dp,nt-k,size)
                            if t+nt>len(space)-1:
                                space.add_danger_level2_at_point(Point4d(new_pos,len(space)-1,None),-danger_lvl)
                            else:
                                space.add_danger_level2_at_point(Point4d(new_pos,t+nt,None),-danger_lvl)
        """
    #space.show_danger_level(size)

def early_game_filter():
    dimin = 1
    cc = -0.2 * dimin
    cd = -1 * dimin
    ld = -0.5 * dimin
    c = -0.8 * dimin
    d = 3 * dimin
    dd = 1.4 * dimin
    ratio = 4.5/10

    filter = [[cc,ld,dd,ld,cc],
            [ld,cd,d,cd,ld],
            [dd,d,c,d,dd],
            [ld,cd,d,cd,ld],
            [cc,ld,dd,ld,cc]]
    
    return filter,ratio


def mid_game_filter():
    dimin = 1
    cc = -0.15 * dimin
    cd = -1.15 * dimin
    ld = -0.5 * dimin
    c = -0.8 * dimin
    d = 3.33 * dimin
    dd = 1.3 * dimin
    ratio = 3.4/10

    filter = [[cc,ld,dd,ld,cc],
            [ld,cd,d,cd,ld],
            [dd,d,c,d,dd],
            [ld,cd,d,cd,ld],
            [cc,ld,dd,ld,cc]]
    return filter,ratio

def LoadGravity(space: Space):
    global saved_gravity2
    if turn%2 == 0:
        for i in range(2*len(space)//3+1,len(saved_gravity2)):
            space[i].set_slice_gravity(saved_gravity2[i-1])
    else:
        for i in range(len(space)//3+1,2*len(space)//3+1):
            space[i].set_slice_gravity(saved_gravity2[i-1])

@Timer
def ApplyGravity(space: Space):
    size = space[0].board.configuration.size
    me = space[0].board.current_player
    
    #filter,ratio = early_game_filter()
    #if turn > 100:
    #    filter,ratio = mid_game_filter()
    ratio = 3.2/10
    X = list(range(len(space)))
    r = ratio
    k = 30
    v = k*(len(space)*r*(len(space)-1))/(1-len(space)*r)
    u = -v/(len(space))
    j = v
    c = 0.0015

    myY = [(k*i+j)/(u*i+v)+c*i for i in X]
    # ---- Gravity for the enemy fleets and shipyards
    def applyfilter(begining,end):
        for n in range(begining,end):
            nops = space[n].board.opponents
            for op in nops:
                for fleet in op.fleets:
                    pos = Point4d(fleet.position,n,None)
                    space.add_gravity2(pos,reward_for_taking_an_ennemy_fleet(fleet.kore,fleet.ship_count)/math.sqrt(space[n].variance))
                    for i in range(4):
                        d = Direction.from_index(i)
                        posi = pos.translate(d,size)
                        space.add_gravity2(posi,reward_for_taking_a_side_ennemy_fleet(fleet.kore,fleet.ship_count)/math.sqrt(space[n].variance))
                #for shipyard in op.shipyards:
                #    space.add_gravity(Point4d(shipyard.position,n,None),reward_for_taking_a_shipyard(shipyard.max_spawn))
            for i in range(size):
                for j in range(size):
                    point = Point4d(Point(i,j),n,None)
                    space.add_gravity2(point,space.get_kore2_at_point(point))
        
        for n in range(end-1,begining-1,-1):
            for i in range(size):
                for j in range(size):
                    sum = 0
                    for k in range(len(filter)):
                        for l in range(len(filter[0])):
                            t = n+abs(-k+len(filter)//2)+abs(-l+len(filter)//2)
                            if t>len(space)-1:
                                t = len(space)-1
                            x = (i-k+len(filter)//2)%size
                            y = (j-l+len(filter[0])//2)%size
                            sum += space.get_gravity2_at_point(Point4d(Point(x,y),t,None))*filter[k][l]
                    point = Point4d(Point(i,j),n,None)
                    space.add_gravity2(point,(sum + space.get_kore2_at_point(point))/10)
        
        for n in range(end-1,begining-1,-1):
            for i in range(size):
                for j in range(size):
                    space.multiply_gravity2(Point4d(Point(i,j),n,None),myY[n]*ratio)
    # ---- Gravity for the enemy fleets and shipyards
    def applyfilterfast(begining,end):
        shipyard_map : Dict[Point,int]
        shipyard_map = dict()
        for n in range(end-1,begining-1,-1):
            nops = space[n].board.opponents
            for op in nops:
                for fleet in op.fleets:
                    pos = Point4d(fleet.position,n,None)
                    #space.add_gravity2(pos,reward_for_taking_an_ennemy_fleet(fleet.kore,fleet.ship_count)/math.sqrt(space[n].variance))
                    for i in range(4):
                        d = Direction.from_index(i)
                        posi = pos.translate(d,size)
                        space.add_gravity2(posi,reward_for_taking_a_side_ennemy_fleet(fleet.kore,fleet.ship_count)/math.sqrt(space[n].variance))
                for shipyard in op.shipyards:
                    shipyard_map[shipyard.position] = 0
                for shipyard in me.shipyards:
                    if n == end-1:
                        shipyard_map[shipyard.position] = 20
                    else:
                        shipyard_map[shipyard.position] += 1
                        space.add_gravity2(Point4d(shipyard.position,n,None),reward_for_defending_a_shipyard(shipyard.max_spawn,shipyard_map[shipyard.position])/math.sqrt(space[n].variance))
            for i in range(size):
                for j in range(size):
                    point = Point4d(Point(i,j),n,None)
                    space.add_gravity2(point,space.get_kore2_at_point(point))
        
        for n in range(end-1,begining-1,-1):
            for i in range(size):
                for j in range(size):
                    sum = 0
                    t = n+1
                    if t>len(space)-1:
                        t = len(space)-1
                    sum += space.get_gravity2_at_point(Point4d(Point((i+1)%size,j),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i-1)%size,j),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point(i,(j+1)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point(i,(j-1)%size),t,None))
                    sumtot = 3.3*sum
                    sum = 0
                    t = n+2
                    if t>len(space)-1:
                        t = len(space)-1
                    sum += space.get_gravity2_at_point(Point4d(Point((i+2)%size,j),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i-2)%size,j),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point(i,(j+2)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point(i,(j-2)%size),t,None))
                    sumtot += 1.25*sum
                    sum = 0
                    t = n+2
                    if t>len(space)-1:
                        t = len(space)-1
                    sum += space.get_gravity2_at_point(Point4d(Point((i+1)%size,(j+1)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i-1)%size,(j+1)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i+1)%size,(j-1)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i-1)%size,(j-1)%size),t,None))
                    sumtot -= sum
                    sum = 0
                    t = n+3
                    if t>len(space)-1:
                        t = len(space)-1
                    sum += space.get_gravity2_at_point(Point4d(Point((i+2)%size,(j+1)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i+2)%size,(j-1)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i-2)%size,(j+1)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i-2)%size,(j-1)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i+1)%size,(j+2)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i+1)%size,(j-2)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i-1)%size,(j+2)%size),t,None))
                    sum += space.get_gravity2_at_point(Point4d(Point((i-1)%size,(j-2)%size),t,None))
                    sumtot -= sum*0.6
                    point = Point4d(Point(i,j),n,None)
                    space.add_gravity2(point,(sumtot + space.get_kore2_at_point(point))/10)
        for n in range(end-1,begining-1,-1):
            for i in range(size):
                for j in range(size):
                    space.multiply_gravity2(Point4d(Point(i,j),n,None),myY[n]*3.2/10)

    
    #if turn%2 == 0:
    #    applyfilterfast(0,2*len(space)//3+1)
    #else:
    #    applyfilterfast(2*len(space)//3+1,len(space))
    #    applyfilterfast(0,len(space)//3+1)
    applyfilterfast(0,len(space))
    #space.show_gravity(size)
    
@Timer
def Add_highways(space: Space):
    size = space[0].board.configuration.size
    me = space[0].board.current_player
    for shipyard in space[-1].board.shipyards.values():
        pos = shipyard.position
        space.set_highway(pos,Direction.NORTH,0,shipyard.max_spawn)
        for i in range(4):
            d = Direction.from_index(i)
            posi = pos
            for j in range(size//2):
                posi = posi.translate(d.to_point(),size)
                exist_highway = space.is_highway(posi)
                if exist_highway != None:
                    (direction,dist,spawn) = exist_highway
                    if dist > j+1 or (dist == j+1 and dist+spawn > j+1+shipyard.max_spawn):
                        space.set_highway(posi,d.opposite(),j+1,shipyard.max_spawn)
                    else:
                        break
                else:
                    space.set_highway(posi,d.opposite(),j+1,shipyard.max_spawn)
    # space.show_highways(size)
    

def domination_strategy(space: Space):
    size = space[0].board.configuration.size
    shipyard_table : List[Dict[Point,Tuple[int,int,int]]] # (my_shipyard:Bool,nb_ships:int,max_spawn:int)
    shipyard_table = []
    for i in range(len(space)):
        me = space[i].board.current_player
        ops = space[i].board.opponents
        shipyard_table.append(dict())
        my_shipyards = me.shipyards
        for shipyard in my_shipyards:
            shipyard_table[-1][shipyard.position]=(True,shipyard.ship_count,shipyard.max_spawn)
        for op in ops:
            for shipyard in op.shipyards:
                shipyard_table[-1][shipyard.position]=(False,shipyard.ship_count,shipyard.max_spawn)

    distance_map : Dict[Point,Dict[int,List[Point]]]
    distance_map = dict()
    for p in shipyard_table[-1]:
        own_distance_map = dict()
        for q in shipyard_table[-1]:
            if q != p:
                dist = p.distance_to(q,size)
                if dist in own_distance_map:
                    own_distance_map[dist] += [q]
                else:
                    own_distance_map[dist] = [q]
        distance_map[p] = own_distance_map
    
    power_center = Point(15,5)
    max_max_spawn = 0
    s_table = shipyard_table[-1]
    for p in s_table:
        (my_shipyard,nb_ships,max_spawn) = s_table[p]
        if not my_shipyard:
            if max_spawn > max_max_spawn:
                max_max_spawn = max_spawn
                power_center = p
    
    
    my_vulnerabilities_map : List[Dict[Point,int]]
    my_vulnerabilities_map  = [dict() for _ in range(len(space))]# Compted in ship number difference
    op_vulnerabilities_map : List[Dict[Point,float]]
    op_vulnerabilities_map  = [dict() for _ in range(len(space))]# Compted in ship number difference
    
    closest_my_shipyard = None
    prepare_attack = None
    best_attack_value = 0
    for i in range(1,len(space)):
        s_table = shipyard_table[i]
        for p in s_table:
            (p_my_shipyard,p_nb_ships,p_max_spawn) = s_table[p]
            if p_my_shipyard == True:
                vulnerability_value = 0
                for j in range(1,i+1): # j la distance avec i et i-j le point dans le temps, le point i=j est au temps 0
                    if j in distance_map[p]:
                        point_list = distance_map[p][j]
                        for q in point_list:
                            if q in shipyard_table[i-j]:
                                (my_shipyard,nb_ships,max_spawn) = shipyard_table[i-j][q]
                                if my_shipyard == True:
                                    vulnerability_value += nb_ships
                                else:
                                    vulnerability_value -= nb_ships
                (my_shipyard,nb_ships,max_spawn) = s_table[p]
                my_vulnerabilities_map[i][p] = p_nb_ships + vulnerability_value
            else:
                vulnerability_value = 0
                nb_needed_allies_to_attack = 0
                closest_ally_to_attack = None
                nb_needed_enemies_to_defend = 0
                for j in range(1,i+1): # j la distance avec i et i-j le point dans le temps, le point i=j est au temps 0
                    if j in distance_map[p]:
                        point_list = distance_map[p][j]
                        for q in point_list:
                            if q in shipyard_table[i-j]:
                                (my_shipyard,nb_ships,max_spawn) = shipyard_table[i-j][q]
                                if my_shipyard == False:
                                    vulnerability_value += nb_ships
                                    nb_needed_enemies_to_defend += 1
                                else:
                                    vulnerability_value -= nb_ships
                                    nb_needed_allies_to_attack += 1
                                    if closest_ally_to_attack == None:
                                        closest_ally_to_attack = q
                (my_shipyard,nb_ships,max_spawn) = s_table[p]
                if closest_ally_to_attack != None:
                    dist_closest_ally = closest_ally_to_attack.distance_to(p,size)
                    val = ((-(p_nb_ships + vulnerability_value))//10)*p_max_spawn*p_max_spawn*p_max_spawn*nb_needed_enemies_to_defend/((dist_closest_ally+1)*(nb_needed_allies_to_attack)*(i+1))
                    if val > best_attack_value:
                        val = best_attack_value
                        prepare_attack = (closest_ally_to_attack,i-dist_closest_ally)
    
    if prepare_attack != None:
        logger.info(f"{prepare_attack}")
        pass
    return my_vulnerabilities_map,op_vulnerabilities_map,power_center,prepare_attack
    
    
    
@Timer
def Normalize(space : Space, size: int):
    shipyard_map : Dict[Point,int]
    shipyard_map = dict() # (pos,ennemy) [Point,Bool]
    for i in range(len(space)-1,-1,-1):
        nb_op_shipyard = sum([len(op.shipyards) for op in space[i].board.opponents])
        potential_map = [[space.get_potential_at_point(Point4d(Point(x,y),i,None),nb_op_shipyard) for x in range(size)] for y in range(size)]
        avg = sum([sum([potential_map[x][y] for x in range(size)]) for y in range(size)])/(size*size)
        v = sum([sum([(potential_map[x][y]-avg)**2 for x in range(size)]) for y in range(size)])/(size*size)
        space[i].variance = v
        for x in range(size):
            for y in range(size):
                space.set_kore2_at_point(Point4d(Point(x,y),i,None),potential_map[y][x] / math.sqrt(v))

# ------------ Arbitrary values to that impacts the Astar search
@Timer
def value(elmt,max_spawn):
    (r,ar,tr,p,cs,ms,Ms,ln,tplan) = elmt
    plan,lmoves = tplan
    kore = 0
    for (delta,k) in r.items():
        ratio = ratio_kore_ship_count(ms+delta)
        kore += k*ratio
    # print(plan+str(lmoves),r*ratio,ar,tr)
    if turn <= 150:
        return kore*3+tr+ar-0.051*ln-0.0023*ln*ln-ln*ms*0.0004-ln*ln*ms*0.000041-0.06*(perso_kore>=100)-0.1*(perso_kore>=300)-0.08*(perso_kore>=1000)+0.15*(10*(max_spawn+1)*ln>=perso_kore)
    return kore*3+tr+ar-0.051*ln-0.0023*ln*ln-ln*ms*0.0004-ln*ln*ms*0.000041

def exploration_cut(ln,max_spawn,ms):
    if turn < 150:
        return -0.3+0.1*(perso_kore>=100)+0.13*(perso_kore>=300)+0.15*(perso_kore>=800)+0.2*(perso_kore>=2000)+0.1*(perso_kore>=4000)+0.01*ln
    return -0.5
    #return -0.3+0.1*(perso_kore>=100)+0.13*(perso_kore>=300)+0.15*(perso_kore>=800)+0.2*(perso_kore>=2000)+0.1*(perso_kore>=4000)+0.01*ln

def compute_vulnerability_cost(ms,ln,max_spawn,pos,my_vulnerabilities:List[Dict[Point, int]]):
    if turn <= 150:
        return 0
    vuln = my_vulnerabilities[ln]
    if pos in vuln:
        nb_ships_of_vulnerability = vuln[pos]
        if nb_ships_of_vulnerability != None and ms > nb_ships_of_vulnerability:
            return - math.sqrt((ms-nb_ships_of_vulnerability)//10+1)*max_spawn*max_spawn*0.02
    return 0

def compute_vulnerability_gain(ms,ln,max_spawn1,pos1: Point,cs,max_spawn2,pos2:Point,my_vulnerabilities:List[Dict[Point, int]]):
    if turn <= 150:
        return 0
    maxi_loss = 0
    maxi_pos1_vuln = 0
    dist = pos1.distance_to(pos2,size)
    for i in range(min(ln+dist,len(my_vulnerabilities))):
        vuln = my_vulnerabilities[i]
        if pos1 in vuln:
            nb_ships_of_vulnerability = vuln[pos1]
            if nb_ships_of_vulnerability != None and ms > nb_ships_of_vulnerability:
                val = - math.sqrt((ms-nb_ships_of_vulnerability)//10+1)*max_spawn1*max_spawn1*0.02
                if val < maxi_loss:
                    maxi_loss = val
            if nb_ships_of_vulnerability < maxi_pos1_vuln:
                maxi_pos1_vuln = nb_ships_of_vulnerability
    maxi_gain = 0
    maxi_pos2_vuln = 0
    for i in range(ln,len(my_vulnerabilities)):
        vuln = my_vulnerabilities[i]
        if pos2 in vuln:
            nb_ships_of_vulnerability = vuln[pos2]
            if nb_ships_of_vulnerability != None and nb_ships_of_vulnerability < 0 and nb_ships_of_vulnerability+cs>=0:
                val = math.sqrt((-nb_ships_of_vulnerability)//10+1)*max_spawn2*max_spawn2*0.02
                if val > maxi_gain:
                    maxi_gain = val
            if nb_ships_of_vulnerability < maxi_pos2_vuln:
                maxi_pos2_vuln = nb_ships_of_vulnerability
    if maxi_loss == 0 and maxi_gain == 0:
        return (min(maxi_pos2_vuln+cs,maxi_pos1_vuln-ms)-min(maxi_pos2_vuln,maxi_pos1_vuln))*0.003
    return maxi_gain + maxi_loss
    
def reward_preparing_for_attack(pos1: Point,pos2: Point,length:int,prepare_attack: Tuple[Point,int],size):
    if prepare_attack == None:
        return 0
    pos,l = prepare_attack
    if length <= l:
        dist1 = pos1.distance_to(pos,size)
        dist2 = pos2.distance_to(pos,size)
        return (dist1-dist2)*0.02+(dist2==0)*0.5
    return 0

def bonus_on_highway(ln):
    return 1#5+ln-ln*ln*0.1

def reward_for_taking_a_shipyard(max_spawn):
    return 2*max_spawn*max_spawn*max_spawn+10*max_spawn*max_spawn+60*max_spawn+400

def reward_for_defending_a_shipyard(max_spawn,turn_remaining):
    return (10*max_spawn*max_spawn+200)*max(0,(7-turn_remaining)/6)

def reward_for_taking_an_ennemy_fleet(kore,ship_count):
    return 2*kore + 2*ship_count

def reward_for_taking_a_side_ennemy_fleet(kore,ship_count):
    return kore

def reward_for_taking_an_ally_fleet(kore,ship_count): # not endangered
    return 0

def reward_for_taking_an_ally_endangered_fleet(kore,ship_count): # not endangered
    return kore+10*ship_count

def malus_for_max_spawn(max_spawn):
    return (4-max_spawn)*max_spawn*max_spawn*0.01# 15-max_spawn*max_spawn*0.8

def minimum_ships_for_this_state(kore,max_spawn):
    ms = 2
    if max_spawn == 5:
        if kore > 50:
            ms = 3
        if kore > 400:
            ms = 5
    if max_spawn == 6:
        if kore > 60:
            ms = 3
        if kore > 120:
            ms = 5
        if kore > 300:
            ms = 8
    if max_spawn == 7:
        if kore > 70:
            ms = 5
        if kore > 150:
            ms = 8
        if kore > 400:
            ms = 13
    if max_spawn == 8:
        if kore > 80:
            ms = 8
        if kore > 200:
            ms = 13
        if kore > 500:
            ms = 21
    if max_spawn == 9:
        if kore > 90:
            ms = 13
        if kore > 250:
            ms = 21
        if kore > 1500:
            ms = 34
        if kore > 5000:
            ms = 55
    if max_spawn == 10:
        if kore > 100:
            ms = 13
        if kore > 200:
            ms = 21
        if kore > 1000:
            ms = 34
        if kore > 4000:
            ms = 55
    return ms

# -------- Auxiliary functions for astar
def where_to_capacity(space: Space,t: int, fleet: Fleet,shipyard: Shipyard):
    fleet_id = fleet.id
    size = space[t].board.configuration.size
    player_id = space[t].board.current_player_id
    i = t
    while i < len(space)-1 and fleet_id in space[i].board.fleets:
        i += 1
    if i == 0:
        return 0,0 # The fleet died so it's not in danger
    if i >= len(space)-1:
        return -1,i,None
    real_ship_count = space[i-1].board.fleets[fleet_id].ship_count
    p = Point4d(next_position(space[i-1].board.fleets[fleet_id],size),i,None)
    this_shipyard = space.get_shipyard_at_point(p)
    if this_shipyard != None:
        if this_shipyard.player_id == player_id:
            if this_shipyard.id == shipyard.id:
                return 100001,i,this_shipyard.position
            return 100000,i,this_shipyard.position
    this_fleet = space.get_fleet_at_point(p)
    if this_fleet != None:
        if this_fleet.player_id == player_id:
            if this_fleet.ship_count > 2*real_ship_count:
                cap,its_length,its_shipyard_pos = where_to_capacity(space,i,this_fleet,shipyard)
                if cap == -1 or cap == 0:
                    return -1,its_length,its_shipyard_pos
                return min(this_fleet.ship_count-2*real_ship_count-1,cap),its_length,its_shipyard_pos
            else:
                return -1,i,None
    return -1,i,None

def sortedAppend_slow(liste, element,max_spawn):
    liste.append(element)
    i = len(liste)-1
    while i > 0 and value(liste[i],max_spawn) < value(liste[i-1],max_spawn):
        liste[i], liste[i-1] = liste[i-1], liste[i]
        i -= 1
        
def sortedAppend(liste, element,max_spawn):
    d = 0
    f = len(liste)
    m = (d+f)//2
    v = value(element,max_spawn)
    while d != f:
        if value(liste[m],max_spawn) <= v:
            d = m+1
        else:
            f = m
        m = (d+f)//2
    liste.insert(d,element)

def update_tplan(tplan: Tuple[str,int],dir: Direction)->Tuple[str,int]:
    plan, lmoves = tplan
    dec_plan = decompose(plan)
    if plan == "":
        return (dir.to_char(),0)
    if lmoves > 0:
        pre_dir = dec_plan[-1]
        if dir.to_char() == pre_dir:
            return (compose(dec_plan),lmoves+1)
        else:
            dec_plan.append(str(lmoves))
            dec_plan.append(dir.to_char())
            return (compose(dec_plan),0)
    else:
        pre_dir = dec_plan[-1]
        if dir.to_char() == pre_dir:
            return (compose(dec_plan),1)
        else:
            dec_plan.append(dir.to_char())
            return (compose(dec_plan),0)


def astar3d(space: Space, shipyard: Shipyard, my_vulnerabilities: List[Dict[Point,int]], op_vulnerabilities: List[Dict[Point,int]], power_center: Point, prepare_attack: Tuple[Point,int], incoming_attacks :List[Tuple[int,int]] = [])->Tuple[str,int]:
    """Pathfinding function that creates the bestplan using an heuristic (value),
    based on A*. There are 2 dimensions of space and one of time.

    Args:
        space (Space): All the next boards and informations associated to them
        shipyard (Shipyard): The current shipyard we start from
        incoming_attacks (List[Tuple[int,int]], optional): The list of the attacks on the shipyard. Defaults to [].

    Returns:
        Tuple[str,int]: _description_
    """
    board = space[0].board
    size = board.configuration.size
    me = board.current_player
    nb_shipyards = len(me.shipyards)
    ms = minimum_ships_for_this_state(me.kore,shipyard.max_spawn)
    Ms = shipyard.ship_count
    max_spawn = shipyard.max_spawn
    logger.info(f"incoming : {incoming_attacks}")
    if incoming_attacks != []:
        if incoming_attacks[0][0] == 1:
            Ms = Ms - incoming_attacks[0][1]
    priority_list: List[Tuple[Dict[int,float],float,float,Point4d,int,int,int,int,Tuple[str,int]]]
    priority_list = []
    if Ms >= ms:
        point = Point4d(shipyard.position,0,None)
        priority_list = [(dict(),0,space.get_gravity2_at_point(point)+bonus_on_highway(0),point,ms,ms,Ms,0,("",0))]# [(reward,local_reward,point4d,nb_current_ships,nb_ships_min_needed,nb_ships_max_possible,nb_steps,tplan)]
    best_plan : List[Tuple[float,str,int]] # kore_number,plan,minimum_ships
    best_plan = []
    nb_iter = 0
    nsatltas= shipyard.ship_count #number_ship_available_to_launch_to_another_shipyard
    for (limit_step,limit_ship_nb) in incoming_attacks:
        nsatltas -= limit_ship_nb
    while priority_list != [] and nb_iter<2000//nb_shipyards:
        nb_iter += 1
        (r,ar,tr,p,cs,ms,Ms,ln,tplan) = priority_list.pop()
        """
        r : Reward Dict kore associated to delta_nb_ships
        ar : additional reward, usally kore picked from other fleet
        tr : Temporary reward, usually gravity
        p : position in 4d (Point4d)
        cs : current number of ships
        ms : minimum number of ships to send by the shipyard for the plan
        Ms : ultimate maximum number of ships that the shipyard can send
        ln : length of the flight
        tplan : plan to arrive to p and the length of the last straight line
        """
        def requirement_satisfied(v,cs,ms,Ms,ln,cut_v,same_shipyard=True):
            if v <= cut_v:
                return False
            if ms > Ms:
                return False
            if turn + ln >= 399:
                return False
            nb_ships_needed = 0
            for (limit_step,limit_ship_nb) in incoming_attacks:
                nb_ships_needed += limit_ship_nb
                if ln >= limit_step or same_shipyard == False:
                    if Ms-ms <= nb_ships_needed:
                        return False
                elif Ms-ms+cs <= nb_ships_needed and same_shipyard == True:
                    return False
            return True
            
        max_move = maximum_flight_plan_length(Ms)
        plan,lmoves = tplan

        # -------------------- Checked if arrived at a destination ----------------
        # We do it here because it means that it had a high value
        logger.debug(f"plan : {plan+str(lmoves)}, value : {value((r,ar,tr,p,cs,ms,Ms,ln,tplan),max_spawn)}, detail: r{r}, ar{ar}, tr{tr},ln{ln}")
        if ln >= len(space)-1 or p.t >= len(space) - 1:
            continue

        (sr,sar,s_tr,sp,scs,sms,sMs,sln,stplan) = (r,ar,tr,p,cs,ms,Ms,ln,tplan) # save the values for the 4 iterations
        for i in range(4):
            (r,ar,tr,p,cs,ms,Ms,ln,tplan) = (deepcopy(sr),sar,s_tr,sp,scs,sms,sMs,sln,stplan)
            new_direction = Direction.from_index(i)
            (new_plan,new_lmoves) = update_tplan(tplan,new_direction)
            
            if len(new_plan)>=max_move:
                continue
            
            if len(new_plan) > len(plan):
                if minimum_ships_num(len(new_plan))>ms:
                    diff = minimum_ships_num(len(new_plan))-ms
                    ms = ms + diff
                    cs = cs + diff
            
            new_p = p.move(new_direction,size)
            kore = space.get_kore2_at_point(new_p)
            gravity = space.get_gravity2_at_point(new_p)
            
            gravity += compute_vulnerability_cost(ms,ln+1,max_spawn,shipyard.position,my_vulnerabilities)
            
            
            # ----- Check if collide with another fleet
            is_multiple_attack = 0
            fleet = space.get_fleet_at_point(new_p)
            pre_fleets = space[new_p.t-1].board.fleets
            fleet_kore = 0
            if fleet != None and fleet.id in pre_fleets:
                pre_fleet = pre_fleets[fleet.id]
                if fleet.player_id != me.id:
                    if fleet.ship_count >= Ms+cs-ms:
                        continue
                    if fleet.ship_count >= cs:
                        diff = fleet.ship_count + 1-cs
                        cs = 1
                        ms = ms + diff
                    else:
                        cs -= fleet.ship_count
                    # fleet_kore += reward_for_taking_an_ennemy_fleet(fleet.kore,fleet.ship_count)
                else:
                    max_capacity_fleet,real_length,shipyard_pos = where_to_capacity(space,ln,fleet,shipyard)
                    #if max_capacity_fleet != -1 and max_capacity_fleet != 0 and max_capacity_fleet != 100000:
                    #    logger.info(f"{max_capacity_fleet}")
                    if max_capacity_fleet == -1:
                        if pre_fleet.ship_count < Ms+cs-ms:    # Get the fleet
                            fleet_kore += reward_for_taking_an_ally_endangered_fleet(fleet.kore,fleet.ship_count)/math.sqrt(space[ln].variance)
                            if pre_fleet.ship_count >= cs:
                                diff = pre_fleet.ship_count + 1 - cs
                                cs = fleet.ship_count + pre_fleet.ship_count + 1
                                ms = ms + diff
                                is_multiple_attack = 1
                            else:
                                continue
                        else:
                            continue
                    else:
                        if cs < max_capacity_fleet and (cs < pre_fleet.ship_count):
                            if max_capacity_fleet != 100001:
                                if ms > nsatltas:
                                    continue
                            shipyard2 = space[real_length].get_shipyard_at_point(shipyard_pos)
                            if shipyard2 != None:
                                gravity = compute_vulnerability_gain(ms,real_length,max_spawn,shipyard.position,cs,shipyard2.max_spawn,shipyard_pos,my_vulnerabilities) + reward_preparing_for_attack(shipyard.position,shipyard_pos,real_length,prepare_attack,size)
                                v = value((r,ar,gravity,p,cs,ms,Ms,real_length,(new_plan,new_lmoves)),max_spawn)
                                if requirement_satisfied(v,cs,ms,Ms,real_length,0,same_shipyard=(shipyard.position==shipyard_pos)):
                                    logger.debug(f"to a fleet : {new_plan}, with value: {v}")
                                    best_plan.append((v,new_plan,ms))
                                continue

                        elif pre_fleet.ship_count < cs:
                            cs += fleet.ship_count
                        else:
                            continue
            
            for j in range(4):
                d = Direction.from_index(j)
                posj = new_p.translate(d,size)
                fleetj = space.get_fleet_at_point(posj)
                if fleetj != None:
                    if fleetj.player_id != me.id:
                        if fleetj.ship_count >= Ms+cs-ms:
                            ms = Ms + 1 # = continue
                            break
                        if fleetj.ship_count >= cs:
                            diff = fleetj.ship_count + 1-cs
                            cs = 1
                            ms = ms + diff
                        fleet_kore += (reward_for_taking_a_side_ennemy_fleet(fleetj.kore,fleetj.ship_count)+10*fleetj.ship_count*(is_multiple_attack==1))/math.sqrt(space[ln].variance)
                        is_multiple_attack = 1
            if ms > Ms:
                continue
            # -------- Check if collide with a shipyard
            """
            nshipyard = space.get_shipyard_at_point(new_p)
            if nshipyard != None:
                if nshipyard.id != shipyard.id:
                    if ms > nsatltas:
                        continue
                if nshipyard.player_id != me.id:
                    if nshipyard.ship_count >= Ms+cs-ms:
                        continue
                    if nshipyard.ship_count >= cs:
                        diff = nshipyard.ship_count + 1-cs
                        cs = 1
                        ms = ms + diff
                        # fleet_kore += reward_for_taking_a_shipyard(nshipyard.max_spawn)
                        gravity = 0
                    else:
                        cs = cs - nshipyard.ship_count
                        # fleet_kore += reward_for_taking_a_shipyard(nshipyard.max_spawn)
                        gravity = 0
                else:
                    gravity = malus_for_max_spawn(shipyard.max_spawn)
                if requirement_satisfied(v,ms,Ms,ln):
                    logger.debug(f"to shipyard : {new_plan}")
                    best_plan.append((value((r,ar,gravity,p,cs,ms,Ms,ln,(new_plan,new_lmoves))),new_plan,ms))
                continue
            """
            # ------ Check if arrived to an Highway
            exit_highway = space.is_highway(new_p.point)
            if exit_highway != None:
                (direction,distance_to_shipyard,nmax_spawn) = exit_highway
                if new_plan != "":
                    if new_direction == direction or distance_to_shipyard==0:
                        np = new_p.move_n_times(direction,size,distance_to_shipyard)
                        if np.t > len(space) - 1:
                            np = Point4d(np.point,len(space)-1,np.d)
                        # -------- Check the danger of the position
                        danger_lvl = space.get_danger_level2_at_point(np)
                        if danger_lvl >= Ms+cs-ms:
                            continue
                        elif danger_lvl >= cs:
                            diff = danger_lvl +1 -cs
                            cs = cs + diff
                            ms = ms + diff
                        nshipyard = space.get_shipyard_at_point(np)
                        if nshipyard != None:
                            if nshipyard.player_id == me.id:
                                bonus_defend = 0
                                nb_fleet_needed_to_defend = -1
                                nnp = np
                                for i in range(7):
                                    nnp = nnp.wait()
                                    if nnp.t >= len(space):
                                        break
                                    nnshipyard = space.get_shipyard_at_point(nnp)
                                    if nnshipyard.player_id != me.id:
                                        if nb_fleet_needed_to_defend < nnshipyard.ship_count:
                                            if nnshipyard.ship_count-nb_fleet_needed_to_defend >= Ms+cs-ms:
                                                break
                                            if nnshipyard.ship_count-nb_fleet_needed_to_defend >= cs:
                                                diff = nshipyard.ship_count + 1-cs
                                                cs = 1
                                                ms = ms + diff
                                                bonus_defend += gravity*(6-i)/6
                                            else:
                                                cs -= (nnshipyard.ship_count-nb_fleet_needed_to_defend)
                                                bonus_defend += gravity*(6-i)/6
                                            nb_fleet_needed_to_defend = nnshipyard.ship_count
                                ln = ln+distance_to_shipyard
                                gravity = malus_for_max_spawn(nmax_spawn)+bonus_defend + compute_vulnerability_gain(ms,ln,max_spawn,shipyard.position,cs,nshipyard.max_spawn,nshipyard.position,my_vulnerabilities) + 2*reward_preparing_for_attack(shipyard.position,nshipyard.position,ln,prepare_attack,size)
                                new_lmoves = new_lmoves+distance_to_shipyard
                                v = value((r,ar,gravity,p,cs,ms,Ms,ln,(new_plan,new_lmoves)),max_spawn)
                                if requirement_satisfied(v,cs,ms,Ms,ln,0,same_shipyard=(shipyard.position==nshipyard.position)):
                                    logger.debug(f"to ally highway : {new_plan} with value: {v}")
                                    best_plan.append((v,new_plan,ms))
                                continue
                            else:
                                if nshipyard.position == shipyard.position:
                                    continue
                                if nshipyard.ship_count >= Ms+cs-ms:
                                    continue
                                if nshipyard.ship_count >= cs:
                                    diff = nshipyard.ship_count + 1-cs
                                    cs = 1
                                    ms = ms + diff
                                    # fleet_kore += reward_for_taking_a_shipyard(nshipyard.max_spawn)
                                    ln = ln+distance_to_shipyard-1
                                    new_lmoves = new_lmoves+distance_to_shipyard

                                else:
                                    cs -= nshipyard.ship_count
                                    # fleet_kore += reward_for_taking_a_shipyard(nshipyard.max_spawn)
                                    ln = ln+distance_to_shipyard
                                    new_lmoves = new_lmoves+distance_to_shipyard

                                #logger.info(f"to enemy highway : {new_plan}")
                                gravity = kore + compute_vulnerability_cost(ms,ln,max_spawn,shipyard.position,my_vulnerabilities) + reward_preparing_for_attack(shipyard.position,nshipyard.position,ln,prepare_attack,size)
                                v = value((r,ar,gravity,p,cs,ms,Ms,ln,(new_plan,new_lmoves)),max_spawn)
                                if requirement_satisfied(v,cs,ms,Ms,ln,0,same_shipyard=(shipyard.position==nshipyard.position)):
                                    logger.debug(f"to enemy highway : {new_plan} with value: {v}")
                                    best_plan.append((v,new_plan,ms))
                                continue
                            new_p = np
                                            
                    if new_direction.opposite() == direction:
                        gravity += bonus_on_highway(ln)
                if distance_to_shipyard == 0:
                    continue
            else:
                if len(new_plan)+(new_lmoves//10)>=max_move-1:
                    continue
            
            # -------- Check the danger of the position
            danger_lvl = space.get_danger_level2_at_point(new_p)
            if danger_lvl >= Ms+cs-ms:
                continue
            elif danger_lvl >= cs:
                diff = danger_lvl +1 -cs
                cs = cs + diff
                ms = ms + diff
            
            new_kore = r
            delta_nb_ships = cs-ms
            if delta_nb_ships in new_kore:
                new_kore[delta_nb_ships] += kore
            else:
                new_kore[delta_nb_ships] = kore
            new_elmt = (new_kore,ar+fleet_kore,gravity,new_p,cs,ms,Ms,ln+1,(new_plan,new_lmoves))

            if not requirement_satisfied(value(new_elmt,max_spawn),cs,ms,Ms,ln+1,exploration_cut(ln+1,max_spawn,ms)):
                continue
            sortedAppend(priority_list,new_elmt,max_spawn)
            
    logger.info(f"found in {nb_iter}")
    if best_plan == []:
        return ("",0)
    best = max(best_plan)
    
    if best[0]>0:
        logger.info(f"best ship : {best[1]} sends {best[2]} ships with value of {best[0]}")
        # TODO : Optimize ms to get the best value (get an intelligent function to do that)
        return (best[1],best[2])
    return ("",0)


def detect_attack(shipyards_pos : List[Point],boards : List[Board]) -> Dict[Point,List[Tuple[int,int]]]:
    """

    Args:
        boards (List[Point]): A list of Points

    Returns:
        Dict[ShipyardId,List[Tuple[int,int]]]: a dictionnary that, for each ShipyardID
        reffers to a list of incoming attacks on the shipyard :
        the format is : (nb of turn to the attack, nb of ships attacking)
    """
    shipyard_next_pos = []
    for posi in shipyards_pos:
        for i in range(4):
            d = Direction.from_index(i)
            new_pos = translate_n(posi,d,1,size)
            shipyard_next_pos.append(new_pos)
    in_danger: Dict[Point,List[Tuple[int,int]]]
    in_danger = dict()
    for i,nboard in enumerate(boards):
        nops = nboard.opponents
        op_fleets = concat([[fleet for fleet in op.fleets] for op in nops])
        for fleet in op_fleets:
            if fleet.position in shipyard_next_pos:
                next_pos = next_position(fleet,size)
                if next_pos != None:
                    p = shipyard_next_pos.index(fleet.position)//4
                    if i < len(boards)-1 and boards[i+1].get_shipyard_at_point(next_pos) != None:
                        if shipyards_pos[p] in in_danger:
                            add = 1
                            for j,(t,nb) in enumerate(in_danger[shipyards_pos[p]]):
                                if i == t:
                                    in_danger[shipyards_pos[p]][j] = (i+1,nb+fleet.ship_count)
                                    add = 0
                            if add:
                                in_danger[shipyards_pos[p]] += [(i+1,fleet.ship_count)]
                        else:
                            in_danger[shipyards_pos[p]] = [(i+1,fleet.ship_count)]
    logger.info(f"attack detected : {in_danger}")
    return in_danger

@Timer
def detect_help_needed(shipyards_pos: List[Point], incoming_attack : Dict[Point,List[Tuple[int,int]]], boards: List[Board]) ->  Dict[Point,List[Tuple[int,int]]]:
    need_help = dict()
    for shipyard_pos in shipyards_pos:
        if shipyard_pos in incoming_attack:
            for i,(nb_step,nb_ships) in enumerate(incoming_attack[shipyard_pos]):
                if nb_step == 0:
                    continue
                if i == 0:
                    iboard = boards[nb_step-1]
                    ime = iboard.current_player
                    ishipyard = iboard.get_shipyard_at_point(shipyard_pos)
                    if ishipyard != None and nb_ships > ishipyard.ship_count:
                        need_help[shipyard_pos] = [(nb_step,nb_ships-ishipyard.ship_count)]
                        logger.info(f"need help !! Need {nb_ships-ishipyard.ship_count} ships in {nb_step} turns")
                else:
                    iboard = boards[nb_step-1]
                    ime = iboard.current_player
                    ishipyard = iboard.get_shipyard_at_point(shipyard_pos)
                    if ishipyard != None and ishipyard.player_id == ime.id:
                        if nb_ships > ishipyard.ship_count:
                            if shipyard_pos in need_help:
                                need_help[shipyard_pos] += [(nb_step,nb_ships-ishipyard.ship_count)]
                                logger.info(f"need 2nd help !! Need {nb_ships-ishipyard.ship_count} ships in {nb_step} turns")
                            else:
                                need_help[shipyard_pos] = [(nb_step,nb_ships-ishipyard.ship_count)]
                                logger.info(f"need help for the others !! Need {nb_ships-ishipyard.ship_count} ships in {nb_step} turns")
                    else:
                        if shipyard_pos in need_help:
                            need_help[shipyard_pos] += [(nb_step,nb_ships)]
                        else:
                            need_help[shipyard_pos] = [(nb_step,nb_ships)]
    return need_help

#building_ships_send = [Point4d(Point(11,18),10,None)]
def agent(obs: Observation, config: Configuration):
    global turn, saved_gravity2, building_ships_send,size, perso_kore
    board = Board(obs, config)
    if obs.step == 0:
        init_logger(logger)
        board.print_kore()
    
    board = Board(obs, config)
    if obs.step == 0:
        print(tab_kore(board))
    step = board.step
    nextt_action = ""
    my_id = obs["player"]
    remaining_time = obs["remainingOverageTime"]
    logger.info(f"<step_{step + 1}>, remaining_time={remaining_time:.1f}")
    me = board.current_player
    perso_kore = me.kore
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    convert_cost = board.configuration.convert_cost
    kore_left = me.kore
    ops = board.opponents
    size = board.configuration.size
    me_tot_ship_count = tot_ship_count(me,board)
    if ops != []:
        biggest_ship_count_threat = max([tot_ship_count(op,board) for op in ops])
    else:
        biggest_ship_count_threat = 0

    #if obs.step%50 == 49:
    #    print(me.kore,len(me.shipyards),sum([fleet.ship_count for fleet in me.fleets]))
    #    shipyard[2] = 2
    # ------ update building ships
    building_ships_send = [p for p in building_ships_send if p.t >= turn]
    # --------------------------------- Predict the next Boards ---------------------------------
    #if obs.step == 0:
    boards : List[Board]
    boards = predicts_next_boards(obs,config)
    logger.info("boards generated")
    
    # --------------------------------- Generate Gravity for all ships ----------------------
    space = Space(boards)
    Normalize(space, size)
    logger.info("normalized")
    ApplyDangerLevel(space)
    logger.info("danger level added")
    #LoadGravity(space)
    #logger.info("saved gravity loaded")
    ApplyGravity(space)
    logger.info("gravity added")
    Add_highways(space)
    logger.info("highways added")
    """
    if obs.step%10 == 0:
        space.show_gravity(size)
    """
    my_vulnerabilities, op_vulnerabilities,power_center,prepare_attack = domination_strategy(space)
    
    # --------------------------------- Choose a move ---------------------------------
    me = boards[0].current_player
    shipyards = me.shipyards
    shipyards_pos : List[Point]
    shipyards_pos = [s.position for s in shipyards] + [p.point for p in building_ships_send]
    # --------------------------------- Detect an Attack ---------------------------------------
    
    incoming_attack = detect_attack(shipyards_pos,boards)
    
    # ------------------- Notify that help is needed

    need_help = detect_help_needed(shipyards_pos,incoming_attack,boards)

    # ------------------- Help a shipyard in need
    
    action_shipyard = dict()
    for shipyard_pos in shipyards_pos:
        if shipyard_pos in need_help:
            help_needed = need_help[shipyard_pos]
            shipyards_sorted_by_distance = sorted(shipyards, key=lambda a: a.position.distance_to(shipyard_pos,size))
            for other_shipyard in shipyards_sorted_by_distance:
                ships_left = other_shipyard.ship_count
                plan = ""
                if other_shipyard.position not in need_help:
                    for (nb_step,nb_ships) in help_needed:
                        if ships_left >= 8 and nb_ships >= 1:
                            if other_shipyard.position.distance_to(shipyard_pos,size)==nb_step or (other_shipyard.position.distance_to(shipyard_pos,size)==nb_step-1 and random.random()<0.5) or (other_shipyard.position.distance_to(shipyard_pos,size)==nb_step-2 and random.random()<0.2):
                                ship_send = max(min(ships_left,nb_ships+10),8)
                                plan = compose(safe_plan_to_pos(boards,other_shipyard.position,shipyard_pos,ship_send,1))
                                if plan == "":
                                    ship_send = min(ships_left,nb_ships+40)
                                    plan = compose(safe_plan_to_pos(boards,other_shipyard.position,shipyard_pos,ship_send,1))
                                if plan != "":
                                    ships_left -= ship_send
                                    logger.info(f"Help found, incoming, {ship_send} ships sent")
                                    if ships_left>= nb_ships:
                                        help_needed.pop(0)
                                    else:
                                        help_needed[0] = (nb_step,nb_ships-ship_send)
                                    break
                                else:
                                    logger.info("No routes found to help :(")
                            elif other_shipyard.position.distance_to(shipyard_pos,size)< nb_step:
                                delta_t = nb_step-other_shipyard.position.distance_to(shipyard_pos,size)
                                possible_ship_number = space.get_shipyard_at_point(Point4d(other_shipyard.position,delta_t,None)).ship_count
                                if other_shipyard.position in incoming_attack:
                                    incoming_attack[other_shipyard.position].append((delta_t,possible_ship_number))
                                    incoming_attack[other_shipyard.position].sort()
                                else:
                                    incoming_attack[other_shipyard.position] = [(delta_t,possible_ship_number)]

                                help_needed[0] = (nb_step,nb_ships-possible_ship_number)
                                logger.info(f"Help found, incoming in {delta_t} turns, {possible_ship_number} ships")
                if plan != "":
                    action_shipyard[other_shipyard.id] = ShipyardAction.launch_fleet_with_flight_plan(other_shipyard.ship_count-ships_left, plan)
                    space.send_ghost_fleet(Point4d(other_shipyard.position,0,None),other_shipyard.ship_count-ships_left,plan,me.id)
    

    for shipyard in shipyards:
        action = None
        if shipyard.id in action_shipyard:
            action = action_shipyard[shipyard.id]
        incoming = []
        if shipyard.position in incoming_attack:
            incoming = incoming_attack[shipyard.position]
        # ----- builder
        nsatltas= shipyard.ship_count #number_ship_available_to_launch_to_another_shipyard
        for (limit_step,limit_ship_nb) in incoming:
            nsatltas -= limit_ship_nb
        if turn < 360:
            if action == None and kore_left>100*len(shipyards) and ((me_tot_ship_count >= ((73-2*(kore_left>1000)-3*(kore_left>3000)-4*(kore_left>1000)-3*(kore_left>2500))*(len(shipyards)+len(building_ships_send))+20*(kore_left<300)+40*(kore_left<100))) or ((me_tot_ship_count-50*len(building_ships_send) > biggest_ship_count_threat+10) and turn < 235)):
                if nsatltas >= convert_cost:
                    pos = best_pos_shipyard(boards,shipyard.id)
                    ship_number_to_send = max(convert_cost, int(shipyard.ship_count/2))
                    if pos != None:
                        if shipyard.id in me.shipyard_ids:
                            plan = safe_plan_to_pos(boards,shipyard.position,pos,ship_number_to_send,50)
                            if plan != []:
                                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(ship_number_to_send, compose(plan) + "C")
                                space.send_ghost_fleet(Point4d(shipyard.position,0,None),ship_number_to_send,compose(plan) + "C",me.id)
                                building_ships_send += [Point4d(pos,pos.distance_to(shipyard.position,size)+turn,None)]
                                action = shipyard.next_action
                elif (nsatltas+shipyard.max_spawn >= convert_cost) and kore_left >= shipyard.max_spawn*10*5:
                    nextt_action = "spawn"
        """ intercept useless
        nb_ships = shipyard.ship_count
        if action == None and turn >= 8:
            if nb_ships >= 21:
                possible_actions = dict()
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
        """
        if action == None:
            plan,nb_ships_to_send = astar3d(space,shipyard,my_vulnerabilities,op_vulnerabilities,power_center,prepare_attack,incoming)
            if plan != "":
                action = ShipyardAction.launch_fleet_with_flight_plan(nb_ships_to_send, plan)
                space.send_ghost_fleet(Point4d(shipyard.position,0,None),nb_ships_to_send,plan,me.id)

        if (action == None or nextt_action == "spawn") and (turn < 375 or me_tot_ship_count<biggest_ship_count_threat) and turn<390 and me_tot_ship_count<biggest_ship_count_threat*2:
            if kore_left >= spawn_cost:
                action = ShipyardAction.spawn_ships(int(min(shipyard.max_spawn,kore_left/spawn_cost)))
                kore_left -= int(min(shipyard.max_spawn,kore_left/spawn_cost))*spawn_cost
        shipyard.next_action = action
    logger.info(me.next_actions)
    saved_gravity2 = [slice.gravity2 for slice in space]
    return me.next_actions


if __name__ == "__main__":
    from kaggle_environments import make
    env = make("kore_fleets",debug = True)
    print(env.name, env.version)
    env.run([agent,balanced_agent])
