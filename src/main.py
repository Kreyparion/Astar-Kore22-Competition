from xmlrpc.client import Boolean
from kore_fleets import balanced_agent, random_agent, do_nothing_agent
from logger import logger, init_logger
from helpers import *
from helpers import Observation, Point, Direction, ShipyardAction, Configuration, Player, Shipyard, ShipyardId, Board, Fleet, FleetId
import random
import math
import time
from copy import deepcopy
from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt

external_timer = time.time()

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

class Index():
    def __init__(self, val: int, size):
        self._val = int(val)%(size*size)
        self.size = size
    def __add__(self, val):
        x,y = self._to_coord()
        if isinstance(val, Index):
            x2,y2 = val._to_coord()
            return Index._from_coord((x+x2,y+y2),self.size)
        return self._val + val
    def __iadd__(self, val):
        if isinstance(val, Index):
            self._val = self._val + val
            return self
        return self._val + val
    def __str__(self):
        return str(self._val)
    
    def __mul__(self, val: int):
        (x,y) = self._to_coord()
        return Index._from_coord((x*val,y*val),self.size)
    
    def __int__(self) -> int:
        return int(self._val)
    
    def to_coord(self)-> Tuple[int,int]:
        y, x = divmod(self._val, self.size)
        return (x, self.size - y - 1)
    
    @staticmethod
    def from_coord(t,size):
        (x,y) = t
        return Index((size - y % size - 1) * size + x % size,size)
    
    @staticmethod
    def from_dir(dir: Direction,size):
        if dir == Direction.NORTH:
            return Index(-2*size,size)
        if dir == Direction.SOUTH:
            return Index(0,size)
        if dir == Direction.EAST:
            return Index(-size+1,size)
        if dir == Direction.WEST:
            return Index(-1,size)

    def translate_n(self, dir: Direction, times: int):
        return self + Index._from_dir(dir,self.size) * times

    def translate_n_m(self, dir1: Direction, n: int, dir2: Direction, m: int):
        i = self._translate_n(dir1,n)
        return i._translate_n(dir2,m)

    _from_dir = from_dir
    _translate_n = translate_n
    _from_coord = from_coord
    _to_coord = to_coord

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

@Timer
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

def predicts_next_boards(obs,config,n=40,my_first_acctions=None,my_agent=spawn_agent,op_agent=spawn_agent):
    board = Board(obs, config)
    me = board.current_player
    new_observation = deepcopy(obs)
    boards = [board]
    ops = board.opponents
    for i in range(n):
        new_actions: List[Dict[str,str]]
        new_actions = [dict()]*len(new_observation['players'])
        new_obs = deepcopy(new_observation)
        if i == 0:
            if my_first_acctions != None:
                new_actions[me.id] = my_first_acctions
            else:
                new_actions[me.id] = my_agent(new_observation,config)
        else:
            new_obs['player'] = me.id
            new_actions[me.id] = my_agent(new_obs,config)
        for op in ops:
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
            if boards[i].get_fleet_at_point(p).ship_count > fleet.ship_count:
                return endangered(boards,boards[i].get_fleet_at_point(p))
            else:
                return True
    return True

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
        res += generate_all_plans_in_2_directions(nb_moves-1, t2-1,dir2,t1,dir1, add_to_all(plan,dir1.to_char()), last_move=False)            
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

@Timer
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
    convert_cost = board.configuration.convert_cost
    spawn_cost = board.configuration.spawn_cost
    me = board.current_player
    if shipyard_id not in me.shipyard_ids:
        return 0
    shipyard_pos = board.shipyards[shipyard_id].position
    nb_moves = maximum_flight_plan_length(ship_number_to_send)
    all_possible_plans = generate_all_plans_A_to_B(boards,shipyard_pos,pos,nb_moves-1)
    nb_ships = ship_number_to_send
    best_plan = []
    best_global_reward = -700
    for plan in all_possible_plans:
        global_reward,_,nb_ships_left = evaluate_plan(boards,plan,pos,nb_ships,spawn_cost,size)
        if nb_ships_left>= convert_cost and global_reward > best_global_reward:
            best_plan = plan
            best_global_reward = global_reward
    return best_plan


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
    liste_ops_shipyards_pos = list_ops_shipyards_position(board)
    liste_my_shipyards_pos = [shipyard.position for shipyard in me.shipyards]
    best_pos = None
    best_kore = 0
    for i in range(2,7):
        for j in range(2,7):
            if 10 >= i+j >= 3:
                for k in range(4):
                    time_board = boards[-1]
                    if i+j < len(boards):
                        time_board = boards[i+j]
                    possible_pos = translate_n_m(s,Direction.from_index(k),i,Direction.from_index(k).rotate_right(),j,size)
                    distance_to_closest_ops_shipyard,_ = distance_to_closest(possible_pos, liste_ops_shipyards_pos, size)
                    distance_to_my_closest_shipyard,_ = distance_to_closest(possible_pos,liste_my_shipyards_pos,size)
                    if distance_to_closest_ops_shipyard >= i+j and distance_to_my_closest_shipyard >= 3 and distance_to_closest_ops_shipyard<distance_to_my_closest_shipyard-1:
                        potential_kore = matrix_kore(time_board,matrix,possible_pos,size)
                        if potential_kore > best_kore:
                            best_kore = potential_kore
                            best_pos = possible_pos
    return best_pos

class Slice:
    def __init__(self, board: Board) -> None:
        self.gravity : Dict[Point,float]
        self.gravity = dict()
        self.danger_zone : Dict[Point,int]
        self.danger_zone = dict()
        self.board = board
        self.me = board.current_player
        
    def add_gravity(self: 'Slice', position: Point, val: float) -> None:
        if position in self.gravity:
            self.gravity[position] += val
        else:
            self.gravity[position] = val

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
        
    def get_gravity_at_point(self: 'Slice', position: Point) -> float:
        if position in self.gravity:
            return self.gravity[position]
        else:
            return 0
    
    def get_danger_level_at_point(self: 'Slice', position: Point) -> int:
        if position in self.danger_zone:
            return self.danger_zone[position]
        else:
            return 0
    
    def get_cell_at_point(self, position: Point):
        return self.board.get_cell_at_point(position)
    
    def get_fleet_at_point(self,position : Point):
        return self.board.get_fleet_at_point(position)
    
    def get_shipyard_at_point(self, position: Point):
        return self.board.get_shipyard_at_point(position)
    
    def get_potential_at_point(self,position: Point) -> float:
        cell = self.get_cell_at_point(position)
        if cell == None:
            return 0
        else:
            fleet_kore = 0
            fleet = cell.fleet
            if fleet != None and fleet.player_id != self.me.id:
                fleet_kore = fleet.kore
            shipyard_max_spawn = 0
            shipyard = cell.shipyard
            if shipyard != None and shipyard.player_id != self.me.id:
                shipyard_max_spawn = shipyard.max_spawn
            return cell.kore + 2*fleet_kore + 50*shipyard_max_spawn
    
    def get_kore_at_point(self,position: Point):
        cell = self.get_cell_at_point(position)
        if cell == None:
            return 0
        else:
            return cell.kore

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
        self.space = []
        for i in range(len(boards)):
            self.space.append(Slice(boards[i]))
        self.ghost_fleets : Dict[Point4d,Fleet]
        self.ghost_fleets = dict()
    
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

    def get_shipyard_at_point(self, point4d: Point4d):
        return self.space[point4d.t].get_shipyard_at_point(point4d.point)
    
    def get_kore_at_point(self, point4d: Point4d):
        return self.space[point4d.t].get_kore_at_point(point4d.point)
    
    def get_potential_at_point(self, point4d: Point4d):
        return self.space[point4d.t].get_potential_at_point(point4d.point)
    
    def get_gravity_at_point(self,point4d: Point4d):
        return self.space[point4d.t].get_gravity_at_point(point4d.point)

    def add_gravity(self,point4d: Point4d, val: float):
        return self.space[point4d.t].add_gravity(point4d.point, val)
    
    
    def show_gravity(self,size):
        res1 = [[0 for _ in range(size)] for _ in range(size)]
        res2 = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                pos = Point(size - 1 - j, i)
                point = Point4d(pos,0,None)
                res1[i][j] = self.get_gravity_at_point(point)
                res2[i][j] = self.get_kore_at_point(point)
        plt.matshow(res1)
        plt.matshow(res2)
        plt.show()
        logger.info(f"{res2}")
    
    def get_danger_level_at_point(self,point4d: Point4d):
        return self.space[point4d.t].get_danger_level_at_point(point4d.point)

    def add_danger_level_at_point(self,point4d: Point4d, val: int):
        return self.space[point4d.t].add_danger_level_at_point(point4d.point, val)
    
    def show_danger_level(self,size):
        res = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                pos = Point(size - 1 - j, i)
                point = Point4d(pos,12,None)
                res[i][j] = self.get_danger_level_at_point(point)
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

                    new_kore = ghost_fleet.kore + self.get_kore_at_point(pos)*ratio_kore_ship_count(nb_ships_left)
                    ghost_fleet = Fleet(FleetId("ghost"),nb_ships_left,pos.d,pos.point,new_kore,plan,player_ID,self.space[pos.t].board)
                    self.ghost_fleets[pos] = ghost_fleet

                    i += 1

@Timer
def ApplyGravitySlow(space: Space):
    # --- first anti gravity with shipyards
    """
    for t in range(1,len(space)):
        slice = space[t]
        ops = slice.opponents
        ops_shipyards = []
        for op in ops:
            for op_shipyard in op.shipyards:
                for mt in range(t-1,-1,-1):
                    mslice = space[mt]
    """
    size = space[0].board.configuration.size
    for t in range(len(space)-1,0,-1):
        for i in range(size):
            for j in range(size):
                pos = Point(j, size - 1 - i)
                curr_kore = space.get_kore_at_point(Point4d(pos,t,None))
                if curr_kore > 10:
                    for nt in range(t-1,-1,-1):
                        curr_kore = curr_kore/2
                        if curr_kore > 1:
                            for l in range(4):
                                d = Direction.from_index(l)
                                dp = d.rotate_left()
                                for k in range(t-nt):
                                    new_pos = translate_n_m(pos,d,k,dp,t-nt-k,size)
                                    space.add_gravity(Point4d(new_pos,t-k-1,None),curr_kore)

@Timer
def ApplyDangerLevel(space: Space):
    size = space[0].board.configuration.size
    for t in range(len(space)):
        if t==0:
            continue
        slice = space[t]
        ops = slice.board.opponents
        for op in ops:
            for op_shipyard in op.shipyards:
                pos = op_shipyard.position
                danger_lvl = op_shipyard.ship_count
                if danger_lvl > 2:
                    for nt in range(10):
                        for l in range(4):
                            d = Direction.from_index(l)
                            dp = d.rotate_left()
                            for k in range(nt):
                                new_pos = translate_n_m(pos,d,k,dp,nt-k,size)
                                if t+nt>len(space)-1:
                                    space.add_danger_level_at_point(Point4d(new_pos,len(space)-1,None),danger_lvl)
                                else:
                                    space.add_danger_level_at_point(Point4d(new_pos,t+nt,None),danger_lvl)
        if t == 1:
            continue
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
                                space.add_danger_level_at_point(Point4d(new_pos,len(space)-1,None),-danger_lvl)
                            else:
                                space.add_danger_level_at_point(Point4d(new_pos,t+nt,None),-danger_lvl)
    
    # space.show_danger_level(size)

@Timer
def ApplyGravity(space: Space):
    size = space[0].board.configuration.size
    me = space[0].board.current_player
    dimin = 1
    cc = -0.2 * dimin
    cd = -1 * dimin
    ld = -0.5 * dimin
    c = -0.8 * dimin
    d = 3 * dimin
    dd = 1.4 * dimin
    ratio = 4

    filter = [[cc,ld,dd,ld,cc],
            [ld,cd,d,cd,ld],
            [dd,d,c,d,dd],
            [ld,cd,d,cd,ld],
            [cc,ld,dd,ld,cc]]
    
    # ---- Gravity for the enemy fleets and shipyards
    
    for n in range(len(space)):
        nops = space[n].board.opponents
        for op in nops:
            for fleet in op.fleets:
                pos = Point4d(fleet.position,n,None)
                space.add_gravity(pos,3*fleet.kore)
                for i in range(4):
                    d = Direction.from_index(i)
                    posi = pos.translate(d,size)
                    space.add_gravity(posi,fleet.kore)
            for shipyard in op.shipyards:
                space.add_gravity(Point4d(shipyard.position,n,None),10*shipyard.max_spawn+10*shipyard.max_spawn*shipyard.max_spawn+300)
    for n in range(len(space)-1,-1,-1):
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
                        sum += space.get_gravity_at_point(Point4d(Point(x,y),t,None))*filter[k][l]
                point = Point4d(Point(i,j),n,None)
                space.add_gravity(Point4d(Point(i,j),n,None),(sum + space.get_kore_at_point(point)*ratio)/10)
    


    # space.show_gravity(size)

def where_to_capacity(space: Space,t: int, fleet: Fleet):
    fleet_id = fleet.id
    size = space[t].board.configuration.size
    player_id = space[t].board.current_player_id
    i = t
    while i < len(space)-1 and fleet_id in space[i].board.fleets:
        i += 1
    if i == 0:
        return 0 # The fleet died so it's not in danger
    if i >= len(space)-1:
        return -1
    real_ship_count = space[i-1].board.fleets[fleet_id].ship_count
    p = Point4d(next_position(space[i-1].board.fleets[fleet_id],size),i,None)
    if space.get_shipyard_at_point(p) != None:
        if space.get_shipyard_at_point(p).player_id == player_id:
            return 100000
    if space.get_fleet_at_point(p) != None:
        if space.get_fleet_at_point(p).player_id == player_id:
            if space.get_fleet_at_point(p).ship_count > 2*real_ship_count:
                cap = where_to_capacity(space,i,space.get_fleet_at_point(p))
                if cap == -1 or cap == 0:
                    return -1
                return min(space.get_fleet_at_point(p).ship_count-2*real_ship_count-1,cap)
            else:
                return -1
    return -1

def endangered_space(space: Space,t: int, fleet: Fleet):
    fleet_id = fleet.id
    size = space[t].board.configuration.size
    player_id = space[t].board.current_player_id
    i = t
    while i < len(space)-1 and fleet_id in space[i].board.fleets:
        i += 1
    if i == 0:
        return False # The fleet died so it's not in danger
    if i >= len(space)-1:
        return True
    p = Point4d(next_position(space[i-1].board.fleets[fleet_id],size),i,None)
    if space.get_shipyard_at_point(p) != None:
        if space.get_shipyard_at_point(p).player_id == player_id:
            return False
    if space.get_fleet_at_point(p) != None:
        if space.get_fleet_at_point(p).player_id == player_id:
            if space.get_fleet_at_point(p).ship_count > fleet.ship_count:
                return endangered_space(space,i,space.get_fleet_at_point(p))
            else:
                return True
    return True



def value(elmt):
    (r,lr,p,cs,ms,Ms,ln,tplan) = elmt
    plan,lmoves = tplan
    return r+lr-3*ms-0.4*ms*ln-4*(len(plan)+lmoves//10)-8*ln-ln*ln*0.2-5*(ms-cs)

def sortedAppend(liste, element):
    liste.append(element)
    i = len(liste)-1
    while i > 0 and value(liste[i]) < value(liste[i-1]):
        liste[i], liste[i-1] = liste[i-1], liste[i]
        i -= 1

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

def astar3d(space: Space, shipyard: Shipyard, incoming_attacks :List[Tuple[int,int]] = [])->Tuple[str,int]:
    board = space[0].board
    size = board.configuration.size

    me = board.current_player
    """
    ops = board.opponents
    ops_shipyards = []
    for op in ops:
        for op_shipyard in op.shipyards:
            ops_shipyards.append(op_shipyard)
    """
    ms = 2
    if shipyard.max_spawn > 5:
        if me.kore > 50:
            ms = 3
        if me.kore > 400:
            ms = 5
    if shipyard.max_spawn > 6:
        if me.kore > 60:
            ms = 3
        if me.kore > 120:
            ms = 5
        if me.kore > 300:
            ms = 8
    if shipyard.max_spawn > 7:
        if me.kore > 70:
            ms = 5
        if me.kore > 150:
            ms = 8
        if me.kore > 400:
            ms = 13
    if shipyard.max_spawn > 8:
        if me.kore > 80:
            ms = 8
        if me.kore > 200:
            ms = 13
        if me.kore > 500:
            ms = 21
    if shipyard.max_spawn > 9:
        if me.kore > 90:
            ms = 13
        if me.kore > 250:
            ms = 21
        if me.kore > 1500:
            ms = 34
        if me.kore > 5000:
            ms = 55
    if shipyard.max_spawn > 10:
        if me.kore > 100:
            ms = 13
        if me.kore > 200:
            ms = 21
        if me.kore > 1000:
            ms = 34
        if me.kore > 4000:
            ms = 55
    
    priority_list = []
    if shipyard.ship_count >= 2 and shipyard.ship_count >= ms:
        point = Point4d(shipyard.position,0,None)
        priority_list = [(0,space.get_gravity_at_point(point),point,ms,ms,shipyard.ship_count,0,("",0))]# [(reward,local_reward,point4d,nb_current_ships,nb_ships_min_needed,nb_ships_max_possible,nb_steps,tplan)]
    best_plan : List[Tuple[float,str,int]]
    best_plan = []
    while priority_list != [] and len(best_plan) < 5:
        (r,lr,p,cs,ms,Ms,ln,tplan) = priority_list.pop()
        ps = ms-cs # number of lost ships
        plan,lmoves = tplan
        # logger.info(f"plan : {plan+str(lmoves)}, value : {value((r,lr,p,cs,ms,Ms,ln,tplan))}")
        nshipyard = space.get_shipyard_at_point(p)
        nfleet = space.get_fleet_at_point(p)
        if nshipyard != None and plan != "":
            if nshipyard.player_id == me.id:
                if plan not in ["NS","EW","WE","SN"]:
                    logger.info(f"to shipyard : {plan}")
                    best_plan.append((value((r,15-nshipyard.max_spawn*nshipyard.max_spawn*0.8,p,cs,ms,Ms,ln,tplan)),plan,ms))
                continue
            else:
                if nshipyard.ship_count >= Ms+cs-ms:
                    continue
                if nshipyard.ship_count >= cs:
                    diff = nshipyard.ship_count + 1-cs
                    cs = 1
                    ms = ms + diff
                    logger.info(f"to enemy shipyard : {plan}")
                    best_plan.append((value((r,0,p,cs,ms,Ms,ln,tplan)),plan,ms))
                    continue
        if nfleet != None:
            if nfleet.player_id == me.id:
                if lr == 0:
                    logger.info(f"to a fleet : {plan}")
                    best_plan.append((value((r,0,p,cs,ms,Ms,ln,tplan)),plan,ms))
                    continue
                
        max_move = maximum_flight_plan_length(Ms)
        if ln >= len(space)-1:
            continue
        (sr,slr,sp,scs,sms,sMs,sln,stplan) = (r,lr,p,cs,ms,Ms,ln,tplan)
        for i in range(4):
            (r,lr,p,cs,ms,Ms,ln,tplan) = (sr,slr,sp,scs,sms,sMs,sln,stplan)
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
            kore = space.get_kore_at_point(new_p)
            fleet = space.get_fleet_at_point(new_p)
            gravity = space.get_gravity_at_point(new_p)
            fleet_kore = 0
            if fleet != None:
                if fleet.player_id != me.id:
                    if fleet.ship_count >= Ms+cs-ms:
                        continue
                    if fleet.ship_count >= cs:
                        diff = fleet.ship_count + 1-cs
                        cs = 1
                        ms = ms + diff
                    fleet_kore += 3*fleet.kore
                else:
                    max_capacity_fleet = where_to_capacity(space,ln,fleet)
                    #if max_capacity_fleet != -1 and max_capacity_fleet != 0 and max_capacity_fleet != 100000:
                    #    logger.info(f"{max_capacity_fleet}")
                    if max_capacity_fleet == -1:
                        if fleet.ship_count < Ms+cs-ms:    # Get the fleet
                            fleet_kore += fleet.kore + 10*fleet.ship_count
                            if fleet.ship_count >= cs:
                                diff = fleet.ship_count + 1 - cs
                                cs = 1
                                ms = ms + diff
                            else:
                                continue
                        else:
                            continue
                    else:
                        if cs < max_capacity_fleet and (cs < fleet.ship_count):
                            gravity = 0
                        elif fleet.ship_count < cs:
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
                            Ms = ms + 1 # = continue
                        if fleetj.ship_count >= cs:
                            diff = fleetj.ship_count + 1-cs
                            cs = 1
                            ms = ms + diff
                        fleet_kore += fleetj.kore
            nshipyard = space.get_shipyard_at_point(new_p)
            if nshipyard != None:
                if nshipyard.player_id != me.id:
                    if nshipyard.ship_count >= Ms+cs-ms:
                        continue
                    if nshipyard.ship_count >= cs:
                        diff = nshipyard.ship_count + 1-cs
                        cs = 1
                        ms = ms + diff
                        fleet_kore += 10*shipyard.max_spawn+10*shipyard.max_spawn*shipyard.max_spawn+300
                        gravity = 0
                else:
                    gravity = 15-nshipyard.max_spawn*nshipyard.max_spawn*0.8
            danger_lvl = space.get_danger_level_at_point(new_p)
            if danger_lvl >= Ms+cs-ms:
                continue
            elif danger_lvl >= cs:
                diff = danger_lvl +1 -cs
                cs = cs + diff
                ms = ms + diff
            nb_ships_needed = 0
            for (limit_step,limit_ship_nb) in incoming_attacks:
                nb_ships_needed += limit_ship_nb
                if ln >= limit_step:
                    if Ms-ms <= nb_ships_needed:
                        ms = Ms +1 # to end up with continue
                        break
            
            if ms > Ms:
                continue
            new_elmt = (r+kore+fleet_kore,gravity,new_p,cs,ms,Ms,ln+1,(new_plan,new_lmoves))
            if value(new_elmt) < -30+3*ln:
                continue
            sortedAppend(priority_list,new_elmt)
            
    logger.info(f"{best_plan}")
    if best_plan == []:
        return ("",0)
    best = max(best_plan)
    if best[0]>0:
        space.send_ghost_fleet(point,best[2],best[1],me.id)
        return (best[1],best[2])
    return ("",0)

def detect_attack(boards : List[Board]) -> Dict[ShipyardId,List[Tuple[int,int]]]:
    """

    Args:
        boards (List[Board]): A list of board

    Returns:
        Dict[ShipyardId,List[Tuple[int,int]]]: a dictionnary that, for each ShipyardID
        reffers to a list of incoming attacks on the shipyard :
        the format is : (nb of turn to the attack, nb of ships attacking)
    """
    board = boards[0]
    size = board.configuration.size
    me = board.current_player
    shipyards = me.shipyards
    shipyard_next_pos = []
    for shipyard in shipyards:
        for i in range(4):
            d = Direction.from_index(i)
            pos = shipyard.position
            new_pos = translate_n(pos,d,1,size)
            shipyard_next_pos.append(new_pos)
    in_danger: Dict[ShipyardId,List[Tuple[int,int]]]
    in_danger = dict()
    for i,nboard in enumerate(boards):
        nops = nboard.opponents
        op_fleets = concat([[fleet for fleet in op.fleets] for op in nops])
        for fleet in op_fleets:
            if fleet.position in shipyard_next_pos:
                next_pos = next_position(fleet,size)
                if next_pos != None:
                    s = shipyard_next_pos.index(fleet.position)//4
                    if nboard.get_shipyard_at_point(next_pos) != None:
                        if nboard.get_shipyard_at_point(next_pos).id == shipyards[s].id:
                            if shipyards[s].id in in_danger:
                                add = 1
                                for j,(t,nb) in enumerate(in_danger[shipyards[s].id]):
                                    if i == t:
                                        in_danger[shipyards[s].id][j] = (i,nb+fleet.ship_count)
                                        add = 0
                                if add:
                                    in_danger[shipyards[s].id] += [(i,fleet.ship_count)]
                            else:
                                in_danger[shipyards[s].id] = [(i,fleet.ship_count)]
    return in_danger
        

def agent2(obs: Observation, config: Configuration):
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
        if action == None and remaining_kore > 300:
            if shipyard.ship_count >= convert_cost + 20:
                pos = best_pos_shipyard(boards,shipyard.id)
                ship_number_to_send = max(convert_cost + 20, int(shipyard.ship_count/2))
                if pos != None:
                    plan = safe_plan_to_pos(boards,shipyard.id,pos,ship_number_to_send)
                    if plan != []:
                        shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(ship_number_to_send, compose(plan) + "C")
                        action = shipyard.next_action
        # ---- invader
        """
        if action == {}:
            action = None
        if action == None and shipyard.ship_count >= 100 and remaining_kore > 500:
            dist,p_op_shipyard = distance_to_closest(shipyard.position, list_ops_shipyards_position(board), size)
            if dist < len(boards):
                if board.get_shipyard_at_point(p_op_shipyard).ship_count < 100:
                    path = get_shortest_flight_path_between(shipyard.position,p_op_shipyard,size)
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(board.get_shipyard_at_point(p_op_shipyard).ship_count + 23,path)
                    action = shipyard.next_action
        """
        if action == {}:
            action = None
        if action == None:
            chance_to_summon = max(1-(kore_left>100)*0.1-math.sqrt(kore_left)*0.01- (nb_ships < 29)*0.3,0)
            if random.random()<chance_to_summon:
                if nb_ships >= 8 + 21*(1-no_endangered):
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
    convert_cost = board.configuration.convert_cost
    kore_left = me.kore
    ops = board.opponents
    size = board.configuration.size
    nb_building_ships_send = 0
    me_tot_ship_count = tot_ship_count(me,board)
    biggest_ship_count_threat = max([tot_ship_count(op,board) for op in ops])
    
        
    # --------------------------------- Predict the next Boards ---------------------------------
    boards = predicts_next_boards(obs,config)
    logger.info("boards generated")
    
    # --------------------------------- Generate Gravity for all ships ----------------------
    space = Space(boards)
    ApplyDangerLevel(space)
    ApplyGravity(space)
    logger.info("gravity added")
    """
    if obs.step%10 == 0:
        space.show_gravity(size)
    """
    # --------------------------------- Detect an Attack ---------------------------------------
    
    incoming_attack = detect_attack(boards)
    
    
    # --------------------------------- Choose a move ---------------------------------
    shipyards = me.shipyards
    need_help = dict()
    
    for shipyard in shipyards:
        if shipyard.id in incoming_attack:
            all_incoming_attacks = deepcopy(incoming_attack[shipyard.id])
            for i,(nb_step,nb_ships) in enumerate(all_incoming_attacks):
                if nb_step == 0:
                    continue
                if i == 0:
                    iboard = boards[nb_step-1]
                    ime = iboard.current_player
                    ishipyard = iboard.shipyards[shipyard.id]
                    if nb_ships > ishipyard.ship_count:
                        nb_step_to_build_kore = (nb_ships - ishipyard.ship_count)//ishipyard.max_spawn+1
                        if nb_step < nb_step_to_build_kore or ((nb_step > 3 or nb_step<nb_step_to_build_kore//2) and kore_left<nb_step_to_build_kore*spawn_cost):
                            need_help[shipyard.id] = [(nb_step,nb_ships-ishipyard.ship_count)]
                            logger.info(f"need help !! Need {nb_ships-ishipyard.ship_count} ships in {nb_step} turns")
                            incoming_attack[shipyard.id][0] = (max(min(nb_step-nb_step_to_build_kore,0),nb_step-3),nb_ships)
                        else:
                            incoming_attack[shipyard.id][0] = (nb_step-nb_step_to_build_kore,nb_ships)
                else:
                    iboard = boards[nb_step-1]
                    ime = iboard.current_player
                    ishipyard = iboard.shipyards[shipyard.id]
                    if ishipyard.player_id == ime.id:
                        if nb_ships > ishipyard.ship_count:
                            nb_step_to_build_kore = (nb_ships - ishipyard.ship_count)//spawn_cost+1
                            if nb_step < nb_step_to_build_kore or ((nb_step > 3 or nb_step<nb_step_to_build_kore//2) and kore_left<nb_step_to_build_kore*spawn_cost):
                                if shipyard.id in need_help:
                                    need_help[shipyard.id] += [(nb_step,nb_ships-ishipyard.ship_count)]
                                    logger.info(f"need 2nd help !! Need {nb_ships-ishipyard.ship_count} ships in {nb_step} turns")
                                else:
                                    need_help[shipyard.id] = [(nb_step,nb_ships-ishipyard.ship_count)]
                                    logger.info(f"need help for the others !! Need {nb_ships-ishipyard.ship_count} ships in {nb_step} turns")
                                incoming_attack[shipyard.id][i] = (max(min(nb_step-nb_step_to_build_kore,0),nb_step-3),nb_ships)
                            else:
                                incoming_attack[shipyard.id][i] =(nb_step-nb_step_to_build_kore,nb_ships)
                    else:
                        if shipyard.id in need_help:
                            need_help[shipyard.id] += [(nb_step,nb_ships)]
                        else:
                            need_help[shipyard.id] = [(nb_step,nb_ships)]

    action_shipyard = dict()
    for shipyard in shipyards:
        if shipyard.id in need_help:
            help_needed = need_help[shipyard.id]
            shipyards_sorted_by_distance = sorted(shipyards, key=lambda a: a.position.distance_to(shipyard.position,size))
            for other_shipyard in shipyards_sorted_by_distance:
                ships_left = other_shipyard.ship_count
                plan = ""
                if other_shipyard.id not in need_help:
                    for (nb_step,nb_ships) in help_needed:
                        if ships_left >= 8:
                            if other_shipyard.position.distance_to(shipyard.position,size)==nb_step:
                                ship_send = min(ships_left,nb_ships)
                                plan = compose(safe_plan_to_pos(boards,other_shipyard.id,shipyard.position,ship_send))
                                ships_left -= ship_send
                                logger.info(f"Help found, incoming, {ship_send} ships sent")
                                if ships_left>= nb_ships:
                                    help_needed.pop(0)
                                else:
                                    help_needed[0] = (nb_step,nb_ships-ship_send)
                            elif other_shipyard.position.distance_to(shipyard.position,size)< nb_step:
                                delta_t = nb_step-other_shipyard.position.distance_to(shipyard.position,size)
                                possible_ship_number = space.get_shipyard_at_point(Point4d(other_shipyard.position,delta_t,None)).ship_count
                                if other_shipyard.id in incoming_attack:
                                    incoming_attack[other_shipyard.id].append((delta_t,possible_ship_number))
                                    incoming_attack[other_shipyard.id].sort()
                                else:
                                    incoming_attack[other_shipyard.id] = [(delta_t,possible_ship_number)]

                                help_needed[0] = (nb_step,nb_ships-possible_ship_number)
                                logger.info(f"Help found, incoming in {delta_t} turns, {possible_ship_number} ships")
                if plan != "":
                    action_shipyard[other_shipyard.id] = ShipyardAction.launch_fleet_with_flight_plan(other_shipyard.ship_count-ships_left, plan)


    for shipyard in shipyards:
        action = None
        if shipyard.id in action_shipyard:
            action = action_shipyard[shipyard.id]
        incoming = []
        if shipyard.id in incoming_attack:
            incoming = incoming_attack[shipyard.id]
        # ----- builder

        if action == None and kore_left > 300 and incoming == [] and me_tot_ship_count-50*nb_building_ships_send > biggest_ship_count_threat-50:
            if shipyard.ship_count >= convert_cost + 20:
                pos = best_pos_shipyard(boards,shipyard.id)
                ship_number_to_send = max(convert_cost + 20, int(shipyard.ship_count/2))
                if pos != None:
                    plan = safe_plan_to_pos(boards,shipyard.id,pos,ship_number_to_send)
                    if plan != []:
                        shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(ship_number_to_send, compose(plan) + "C")
                        space.send_ghost_fleet(Point4d(shipyard.position,0,None),ship_number_to_send,compose(plan) + "C",me.id)
                        nb_building_ships_send += 1
                        action = shipyard.next_action

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
            plan,nb_ships_to_send = astar3d(space,shipyard,incoming)
            if plan != "":
                action = ShipyardAction.launch_fleet_with_flight_plan(nb_ships_to_send, plan)

        if action == None:
            if kore_left >= spawn_cost:
                action = ShipyardAction.spawn_ships(int(min(shipyard.max_spawn,kore_left/spawn_cost)))
                kore_left -= int(min(shipyard.max_spawn,kore_left/spawn_cost))*spawn_cost
        shipyard.next_action = action
    logger.info(me.next_actions)
    return me.next_actions



if __name__ == "__main__":
    from kaggle_environments import make

    env = make("kore_fleets", debug=True)
    print(env.name, env.version)

    env.run([agent,balanced_agent])
