# Kore22 project
*By Julien Cardinal*

## Startup
Run the main.ipynb file that plays the main agent against the top agent of the beta version 
https://github.com/w9PcJLyb/kore-beta-bot. I don't really use his code for any other reasons than to challenge mine. Thanks to him, it was a really good baseline to improve my code !


# Cooperative A* 3d with gravity

The function astar3d finds the best plan *for a single shipyard* and the associated number of ships to send. It uses an heuristic function that depends on the gravity (used to pull the ships towards the high value kores and enemy fleets and shipyards) and the state of the current fleet.

## Generating the gravity
### Setting up the generators of the gravity
The principle is to put all important values in a 3 dimensions Space (size x size x number of predicted boards)
1. We first put the kore on each of the size x size shaped arrays that are modeling every boards at different timestamps. That way we get a list with the next ~30 kore maps.
2. We also put on it the enemy shipyard translated by a reward for taking them (depending on the max_spawn of the shipyard)
3. We finally take into account all the fleets at there position at each timestamp and put the reward for taking them on the list (depending on their kore and their ship_count) we also put a reward in the adjacent cells for attacking a fleet on its side

### The propagation of gravity
The idea is that the astar algorithm will only continue the routes where the heuristic has the best values. To attract the ship in a particular place, we need to artificially create a staircase of heuristic value increasing when we get closer to the high gravity points with the time increasing.

#### First attempt and ideas
We can think of doing some funnels with the gravity point at the bottom and maybe try a smoothing function in the x and y direction with t decreasing :

![Funel](docs/Funel.png)

We could do that by applying a 3 x 3 filter, maybe something like this :
```python
filter = [[0,1,0],
		  [1,1,1],
		  [0,1,0]]
# with every value divided by a certain value so that it doesn't diverge
```
Here the center value is calculated on the board at the starting point (where we want the gravity value), the ones directly adjacent are calculated from the next board (at **t+1**) at the corresponding position in x and y (as if it were on the same board)

All the test were done in a separate file (filter_test.py) with matplotlib to visualize the gravity on each layer
This is a result for a rather even filter and one gravity center that stays that way on every steps :

![even funel 3x3](docs/even_funel.png)

#### The best filter
The idea is that de diagonals are not important to pull a ship toward an objective, the fleets travel more efficiently in straight lines (either on the **x** axis or on the **y** axis)
Also a fleet can easily detect a good center of gravity by passing by a perpendicular line and having the gravity in diagonal points can clog the gravity map, whereas focusing on the lines makes routes for the fleet appear.

The selection criteria for a filter depends on many factors :
1. The viability factors
- The divergence/the stability in x and y axis, it's the influence from one single center of gravity on far away points. A bad stability shows immediately, the far away points start to oscillate with important values
- The bias factor adding a value at each apply of the filter : it is shown by the difference, for a stable single center of gravity, between the value of the center at t=0 and the few lats ts. This factor can be approximated by an homothetic function that can be calculate with only one point for a fixed filter and then apply on the rest of the board (this way of doing is not that important with a low bias filter)

2. The utility factors
- The values in diagonal directions must stay as near to 0 as possible, usually they are negative due to the nature of the filter shape
- The values in the straight lines must be decreasing from the source with linear pace, I chose to get close to something like [v v/2 v/3 v/4 v/5]

The best 3x3 filter for the job was not very viable as the bias was too important and the diagonals were too impacted, here's the shape of it :
![3x3 filter](docs/funel_3x3.png)

A 5x5 filter was necessary here and it was made so that we can have a maximum of zeros and reduce the temporal complexity.
![best_funel_5x5](docs/best_funel_5x5.png)

With such a filter it makes the routes appears on the map :
![Kore map](docs/Kore_map.png)

![Filtered kore map](docs/filtered_kore_map.png)

The filter :
```python
dimin = 1
c = -1.0 * dimin
l = -0.6 * dimin
d = 3.3 * dimin
e = 1.25 * dimin
ratio = 3.2/10

filter = [[0,l,e,l,0],
		  [l,c,d,c,l],
		  [e,d,0,d,e],
		  [l,c,d,c,l],
		  [0,l,e,l,0]]
```

## Generate the danger level
The danger level is a metric put on every cell, which states the minimum number of ships to go on a certain cell. The point for a fleet is to avoid dangerous cells, that are near an enemy shipyard, to avoid being easily taken by the enemy.

We put that in the space variable created (size x size x number of predicted boards).
-> From every enemy shipyard we put the corresponding danger level to all accessible cells (number of ships that can reach the cell), 

## Adding highways
Highways are not necessary for the algorithm to work well but they help reducing the number of cell visited : They are created at each shipyard in the four directions 
-> They only allow to detect the shipyards from further away

## The Pathfinding algorithm
### Coop A* 3d
For each shipyard we find the best move. There is no particular order, and maybe that can be upgraded. But for every shipyard we take into account the moves decided for the other (we refer to them as ghost_fleets), that's the part that is cooperative.
The best plan is found among a list of plans with a value for each of them.

We use the A* algorithm to find quickly the best routes. At every steps, the fleets choose the next cell for their paths according to the value giving to it : The value function is in fact the Heuristic of the pathfinding algorithm. Other values are taking into account to rule out certain path, judged too risky (danger_level), too long (if there is an incoming attack) or of no use (if it encounters an enemy shipyard/fleet that is too strong)

### The A* algorithm is really only about balance
The global result (for multiple steps) depends a lot on the utility of every ship launch. So we need to launch the best possible fleet for every situation. The problem is to converge quickly to the most useful paths.

The value function is there to give a value : how useful is a path. If it is negative, the path is seen as useless and we prefer spawning new ships instead. The point is to find the perfect behavior overall. Launching small fleets can be useful because they don't cost a lot to ship tho send but it also takes a turn during which we don't spawn a new fleet.

The exploration_cut is the limit value that a fleet is accepted, it forces the algorithm to find other paths 

Bonus_on_high is a bonus function that gives a boost on the first few cells for the fleets so that they explore a lot in the nearby cells to better find the global maximum (or a better local maximum)

This is all to make sure that we converge as quick as possible towards the best solution. That way we can actually manage capping the number of tiles visited to a small amount (maybe 100) and still have the best results. That will be very useful to improve time complexity.


# Time complexity