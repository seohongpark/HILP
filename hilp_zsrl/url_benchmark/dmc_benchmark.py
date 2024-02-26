from typing import List

DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    'point_mass_maze'
    'cheetah'
]

CHEETAH_TASKS = [
    'cheetah_walk',
    'cheetah_walk_backward',
    'cheetah_run',
    'cheetah_run_backward'
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

POINT_MASS_MAZE_TASKS = [
    'point_mass_maze_reach_top_left',
    'point_mass_maze_reach_top_right',
    'point_mass_maze_reach_bottom_left',
    'point_mass_maze_reach_bottom_right',
]



TASKS: List[str] = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + POINT_MASS_MAZE_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk'
}
