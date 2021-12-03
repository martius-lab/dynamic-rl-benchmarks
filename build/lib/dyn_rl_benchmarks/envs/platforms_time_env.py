import numpy as np
from gym import spaces

from . import PlatformsEnv


class TimedGoal:
    def __init__(self, goal, delta_t_ach, delta_t_comm = None):
        self.goal = goal
        self.delta_t_ach = delta_t_ach
        self.delta_t_comm = delta_t_comm

    @staticmethod
    def from_dict(d):
        return TimedGoal(d["desired_goal"], d["delta_t_ach"][0], 
                d["delta_t_comm"][0] if "delta_t_comm" in d else d["delta_t_ach"][0])


class PlatformsTimeEnv(PlatformsEnv):  

    def __init__(self, max_episode_length = 500, subgoal_radius = 0.05):
        super().__init__(max_episode_length, subgoal_radius)

        # define spaces by adding time to obs
        desired_goal_space = spaces.Box(
                low = -1., 
                high = 1., 
                shape = (2,),
                dtype = np.float32)
        achieved_goal_space = desired_goal_space
        obs_space_dict = {
            "position": spaces.Box(low = -1., high = 1., shape = (2,), dtype = np.float32),
            "velocity": spaces.Box(low = -1., high = 1., shape = (2,), dtype = np.float32),
            "ang_vel": spaces.Box(low = -1., high = 1., shape = (1,), dtype = np.float32),
            "time": spaces.Box(low = -1., high = 1., shape = (1,), dtype = np.float32)
                }
        obs_platforms_dict = {
                "platform{}".format(i): spaces.Box(
                    low = np.array([-1., -1., -np.inf, -np.inf]), 
                    high = np.array([1., 1., np.inf, np.inf]), 
                    dtype = np.float32) for i in range(len(self.kinematic_rects))}
        obs_space_dict.update(obs_platforms_dict)
        obs_space = spaces.Dict(obs_space_dict)


        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "desired_goal": desired_goal_space,
            "achieved_goal": achieved_goal_space
            })

        self.reset()

    def _get_obs(self):
        obs = super()._get_obs()
        obs["observation"]["time"] = np.array([(self.current_step/float(self.max_episode_length) - 0.5)*2.0])
        return obs

    def update_subgoals(self, subgoals):
        tolerances = [None]*len(subgoals)
        timed_subgoals = []
        for sg in subgoals:
            actual_sg = {k:sg[k] for k in sg if k != "time"}
            t_sg = (0.5*sg["time"] + 0.5)*self.max_episode_length
            dt_sg = max(t_sg - self.current_step, 0.0)
            timed_subgoals.append(TimedGoal(actual_sg, dt_sg, dt_sg))
        self.update_timed_subgoals(timed_subgoals, tolerances)
