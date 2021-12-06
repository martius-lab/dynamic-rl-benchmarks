from copy import copy
import time
import  os

import numpy as np
import numpy.linalg as linalg
import gym
from gym import spaces
from gym.utils import seeding

from roboball2d.physics import B2World
from roboball2d.robot import DefaultRobotConfig
from roboball2d.robot import DefaultRobotState
from roboball2d.ball import BallConfig
from roboball2d.ball_gun import DefaultBallGun
from roboball2d.utils import Box


class Tennis2DEnv(gym.GoalEnv):
    """2D Toy Robotic Tennis Environment.

    Task: 2D robot with 3 degrees of freedom has to return tennis ball to given
    goal landing point by hitting it appropriately. A sparse reward is given 
    only if the ball lands within the goal region."""

    metadata = {"render.mode": ["human"]}

    # This version of the environment uses a sparse reward
    dense_reward = False

    # goal properties
    _goal_min = 3.0
    _goal_max = 6.0
    _goal_diameter = 0.8
    _goal_color = (0.2, 0.7, 0.0)

    # variables for reward design
    _hit_reward = 2.0

    # maximum episode length in seconds
    _max_episode_length_sec = 5.0

    # divide by this attribute to normalize angles
    _angle_normalization = 0.5*np.pi

    # maximum angular velocity
    _max_angular_vel = 8.0

    # safety factor (for joint limits because solver 
    # can't ensure that they are always satisfied)
    _safety_factor = 1.3

    # simulation steps per second
    _steps_per_sec = 100

    _arrow_width = 0.02
    _arrow_head_size = 0.06
    _arrow_scaling = 0.3

    def __init__(self, slow_motion_factor = 2.0):
        super().__init__()

        self._subgoals = []
        self._timed_subgoals = []
        self._tolerances = None
        self._subgoal_colors = []

        # maximum episode length in steps
        self.max_episode_length = int(self._max_episode_length_sec*self._steps_per_sec)

        self.seed()
        self.verbose = 0
        self._slow_motion_factor = slow_motion_factor

        self._renderer = None
        self._callbacks = []

        #####################################
        # Physics simulation using Roboball2D
        #####################################

        # robot and ball configuration
        self._robot_config = DefaultRobotConfig()
        self._robot_config.linear_damping = 0.1
        self._robot_config.angular_damping = 4.4
        self._ball_configs = [BallConfig()]
        self._ball_configs[0].color = (0.3, 0.3, 0.3)
        self._ball_configs[0].line_color = (0.8, 0.8, 0.8)

        # safety factors for joint angles to avoid giving observations out of interval
        self._joint_factor = []
        for index in range(3):
            if index in [0, 1]:
                factor = self._robot_config.rod_joint_limit*self._safety_factor 
            else:
                factor = self._robot_config.racket_joint_limit*self._safety_factor
            self._joint_factor.append(factor)

        self._visible_area_width = 6.0
        self._visual_height = 0.05

        # physics simulation
        self._world = B2World(
                robot_configs = self._robot_config,
                ball_configs = self._ball_configs,
                visible_area_width = self._visible_area_width,
                steps_per_sec = self._steps_per_sec
                )

        # ball gun : specifies the reset of
        # the ball (by shooting a new one)
        self._ball_guns = [DefaultBallGun(self._ball_configs[0])]

        # robot init : specifies the reinit of the robot
        # (e.g. angles of the rods and rackets, etc)
        self._robot_init_state = DefaultRobotState(
                robot_config = self._robot_config, 
                #generalized_coordinates = [0., -0.5*np.pi, 0.],
                generalized_coordinates = [0.25*np.pi, -0.5*np.pi, 0.],
                generalized_velocities = [0., 0., 0.])

        ###################
        # Observation space
        ###################

        obs_space_dict = {}
        bounded_space = spaces.Box(low = -1.0, high = 1.0, shape= (1,), dtype = np.float32)
        unbounded_space = spaces.Box(low = -np.inf, high = np.inf, shape= (1,), dtype = np.float32)
        unit_interval = spaces.Box(low = 0.0, high = 1.0, shape= (1,), dtype = np.float32)
        for index in [0, 1, 2]:
            obs_space_dict["joint_" + str(index) + "_angle"] = bounded_space
        for index in [0, 1, 2]:
            obs_space_dict["joint_" + str(index) + "_angular_vel"] = bounded_space
        obs_space_dict["ball_pos_x"] = unbounded_space
        obs_space_dict["ball_pos_y"] = unbounded_space
        obs_space_dict["ball_vel_x"] = unbounded_space
        obs_space_dict["ball_vel_y"] = unbounded_space
        obs_space_dict["ball_anguler_vel"] = unbounded_space
        obs_space_dict["ball_bounced_at_least_once"] = unit_interval
        obs_space_dict["ball_bouncing_second_time"] = unit_interval
        obs_space_dict["ball_bounced_at_least_twice"] = unit_interval

        if self.dense_reward == True:
            # in case of dense reward have to include (first component of) desired goal into observation space
            # (second and third component are always one and therefore not useful as observation)
            obs_space_dict["desired_landing_pos_x"] = bounded_space

        # partial observation space (without goal)
        self._preliminary_obs_space = spaces.Dict(obs_space_dict)

        # Note: Observations are scaled versions of corresponding quantities
        # in physics simulation.

        # in sparse reward case, also have to specifiy desired and achieved
        # goal spaces
        if self.dense_reward == False:
            # goal space has components
            # 1. ball position x
            # 2. bool indicating whether ball bounced at least once
            # 3. bool indicating whether ball is bouncing for the second time
            #    in this time step
            # 4. bool indicating whether ball bounced at least twice
            desired_goal_space = spaces.Box(
                    low = np.array([-np.inf, 0., 0., 0.]), 
                    high = np.array([np.inf, 1., 1., 1.]),
                    dtype = np.float32)
            achieved_goal_space = desired_goal_space

            # observation space consists of dictionary of subspaces
            # corresponding to observation, desired goal and achieved
            # goal spaces
            self.observation_space = spaces.Dict({
                "observation": self._preliminary_obs_space,
                "desired_goal": desired_goal_space,
                "achieved_goal": achieved_goal_space
                })
        # in dense reward case, observation space is simply preliminary 
        # observation space
        else:
            self.observation_space = self._preliminary_obs_space

        ###################
        # Action space
        ###################

        # action space consists of torques applied to the three joints
        # Note: Actions are scaled versions of torques in physics simulation.
        act_space_dict = {}
        for index in range(3):
            act_space_dict["joint_" + str(index) + "_torque"] = bounded_space

        self.action_space = spaces.Dict(act_space_dict)

        # reset to make sure environment is not used without resetting it first
        self.reset()

    def step(self, action):

        ####################
        # Physics simulation
        ####################

        action_keys = sorted(action.keys())
        torques = [action[key][0] for key in action_keys]

        # perform one step of physics simulation, receive new world state
        self._world_state = self._world.step(torques, relative_torques = True)

        # clip angular velocities to make sure they are in a bounded interval
        for joint in self._world_state.robots[0].joints:
            joint.angular_velocity = np.clip(joint.angular_velocity, -self._max_angular_vel, 
                    self._max_angular_vel)

        ####################
        # Reward calculation
        ####################

        reward = 0
        info = {}

        # check whether the ball is bouncing off the floor in this time step
        self._ball_bouncing_second_time = False 
        if self._world_state.ball_hits_floor:
            self._n_ball_bounces += 1
            if self._n_ball_bounces == 2:
                self._ball_bouncing_second_time = True

        # set achieved goal
        achieved_goal = self._get_achieved_goal()

        # dense reward case
        if self.dense_reward == True:
            # reward for hitting ball with racket
            if self._world_state.balls_hits_racket[0]:
                self._n_hits_ball_racket += 1
                if self._n_hits_ball_racket == 1:
                    reward += self._hit_reward
            # reward for bouncing off ground in goal area
            goal_reward = self.compute_reward(achieved_goal, self._desired_goal, info)
            reward += goal_reward
            if goal_reward == 0.:
                done = True
        # sparse reward case
        else:
            goal_reward = self.compute_reward(achieved_goal, self._desired_goal, info)
            reward += goal_reward
            if goal_reward == 0.:
                done = True

        # end episode after some time
        if self._world_state.t >= self._max_episode_length_sec:
            self.done = True



        return self.get_observation(), reward, self.done, info

    def _get_achieved_goal(self):
        return  [(self._world_state.balls[0].position[0] - self._goal_min)/(self._goal_max - self._goal_min), 
                int(self._n_ball_bounces >= 1), 
                int(self._ball_bouncing_second_time),
                int(self._n_ball_bounces >= 2)]

    def update_subgoals(self, subgoals, tolerances = None):
        self._subgoals = subgoals
        self._tolerances = tolerances

    def update_timed_subgoals(self, timed_subgoals, tolerances = None):
        self._subgoals = [tsg.goal for tsg in timed_subgoals if tsg is not None]
        self._timed_subgoals = timed_subgoals
        self._tolerances = tolerances

    def reset(self):
        self.t = 0
        # check for consistency with GoalEnv
        if self.dense_reward == False:
            super().reset()

        self.done = False

        # reset physics simulation
        self._world_state = self._world.reset(self._robot_init_state, self._ball_guns) 

        # reset variables necessary for computation of reward
        self._n_ball_bounces = 0
        self._ball_bouncing_second_time = False
        self._n_hits_ball_racket = 0

        # sample goal position (last three components indicate that ball bounced for 
        # the second time in this time step)
        self._desired_goal = np.array([self.np_random.uniform(0., 1.), 1., 1., 1.])

        return self.get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode = "human", close = False):
        # have to import renderer here to avoid problems when running training without display
        # server
        from roboball2d.rendering import PygletRenderer
        from roboball2d.rendering import RenderingConfig
        import roboball2d.rendering.pyglet_utils as pyglet_utils
        import pyglet.gl as gl
        from ..utils.graphics_utils import get_default_subgoal_colors

        # render callback method which draws arrow for velocity of racket
        def render_racket_vel_callback(ws):
            scaled_vector = [self._arrow_scaling*x for x in ws.robot.racket.linear_velocity]
            pyglet_utils.draw_vector(
                    initial_point = ws.robot.racket.position, 
                    vector = scaled_vector, 
                    width = self._arrow_width, 
                    arrow_head_size = self._arrow_head_size, 
                    color = (0.8, 0.8, 0.8))

        # callback function for rendering of subgoals
        def render_subgoal_callback(ws):
            z = -0.01
            for sg, color in zip(self._subgoals, self._subgoal_colors):
                # robot
                generalized_coordinates = [sg[f"joint_{i}_angle"]*self._joint_factor[i] for i in range(3)]
                generalized_velocities = [sg[f"joint_{i}_angular_vel"]*self._max_angular_vel for i in range(3)]
                robot_state = DefaultRobotState(
                        robot_config = self._robot_config, 
                        generalized_coordinates = generalized_coordinates,
                        generalized_velocities = generalized_velocities)
                robot_state.render(
                        color = color,
                        z_coordinate = z)
                # racket velocity
                scaled_vector = [self._arrow_scaling*x for x in robot_state.racket.linear_velocity]
                gl.glPushMatrix()
                gl.glTranslatef(0., 0., z)
                pyglet_utils.draw_vector(
                        initial_point = robot_state.racket.position, 
                        vector = scaled_vector, 
                        width = self._arrow_width, 
                        arrow_head_size = self._arrow_head_size, 
                        color = color)
                gl.glPopMatrix()
                z += -0.01

        def render_time_bars_callback(ws):
            y_pos = 2.5
            for tsg, color in zip(self._timed_subgoals, self._subgoal_colors):
                if tsg is not None:
                    width = tsg.delta_t_ach*0.01
                    pyglet_utils.draw_box((0.16 + 0.5*width, y_pos), width, 0.1, 0., color)
                y_pos -= 0.1



        ########################
        # Renderer from Tennis2D
        ########################

        if self._renderer is None:
            self._subgoal_colors = get_default_subgoal_colors()

            self._callbacks.append(render_racket_vel_callback)
            self._callbacks.append(render_subgoal_callback)
            self._callbacks.append(render_time_bars_callback)

            renderer_config = RenderingConfig(self._visible_area_width,
                                              self._visual_height)
            renderer_config.window.width = 1920
            renderer_config.window.height = 960

            renderer_config.background_color = (1.0, 1.0, 1.0, 1.0)
            renderer_config.ground_color = (0.702, 0.612, 0.51)
            self._renderer = PygletRenderer(renderer_config,
                                      self._robot_config,
                                      self._ball_configs,
                                      self._callbacks)


        # render based on the information provided by
        # the physics simulation and the desired goal
        goals = [(
                self._desired_goal[0]*(self._goal_max - self._goal_min) \
                        + self._goal_min - 0.5*self._goal_diameter,
                self._desired_goal[0]*(self._goal_max - self._goal_min) \
                        + self._goal_min + 0.5*self._goal_diameter,
                self._goal_color
                )]

        self._renderer.render(
            world_state = self._world_state, 
            goals = goals, 
            time_step = self._slow_motion_factor*self._world_state.applied_time_step)

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.dense_reward == False:
            if np.all(achieved_goal[1:] == desired_goal[1:]):
                if abs((desired_goal[0] - achieved_goal[0])*(self._goal_max - self._goal_min)) <= 0.5*self._goal_diameter:
                    return 0.
            return -1.
        else:
            if np.all(achieved_goal[1:] == desired_goal[1:]):
                return (-min(abs((desired_goal[0] - achieved_goal[0])*(self._goal_max - self._goal_min)), 
                        self._goal_max + self._goal_diameter - self._robot_config.position) + 
                        self._goal_max + self._goal_diameter - self._robot_config.position)
            return 0.


    # part of observation depending only on env state and not on goal
    def _get_env_observation(self):
        ws = self._world_state
        env_observation = {}
        for index, joint in enumerate(ws.robots[0].joints):
            env_observation["joint_" + str(index) + "_angle"] = np.clip([joint.angle/self._joint_factor[index]], -1., 1.)
            env_observation["joint_" + str(index) + "_angular_vel"] = [joint.angular_velocity/self._max_angular_vel]

        ball = ws.balls[0]
        env_observation["ball_pos_x"] = [ball.position[0]/self._ball_guns[0].initial_pos_x]
        env_observation["ball_pos_y"] = [ball.position[1]/self._ball_guns[0].initial_pos_x]
        env_observation["ball_vel_x"] = [ball.linear_velocity[0]/self._ball_guns[0].speed_mean]
        env_observation["ball_vel_y"] = [ball.linear_velocity[1]/self._ball_guns[0].speed_mean]
        env_observation["ball_anguler_vel"] = [ball.angular_velocity/self._ball_guns[0].spin_std]
        env_observation["ball_bounced_at_least_once"] = [int(self._n_ball_bounces >= 1)]
        env_observation["ball_bouncing_second_time"] = [int(self._ball_bouncing_second_time)]
        env_observation["ball_bounced_at_least_twice"] = [int(self._n_ball_bounces >= 2)]

        for key, value in env_observation.items():
            env_observation[key] = np.array(value)
                
        return env_observation

    def get_observation(self):
        observation = self._get_env_observation()
        if self.dense_reward == False:
            result = { 
                "observation": observation,
                "achieved_goal": np.array(self._get_achieved_goal()),
                "desired_goal" : np.array(self._desired_goal)
                }
            return result
        else:
            observation["desired_landing_pos_x"] = self._desired_goal[0]
            return observation

    def map_to_achieved_goal(self, partial_obs):
        pos_x = partial_obs["ball_pos_x"][0]*self._ball_guns[0].initial_pos_x
        achieved_goal = [
                [(pos_x - self._goal_min)/(self._goal_max - self._goal_min)], 
                partial_obs["ball_bounced_at_least_once"], 
                partial_obs["ball_bouncing_second_time"], 
                partial_obs["ball_bounced_at_least_twice"]]

        return np.concatenate(achieved_goal)

    def _get_robot_conf_and_vel(self, robot_state):
        pass


class Tennis2DDenseRewardEnv(Tennis2DEnv):
    """Dense reward version of the 2D robotic toy tennis environment.

    In contrast to the sparse reward version, the dense reward version of the 
    environment gives a constant reward to the agent when it hits the ball 
    and another one when the ball bounces on the ground for the second time after being hit 
    by the racket. The latter reward is proportional to the nagative distance
    to the goal landing point."""

    dense_reward = True
