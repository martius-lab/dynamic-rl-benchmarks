import numpy as np
import gym
from gym import spaces
from Box2D import (b2PolygonShape, b2World, b2FixtureDef, b2CircleShape, b2ContactListener)

from ..utils.graphics_utils import ArrowConfig, get_default_subgoal_colors


class PlatformsEnv(gym.GoalEnv):  
    metadata = {'render.modes': ['human']}   

    class _Trigger:

        def __init__(self, position, width, height, max_act_time):
            self.pos = position
            self.width = width
            self.height = height
            self.max_act_time = max_act_time
            self.is_active = False
            self.time_since_act = 0.

        def update(self, agent_pos, delta_t):
            if self.is_active:
                self.time_since_act += delta_t
                if self.time_since_act >= self.max_act_time:
                    self.is_active = False
            if not self.is_active:
                if self.pos[0] - 0.5*self.width < agent_pos[0] < self.pos[0] + 0.5*self.width and \
                        self.pos[1] - 0.5*self.height < agent_pos[1] < self.pos[1] + 0.5*self.height:
                    self.is_active = True
                    self.time_since_act = 0.

        def reset(self):
            self.is_active = False
            self.time_since_act = 0.

    def _create_static_b2_rect(self, center_x, center_y, width, height):
        self.world.CreateStaticBody(position = (center_x, center_y), shapes = b2PolygonShape(box = (0.5*width, 0.5*height)))
        self.static_rects.append((center_x, center_y, width, height))

    def _create_kinematic_b2_rect(self, pos_func, vel_func, width, height, color):
        body = self.world.CreateKinematicBody(position = (0., 0.), shapes = b2PolygonShape(box = (0.5*width, 0.5*height)))
        self.kinematic_rects.append([body, pos_func, vel_func, width, height, color])

    def __init__(self, max_episode_length = 500, subgoal_radius = 0.05):
        super().__init__()

        self.max_episode_length = max_episode_length
        self.current_step = 0
        self.level_width = 2.

        # safety distance from boundaries of interval [-1., 1.]
        self._eps = 1e-1

        self.window = None
        self.window_width = 800
        self.window_height = 600
        self.background_color = (1.0, 1.0, 1.0, 1.0)

        self._vector_width = 0.02 
        self._arrow_head_size = 0.05
        self._velocity_color = (0.0, 0.0, 0.0)
        self._velocity_scale = 0.2

        self.static_color = (0.4, 0.4, 0.4)
        self.platform_color = (0.702, 0.612, 0.51)

        self.agent_radius = 0.05
        self.agent_color = (0.0, 0.0, 0.0)
        self.line_color = (0.5, 0.5, 0.5)

        self._agent_ac = ArrowConfig(self._velocity_scale, self._vector_width, 
                self._arrow_head_size, self.line_color)

        self._subgoals = []
        self._timed_subgoals = []
        self._tolerances = []
        self._subgoal_colors = get_default_subgoal_colors()
        self._subgoal_acs = [ArrowConfig(self._velocity_scale, self._vector_width, 
                self._arrow_head_size, color) for color in self._subgoal_colors]
        self.subgoal_radius = float(subgoal_radius)*self.level_width*0.5

        self.goal_radius = 0.05
        self.goal_color = (0.0, 0.0, 0.0)


        self.static_rects = []
        self.kinematic_rects = []

        # set up physics in Box2D         
        self.world = b2World(gravity = (0., -9.8))
        self.vel_iters, self.pos_iters = 10, 8
        self.time_step = 1./60.

        # static elements of level
        ground_level = -0.6
        # keep ball from falling out of play area
        self._create_static_b2_rect(-0., -0.5*self.level_width - 0.5, self.level_width, 1.0) 
        self._create_static_b2_rect(-0.25, ground_level - 0.35, 1.5, 0.7) # ground 
        self._create_static_b2_rect(-0.5*self.level_width - 0.5, .0, 1.0, 10) # left wall
        self._create_static_b2_rect( 0.5*self.level_width + 0.5, .0, 1.0, 10) # right wall
        level_1 = -0.1
        diff1 = level_1 - ground_level 
        self.level_2 = level_1 + diff1
        self.level_1 = level_1
        level_thickness = 0.1
        self._create_static_b2_rect(-0.5, self.level_2 - 0.5*level_thickness, 1., level_thickness) # level 1

        self.agent_initial_position = np.array((-0.8, ground_level + self.agent_radius))

        # triggers
        self.active_trigger_color = (0.2, 0.8, 0.2)
        self.inactive_trigger_color = (0.8, 0.2, 0.2)
        self.triggers = []
        omega2 = 2.
        self.triggers.append(self._Trigger(
            position = [-0.0, ground_level + 0.05], 
            width = 0.1, 
            height = 0.1, 
            max_act_time = 2.*np.pi/self.time_step/omega2))

        # dynamic elements (platforms)
        # continually moving platform
        omega1 = 3
        phi1 = -.8*np.pi
        self._create_kinematic_b2_rect(
                pos_func = lambda t: (0.75, 0.5*(ground_level + level_1) - 0.5*level_thickness + 0.5*diff1*np.sin(omega1*t + phi1)), 
                vel_func = lambda t: (0., omega1*0.5*diff1*np.cos(omega1*t + phi1)), 
                width = 0.5, 
                height = level_thickness, 
                color = self.platform_color)

        # triggered platform
        phi2 = 0.
        self._create_kinematic_b2_rect(
                pos_func = lambda t: (0.25, level_1 + 0.5*diff1 - 0.5*level_thickness + 0.5*diff1*np.cos(omega2*self.triggers[0].time_since_act*self.time_step + phi2)), 
                vel_func = lambda t: (0., -omega1*0.5*diff1*np.sin(omega2*self.triggers[0].time_since_act*self.time_step + phi2)) if self.triggers[0].is_active else (0., 0.), 
                width = 0.5, 
                height = level_thickness, 
                color = self.inactive_trigger_color)

        # ball/agent
        self.max_force = 1e-1
        #self.max_vel_comp = 1e-1
        self.max_vel_comp = 1
        self.max_ang_vel = 2*np.pi*self.max_vel_comp/(2*np.pi*self.agent_radius)
        ball_fixture = b2FixtureDef(
            shape = b2CircleShape(pos = (0., 0.), radius = self.agent_radius),
            density = 1.0, 
            restitution = 0.4
        )
        self.ball = self.world.CreateDynamicBody(
            position = self.agent_initial_position,
            fixtures = ball_fixture
        )


        # define spaces
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

        self.action_space = spaces.Box(
                low = -1., 
                high = 1., 
                shape = (1,),
                dtype = np.float32)

        self.reset()


    def _draw_goal_return(self):
        mean_x = -0.5
        mean_x += np.random.uniform(-0.1, 0.1)
        y = self.level_2 + self.goal_radius
        return np.array((mean_x, y))

    def _draw_goal(self):
        self.goal = self._draw_goal_return()

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.linalg.norm(achieved_goal - desired_goal) <= \
                self.goal_radius/self.level_width/0.5:
                    return 0.
        else:
            return -1.

    def _get_obs(self):
        agent_obs = {
                "position": np.array(self.ball.position)/self.level_width*2., 
                "velocity": np.array(self.ball.linearVelocity)/self.max_vel_comp, 
                "ang_vel": np.array([self.ball.angularVelocity/self.max_ang_vel]), 
                }
        platf_obs = {"platform{}".format(i): np.concatenate([platf[0].position/self.level_width*2., 
            platf[0].linearVelocity/self.max_vel_comp]) for i, platf in enumerate(self.kinematic_rects)}
        partial_obs = {**agent_obs, **platf_obs}
        obs = { 
            "observation": partial_obs,
            "desired_goal" : self.goal/self.level_width*2.,
            "achieved_goal": np.array(self.ball.position)/self.level_width*2.
            }
        return obs


    def step(self, action):
        self.ball.ApplyForce((action[0]*self.max_force, 0.), point = self.ball.position, wake = True)

        t = self.current_step*self.time_step
        for k_rect in self.kinematic_rects:
            k_rect[0].position = k_rect[1](t)
            k_rect[0].linearVelocity = k_rect[2](t)

        # run integrator and constraint solver in Box2D
        self.world.Step(self.time_step, self.vel_iters, self.pos_iters)
        self.world.ClearForces()

        # clip linear and angular velocity
        self.ball.linearVelocity = np.clip(self.ball.linearVelocity, -self.max_vel_comp + self._eps, 
                self.max_vel_comp - self._eps)
        self.ball.angularVelocity = np.clip(self.ball.angularVelocity, -self.max_ang_vel + self._eps, 
                self.max_ang_vel - self._eps)

        # update triggers
        for trigger in self.triggers:
            trigger.update(self.ball.position, 1.)
            if trigger.is_active:
                self.kinematic_rects[1][5] = self.active_trigger_color
            else:
                self.kinematic_rects[1][5] = self.inactive_trigger_color


        self.current_step += 1
        info = {}
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        done = reward == 0. or self.current_step >= self.max_episode_length
        return obs, reward, done, info

    def reset(self):
        self.ball.position = self.agent_initial_position
        self.ball.angle = 0.
        self.ball.linearVelocity = np.zeros((2, ))
        self.ball.angularVelocity = 0.

        for trigger in self.triggers:
            trigger.reset()


        self.current_step = 0
        self._draw_goal()
        return self._get_obs()

    def update_subgoals(self, subgoals):
        self._subgoals = subgoals

    def update_timed_subgoals(self, timed_subgoals, tolerances):
        self._timed_subgoals = timed_subgoals
        self._tolerances = tolerances

    def render(self, mode='human', close=False):
        import pyglet
        import pyglet.gl as gl

        from ..utils.pyglet_utils import (draw_circle_sector, draw_box, draw_line, draw_vector, draw_vector_with_outline, 
                draw_circular_subgoal)

        if self.window is None:
            self.window = pyglet.window.Window(width = self.window_width,
                                               height = self.window_height,
                                               vsync = True,
                                               resizable = True)
            gl.glClearColor(*self.background_color)

        @self.window.event
        def on_resize(width, height):
            gl.glViewport(0, 0, width, height)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(-0.5*self.level_width, 
                       0.5*self.level_width, 
                       -0.5*float(height)/width*self.level_width,
                       0.5*float(height)/width*self.level_width, 
                       -1., 
                       1.)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            return pyglet.event.EVENT_HANDLED


        def draw_timed_circular_subgoal(position, velocity, delta_t_ach, delta_t_comm, radius, color, arrow_config):
            # desired time until achievement
            draw_box(position + (0., radius + 0.04), delta_t_ach/100. + 0.02, 0.03 + 0.02, 0., (0., 0., 0.))
            draw_box(position + (0., radius + 0.04), delta_t_ach/100., 0.03, 0., color)
            # subgoal
            draw_circular_subgoal(position, velocity, radius, color, arrow_config)


        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        gl.glLoadIdentity()

        n_triangles = 32

        # static rectangles making up level
        for rect in self.static_rects:
            draw_box(rect[0:2], rect[2], rect[3], 0., self.static_color)

        # kinematic rectangles making up platforms
        for k_rect in self.kinematic_rects:
            draw_box(k_rect[0].position, k_rect[3], k_rect[4], 0., k_rect[5])

        # trigger
        tr = self.triggers[0]
        draw_box((tr.pos[0], tr.pos[1] - tr.height), tr.width, tr.height, 0., self.active_trigger_color if tr.is_active else self.inactive_trigger_color)

        # goal
        draw_circle_sector(self.goal, 
                0., 
                self.goal_radius,
                n_triangles, 
                self.goal_color,
                n_triangles)
        draw_circle_sector(self.goal, 
                0., 
                self.goal_radius*0.8,
                n_triangles, 
                self.background_color[:3],
                n_triangles)

        # subgoals
        for sg, color, ac in zip(self._subgoals, self._subgoal_colors, self._subgoal_acs):
            if sg is not None:
                draw_circular_subgoal(sg["position"], sg["velocity"]*self.max_vel_comp, self.subgoal_radius, color, ac)

        # timed subgoals
        for ts, color, ac, tol in zip(self._timed_subgoals, self._subgoal_colors, 
                self._subgoal_acs, self._tolerances):
            if ts is not None:
                r = tol["position"] if tol is not None else self.subgoal_radius
                draw_timed_circular_subgoal(ts.goal["position"], ts.goal["velocity"]*self.max_vel_comp, ts.delta_t_ach, 
                        ts.delta_t_comm, r, color, ac)

        # agent 
        draw_circle_sector(self.ball.position, 
                self.ball.angle, 
                self.agent_radius,
                n_triangles, 
                self.agent_color,
                n_triangles)
        draw_line(self.ball.position, self.agent_radius, self.ball.angle, self.line_color)
        draw_vector(self.ball.position, self.ball.linearVelocity, self._agent_ac)
                                                        
        self.window.flip()
