import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    """
    minimal requiered methods: make_world, reset_world_at, observation, and reward.
    optional methods: done, info, process_action, extra_render.

    Main hardcoded attributes
    ----------
    n_agents : int = 4
        number of agents
    agent_radius : float = 0.16
        radius of agents if box_agents=False
    agent_box_length : float = 0.32
        length of agents if box_agents=True
    agent_box_width : float = 0.24
        width of agents if box_agents=True
    min_collision_distance: float = 0.005
        minimal distance between agents without penalties
    scenario_length: int|float = 5
        length of the hall/tunnel of each agent
    scenario_width: int|float = 0.4
        width of the hall/tunnel of each agent
    goal_dist_from_wall: int|float = agent_radius + 0.05
    agent_dist_from_wall: int|float = 0.5


    Methods
    -------
    make_world(batch_dim, device, **kwargs):
        initializing world parameters, agents, rewards + spawn_map (see below)

    reset_world_at(env_index):
        reset world by its index in parallel envs. resetting agents, rewards + reset_map (see below)

    spawn_map(world):
        similar to make_world, but for walls and obstacles (without coordinates, just objects's attributes)

    reset_map(env_index):
        similar to reset_world_at, but for walls and obstacles s

    observation(agent):
        return the agent observation - stack of the agent position, the agent speed,
        the distance to the goal (coordinate-wise) and its norm

    reward(agent):
        return the agent reward (sum agent rewards if shared) + the collision penalty

    process_action(agent):
        clamp an action with big norm
        fill small action with zero

    extra_render(env_index):
        return the list of Geom objects from vmas.simulator.rendering to render

    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # base params
        self.u_range = kwargs.pop("u_range", 0.5)  # control range for agents
        self.a_range = kwargs.pop(
            "a_range", 1
        )  # force constaints = a_range + linear friction
        self.obs_noise = kwargs.pop("obs_noise", 0)
        self.box_agents = kwargs.pop("box_agents", False)
        self.linear_friction = kwargs.pop("linear_friction", 0.1)
        self.min_input_norm = kwargs.pop("min_input_norm", 0.08)
        self.comms_range = kwargs.pop("comms_range", 5)
        self.shared_rew = kwargs.pop("shared_rew", True)
        self.n_agents = kwargs.pop("n_agents", 4)

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)  # max is 8
        self.final_reward = kwargs.pop("final_reward", 0.01)
        self.energy_reward_coeff = kwargs.pop("energy_rew_coeff", 0)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.1)
        # self.passage_collision_penalty = kwargs.pop("passage_collision_penalty", 0)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", 0)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.viewer_zoom = 1.7

        controller_params = [2, 6, 0.002]

        self.n_agents = 4
        self.f_range = self.a_range + self.linear_friction

        # Make world
        world = World(
            batch_dim,
            device,
            drag=0,
            dt=0.1,
            linear_friction=self.linear_friction,
            substeps=16 if self.box_agents else 5,
            collision_force=10000 if self.box_agents else 500,
        )

        self.agent_radius = 0.16
        self.agent_box_length = 0.32
        self.agent_box_width = 0.24

        self.min_collision_distance = 0.005

        self.colors = [Color.GREEN, Color.BLUE, Color.RED, Color.GRAY]

        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                rotatable=False,
                linear_friction=self.linear_friction,
                shape=(
                    Sphere(radius=self.agent_radius)
                    if not self.box_agents
                    else Box(length=self.agent_box_length, width=self.agent_box_width)
                ),
                u_range=self.u_range,
                f_range=self.f_range,
                render_action=True,
                color=self.colors[i],
            )
            agent.controller = VelocityController(
                agent, world, controller_params, "standard"
            )
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius / 2),
                color=self.colors[i],
            )
            agent.goal = goal
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)
            world.add_landmark(goal)

        # modification
        self.scenario_length = kwargs.get("scenario_length", 5)
        self.scenario_width = kwargs.get("scenario_width", 0.4)
        self.scenario_angle = kwargs.get("scenario_angle", torch.pi / 3)

        self.spawn_map(world)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        scenario_angle = torch.tensor(self.scenario_angle, dtype=torch.float32, device=self.world.device)
        sin_angle = torch.sin(scenario_angle)
        cos_angle = torch.cos(scenario_angle)

        base_x_agent_pos = torch.tensor(
            [
                - (self.scenario_length / 2 - self.agent_dist_from_wall), # agent_0
                0,                                                        # agent_1
                self.scenario_length / 2 - self.agent_dist_from_wall,     # agent_2
                0,                                                        # agent_3
            ],
            dtype=torch.float32,
            device=self.world.device,
        )
        base_y_agent_pos = torch.tensor(
            [
                0,                                                        # agent_0
                self.scenario_length / 2 - self.agent_dist_from_wall,     # agent_1
                0,                                                        # agent_2
                - (self.scenario_length / 2 - self.agent_dist_from_wall), # agent_3
            ],
            dtype=torch.float32,
            device=self.world.device,
        )
        base_x_goal_pos = torch.tensor(
            [
                0,                                                       # agent_0
                - (self.scenario_length / 2 - self.goal_dist_from_wall), # agent_1
                0,                                                       # agent_2
                self.scenario_length / 2 - self.goal_dist_from_wall,     # agent_3
            ],
            dtype=torch.float32,
            device=self.world.device,
        )
        base_y_goal_pos = torch.tensor(
            [
                - (self.scenario_length / 2 - self.goal_dist_from_wall), # agent_0
                0,                                                        # agent_1
                self.scenario_length / 2 - self.goal_dist_from_wall,      # agent_2
                0,                                                        # agent_3
            ],
            dtype=torch.float32,
            device=self.world.device,
        )
        for base_x, base_y, base_x_goal, base_y_goal, agent in zip(base_x_agent_pos, base_y_agent_pos, base_x_goal_pos, base_y_goal_pos, self.world.agents):
            x = base_x * cos_angle - base_y * sin_angle
            y = base_x * sin_angle + base_y * cos_angle
            agent.set_pos(torch.Tensor((x, y)), batch_index=env_index)

            x_goal = base_x_goal * cos_angle - base_y_goal * sin_angle
            y_goal = base_x_goal * sin_angle + base_y_goal * cos_angle

            agent.goal.set_pos(torch.Tensor((x_goal, y_goal)), batch_index=env_index)

        for agent in self.world.agents:
            if env_index is None:
                agent.shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

        self.reset_map(env_index)

        if env_index is None:
            self.reached_goal = torch.full(
                (self.world.batch_dim,), False, device=self.world.device
            )
        else:
            self.reached_goal[env_index] = False

    def process_action(self, agent: Agent):
        # Clamp square to circle
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, self.u_range)

        # Zero small input
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < self.min_input_norm] = 0

        # Copy action
        agent.vel_action = agent.action.u.clone()

        # Reset controller
        vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 1e-3
        agent.controller.reset(vel_is_zero)

        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                a.distance_to_goal = torch.linalg.vector_norm(
                    a.state.pos - a.goal.state.pos,
                    dim=-1,
                )
                a.on_goal = a.distance_to_goal < a.goal.shape.radius

                pos_shaping = a.distance_to_goal * self.pos_shaping_factor
                a.pos_rew = (
                    (a.shaping - pos_shaping)
                    if self.pos_shaping_factor != 0
                    else -a.distance_to_goal * 0.0001
                )
                a.shaping = pos_shaping

                self.pos_rew += a.pos_rew

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward
            self.reached_goal += self.all_goal_reached

        agent.agent_collision_rew[:] = 0
        # agent.obstacle_collision_rew = torch.zeros(
        #     (self.world.batch_dim,), device=self.world.device
        # )
        for a in self.world.agents:
            if a != agent:
                agent.agent_collision_rew[
                    self.world.get_distance(agent, a) <= self.min_collision_distance
                ] += self.agent_collision_penalty
        # for l in self.world.landmarks:
        #     if self.world._collides(agent, l):
        #         if l in [*self.passage_1, *self.passage_2]:
        #             penalty = self.passage_collision_penalty
        #         else:
        #             penalty = self.obstacle_collision_penalty
        #         agent.obstacle_collision_rew[
        #             self.world.get_distance(agent, l) <= self.min_collision_distance
        #         ] += penalty

        # Energy reward
        # agent.energy_expenditure = torch.linalg.vector_norm(
        #     agent.action.u, dim=-1
        # ) / math.sqrt(self.world.dim_p * (agent.f_range**2))
        #
        # agent.energy_rew = -agent.energy_expenditure * self.energy_reward_coeff

        return (
            (self.pos_rew if self.shared_rew else agent.pos_rew)
            # + agent.obstacle_collision_rew
            + agent.agent_collision_rew
            # + agent.energy_rew
            + self.final_rew
        )

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.state.pos - agent.goal.state.pos,
            torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos,
                dim=-1,
            ).unsqueeze(-1),
        ]

        if self.obs_noise > 0:
            for i, obs in enumerate(observations):
                noise = torch.zeros(
                    *obs.shape,
                    device=self.world.device,
                ).uniform_(
                    -self.obs_noise,
                    self.obs_noise,
                )
                observations[i] = obs + noise
        return torch.cat(
            observations,
            dim=-1,
        )

    def info(self, agent: Agent):
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            # "energy_rew": agent.energy_rew,
            "agent_collision_rew": agent.agent_collision_rew,
            # "obstacle_collision_rew": agent.obstacle_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms

    def spawn_map(self, world: World):
        self.long_wall_length = (self.scenario_length / 2) - (self.scenario_width / 2)
        self.short_wall_length = self.scenario_width
        self.goal_dist_from_wall = self.agent_radius + 0.05
        self.agent_dist_from_wall = 0.5

        self.long_walls = []
        for i in range(8):
            landmark = Landmark(
                name=f"wall {i}",
                collide=True,
                shape=Line(length=self.long_wall_length),
                color=Color.BLACK,
            )
            self.long_walls.append(landmark)
            world.add_landmark(landmark)

        self.short_walls = []
        for i in range(4):
            landmark = Landmark(
                name=f"short wall {i}",
                collide=True,
                shape=Line(length=self.short_wall_length),
                color=Color.BLACK,
            )
            self.short_walls.append(landmark)
            world.add_landmark(landmark)

    def reset_map(self, env_index):
        scenario_angle = torch.tensor(self.scenario_angle, dtype=torch.float32, device=self.world.device)
        sin_angle = torch.sin(scenario_angle)
        cos_angle = torch.cos(scenario_angle)

        base_x_short_wall_pos = torch.tensor(
            [
                self.scenario_length / 2, 
                0, 
                -self.scenario_length / 2, 
                0
            ],
            dtype=torch.float32,
            device=self.world.device,
        )

        base_y_short_wall_pos = torch.tensor(
            [
                0, 
                self.scenario_length / 2, 
                0, 
                -self.scenario_length / 2
            ],
            dtype=torch.float32,
            device=self.world.device,
        )

        base_angle_short_wall = torch.tensor(
            [
                torch.pi / 2, 
                0, 
                torch.pi / 2, 
                0
            ],
            dtype=torch.float32,
            device=self.world.device,
        )

        for base_x, base_y, base_angle, landmark in zip(
            base_x_short_wall_pos,
            base_y_short_wall_pos,
            base_angle_short_wall,
            self.short_walls,
        ):
            x = base_x * cos_angle - base_y * sin_angle
            y = base_x * sin_angle + base_y * cos_angle
            angle = base_angle + scenario_angle

            landmark.set_pos(torch.Tensor((x, y)), batch_index=env_index)
            landmark.set_rot(angle, batch_index=env_index)

        base_x_long_wall_pos = torch.tensor(
            [
                self.scenario_width / 2 + self.long_wall_length / 2,
                self.scenario_width / 2,
                -self.scenario_width / 2,
                -self.scenario_width / 2 - self.long_wall_length / 2,
                -self.scenario_width / 2 - self.long_wall_length / 2,
                -self.scenario_width / 2,
                self.scenario_width / 2,
                self.scenario_width / 2 + self.long_wall_length / 2,
            ],
            dtype=torch.float32,
            device=self.world.device,
        )

        base_y_long_wall_pos = torch.tensor(
            [
                self.scenario_width / 2,
                self.scenario_width / 2 + self.long_wall_length / 2,
                self.scenario_width / 2 + self.long_wall_length / 2,
                self.scenario_width / 2,
                -self.scenario_width / 2,
                -self.scenario_width / 2 - self.long_wall_length / 2,
                -self.scenario_width / 2 - self.long_wall_length / 2,
                -self.scenario_width / 2,
            ],
            dtype=torch.float32,
            device=self.world.device,
        )

        base_angle_long_wall = torch.tensor(
            [
                0, 
                torch.pi / 2, 
                torch.pi / 2, 
                0, 
                0, 
                torch.pi / 2, 
                torch.pi / 2, 
                0
            ],
            dtype=torch.float32,
            device=self.world.device,
        )

        for base_x, base_y, base_angle, landmark in zip(
            base_x_long_wall_pos,
            base_y_long_wall_pos,
            base_angle_long_wall,
            self.long_walls,
        ):
            x = base_x * cos_angle - base_y * sin_angle
            y = base_x * sin_angle + base_y * cos_angle
            angle = base_angle + scenario_angle

            landmark.set_pos(torch.Tensor((x, y)), batch_index=env_index)
            landmark.set_rot(angle, batch_index=env_index)


if __name__ == "__main__":
    render_interactively(Scenario(), control_two_agents=True)
