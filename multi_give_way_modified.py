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


def polar2cartesian(r, phi):
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return x, y

def rotate_cartesian(x, y, angle):
    sin_angle = torch.sin(angle)
    cos_angle = torch.cos(angle)

    new_x = x * cos_angle - y * sin_angle
    new_y = x * sin_angle + y * cos_angle

    return new_x, new_y


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
        length of the tunnel of each agent
    scenario_width: int|float = 0.4
        width of the tunnel of each agent
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
        # self.n_agents = kwargs.pop("n_agents", 4)

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)  # max is 8
        self.final_reward = kwargs.pop("final_reward", 0.01)
        self.energy_reward_coeff = kwargs.pop("energy_rew_coeff", 0)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.1)
        # self.passage_collision_penalty = kwargs.pop("passage_collision_penalty", 0)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", 0)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.viewer_zoom = 1.7

        controller_params = [2, 6, 0.002]

        # modified
        self.n_tunnels = kwargs.get("n_tunnels", 5)
        assert self.n_tunnels > 2, "minimum n_tunnels = 3"

        self.n_agents = kwargs.get("n_agents", [2 for _ in range(self.n_tunnels)])
        if isinstance(self.n_agents, int):
            self.n_agents = [self.n_agents for _ in range(self.n_tunnels)]
        elif not isinstance(self.n_agents, list) and len(self.n_agents) != self.n_tunnels:
            raise ValueError(f"n_agents must be int or List[int] with the size = n_tunnels, received {self.n_agents} and n_tunnels={self.n_tunnels}")
        
        self.f_range = self.a_range + self.linear_friction
        self.phi = torch.tensor(2 * torch.pi / self.n_tunnels, dtype=torch.float32, device=device)
        self.phis_tunnels = self.phi * torch.arange(self.n_tunnels, dtype=torch.float32, device=device)

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

        self.color = Color.GREEN # need modification

        # Add agents
        for tunnel_id in range(self.n_tunnels):
            for agent_id in range(self.n_agents[tunnel_id]):
                agent = Agent(
                    name=f"agent{tunnel_id}_{agent_id}",
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
                    color=self.color,
                )
                agent.controller = VelocityController(
                    agent, world, controller_params, "standard"
                )
                goal = Landmark(
                    name=f"goal{tunnel_id}_{agent_id}",
                    collide=False,
                    shape=Sphere(radius=self.agent_radius / 2),
                    color=self.color,
                )
                agent.goal = goal
                agent.pos_rew = torch.zeros(batch_dim, device=device)
                agent.agent_collision_rew = agent.pos_rew.clone()
                world.add_agent(agent)
                world.add_landmark(goal)

        # modification
        self.tunnel_spaces = kwargs.get("center_lengths", [2 for _ in range(self.n_tunnels)])
        if isinstance(self.tunnel_spaces, int):
            self.tunnel_spaces = [self.tunnel_spaces for _ in range(self.n_tunnels)]
        elif not isinstance(self.tunnel_spaces, list) and len(self.tunnel_spaces) != self.n_tunnels:
            raise ValueError(f"center_lengths must be int or List[int] with the size = n_tunnels, received {self.tunnel_spaces} and n_tunnels={self.n_tunnels}")

        self.scenario_width = kwargs.get("scenario_width", 0.4)
        self.scenario_angle = kwargs.get("scenario_angle", 0)
        self.scenario_angle = torch.tensor(self.scenario_angle, dtype=torch.float32, device=device)

        self.center_size_2 = self.scenario_width ** 2 / (2 * (1 - torch.cos(self.phi)))
        self.center_size = torch.sqrt(self.center_size_2 - self.scenario_width ** 2 / 4)

        self.spawn_map(world)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def spawn_map(self, world: World):
        self.long_walls = [[] for _ in range(self.n_tunnels)]
        self.short_walls = []
        for tunnel_id in range(self.n_tunnels):
            tunnel_length = self.tunnel_spaces[tunnel_id] + self.scenario_width * self.n_agents[tunnel_id]

            # 2 long walls
            for wall_id in range(2):
                long_wall = Landmark(
                    name=f"long-wall{tunnel_id}_{wall_id}",
                    collide=True,
                    shape=Line(length=tunnel_length),
                    color=Color.BLACK,
                )

                world.add_landmark(long_wall)
                self.long_walls[tunnel_id].append(long_wall)
            
            # 1 short wall
            short_wall = Landmark(
                name=f"short-wall{tunnel_id}",
                collide=True,
                shape=Line(length=self.scenario_width),
                color=Color.BLACK,
            )

            world.add_landmark(short_wall)
            self.short_walls.append(short_wall)

    def reset_world_at(self, env_index: int = None):
        for tunnel_id in range(self.n_tunnels):
            for agent_id in range(self.n_agents[tunnel_id]):
                agent = self.world.agents[tunnel_id * self.n_agents[tunnel_id] + agent_id]
                r_agent = self.tunnel_spaces[tunnel_id] + (agent_id + 1) * self.scenario_width - self.scenario_width + self.center_size
                phi_agent = self.phis_tunnels[tunnel_id]
                phi_agent_rotated = phi_agent + self.scenario_angle

                x_agent, y_agent = polar2cartesian(r_agent, phi_agent_rotated)
                agent.set_pos(torch.Tensor((x_agent, y_agent)), batch_index=env_index)

                r_goal = r_agent
                phi_goal = self.phis_tunnels[(tunnel_id + 1) % self.n_tunnels]
                phi_goal_rotated = phi_goal + self.scenario_angle

                x_goal, y_goal = polar2cartesian(r_goal, phi_goal_rotated)
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

    def reset_map(self, env_index):
        for tunnel_id in range(self.n_tunnels):
            # 2 long walls
            for long_wall_id, landmark in enumerate(self.long_walls[tunnel_id]):
                base_x = self.center_size + self.tunnel_spaces[tunnel_id] / 2 + self.n_agents[tunnel_id] * self.scenario_width / 2
                base_y = self.scenario_width / 2 * (-1 if long_wall_id == 0 else 1)
                angle = self.scenario_angle + self.phis_tunnels[tunnel_id]

                x, y = rotate_cartesian(base_x, base_y, angle)
                landmark.set_pos(torch.Tensor((x, y)), batch_index=env_index)
                landmark.set_rot(angle, batch_index=env_index)

            # 1 short wall
            landmark = self.short_walls[tunnel_id]
            base_x = self.center_size + self.tunnel_spaces[tunnel_id] + self.n_agents[tunnel_id] * self.scenario_width
            base_y = 0
            angle = self.scenario_angle + self.phis_tunnels[tunnel_id]

            x, y = rotate_cartesian(base_x, base_y, angle)
            landmark.set_pos(torch.Tensor((x, y)), batch_index=env_index)
            landmark.set_rot(torch.pi / 2 + angle, batch_index=env_index)

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


if __name__ == "__main__":
    render_interactively(Scenario(), control_two_agents=True)
