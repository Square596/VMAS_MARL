import typing
from typing import List

import torch
import torch.nn.functional as F

from vmas import render_interactively
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

import yaml


def grid_pos2coords(grid_pos, scenario_width): 
    return (grid_pos + 1/2) * scenario_width

def str_cfg2num_cfg(str_cfg):
    cfg = list(str_cfg.values())[0]
    cfg_ = '[[' + '],['.join(cfg.split('\n')) + ']]'

    num_cfg = torch.tensor(eval(cfg_.replace('.', '-1, ').replace('#', '1, ')))
    return num_cfg

class Scenario(BaseScenario):
    """
    minimal requiered methods: make_world, reset_world_at, observation, and reward.
    optional methods: done, info, process_action, extra_render.

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
        self.obstacles_cfg = kwargs.get("obstacles_cfg", None)
        assert self.obstacles_cfg is not None
        with open(self.obstacles_cfg, 'rb') as f:
            self.obstacles_cfg = yaml.safe_load(f)
        
        self.obstacles_cfg = str_cfg2num_cfg(self.obstacles_cfg)

        self.obstacles_cfg = F.pad(self.obstacles_cfg, (1, 1, 1, 1), value=1)

        self.obstacles_grid_positions = torch.nonzero(self.obstacles_cfg == 1)
        assert self.obstacles_grid_positions.shape[1] == 2

        self.num_rows, self.num_cols = self.obstacles_cfg.shape

        self._rows, self._cols = torch.meshgrid(torch.arange(self.num_rows, device=device), torch.arange(self.num_cols, device=device), indexing='ij')

        self.num_obstacles = self.obstacles_grid_positions.shape[0]

        self.agents_cfg = kwargs.get("agents_cfg", torch.zeros(self.num_rows - 2, self.num_cols - 2, dtype=torch.int8, device=device) - 1)
        self.agents_cfg[0, 3] = 0 # TODO
        self.agents_cfg[-1, 3] = 1 # TODO
        self.agents_cfg = F.pad(self.agents_cfg, (1, 1, 1, 1), value=-1)

        self.num_agents = torch.max(self.agents_cfg) + 1

        assert self.agents_cfg.unique().shape[0] == self.num_agents + 1

        self.rewards_cfg = kwargs.get("rewards_cfg", torch.zeros(self.num_rows - 2, self.num_cols - 2, dtype=torch.int8, device=device) - 1)
        self.rewards_cfg[-1, -3] = 0 # TODO
        self.rewards_cfg[0, -3] = 1 # TODO
        self.rewards_cfg = F.pad(self.rewards_cfg, (1, 1, 1, 1), value=-1)

        num_rewards = self.rewards_cfg.max() + 1

        assert self.rewards_cfg.unique().shape[0] == num_rewards + 1

        assert self.obstacles_cfg.shape == self.agents_cfg.shape == self.rewards_cfg.shape
        
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
        self.scenario_width = 0.4

        self.min_collision_distance = 0.005

        self.color = Color.GREEN # TODO

        # Add agents
        for agent_id in range(self.num_agents):
            agent = Agent(
                name=f"agent{agent_id}",
                rotatable=False,
                linear_friction=self.linear_friction,
                shape=(
                    Sphere(radius=self.agent_radius)
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
                name=f"goal{agent_id}",
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
        self.spawn_map(world)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def spawn_map(self, world: World):
        self.obstacles = []
        for obstacle_id in range(self.num_obstacles):
            obstacle = Landmark(
                name=f"obstacle{obstacle_id}",
                collide=True,
                shape=Box(length=self.scenario_width, width=self.scenario_width),
                color=Color.BLACK,
            )

            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

    def reset_world_at(self, env_index: int = None):
        for agent_id in range(self.num_agents):
            agent = self.world.agents[agent_id]
            agent_pos_mask = self.agents_cfg == agent_id
            agent_grid_pos = torch.hstack([self._rows[agent_pos_mask], self._cols[agent_pos_mask]])
            agent_coords = grid_pos2coords(agent_grid_pos, self.scenario_width)
            agent.set_pos(agent_coords, batch_index=env_index)

            reward_pos_mask = self.rewards_cfg == agent_id
            reward_grid_pos = torch.hstack([self._rows[reward_pos_mask], self._cols[reward_pos_mask]])
            reward_coords = grid_pos2coords(reward_grid_pos, self.scenario_width)
            agent.goal.set_pos(reward_coords, batch_index=env_index)

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
        for obstacle_id, obstacle_grid_pos in enumerate(self.obstacles_grid_positions):
            landmark = self.obstacles[obstacle_id]
            obstacle_coords = grid_pos2coords(obstacle_grid_pos, self.scenario_width)
            landmark.set_pos(obstacle_coords, batch_index=env_index)

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
    render_interactively(Scenario(), control_two_agents=True, obstacles_cfg='validation-random.yaml')
