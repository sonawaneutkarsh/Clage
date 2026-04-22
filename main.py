import pygame as pg
import random
import neat
import os
import grid
from typing import List, Optional, Dict
from dataclasses import dataclass

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "neat_config.txt")

@dataclass
class EnvConfig:
    """
    Environmental parameters for the research experiment.
    Vary one at a time to isolate the effect on evolved behavior.

    Research conditions (change one variable per experiment run):
        Abundance:  FOOD_DENSITY=0.15, FOOD_SPAWN_RATE=0.15, MAX_FOOD=150
        Scarcity:   FOOD_DENSITY=0.02, FOOD_SPAWN_RATE=0.02, MAX_FOOD=20
        Crowded:    set pop_size in neat_config.txt to 100
    """
    WIDTH: int = 800
    HEIGHT: int = 600
    CELL_SIZE: int = 16
    FPS: int = 30
    FOOD_DENSITY: float = 0.05       # fraction of grid cells seeded with food at start
    FOOD_SPAWN_RATE: float = 0.08    # probability of new food appearing each tick
    MAX_FOOD: int = 80
    STEPS_PER_GENERATION: int = 400  # ticks per generation
    METABOLISM: float = 0.005        # energy lost per tick


DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W


class Food:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.energy = 0.5

    def draw(self, surface: pg.Surface, px: int, py: int):
        s = EnvConfig.CELL_SIZE - 6
        pg.draw.rect(surface, (220, 80, 80), (px + 3, py + 3, s, s))


class Agent:
    """
    Neural agent whose brain is a NEAT FeedForwardNetwork.
    Fitness = food eaten * 3 + ticks survived * 0.01
    """
    def __init__(self, x: int, y: int, net: neat.nn.FeedForwardNetwork, genome_id: int):
        self.x = x
        self.y = y
        self.net = net
        self.genome_id = genome_id
        self.energy = 1.0
        self.age = 0
        self.food_eaten = 0
        self.direction = random.randint(0, 3)
        self.alive = True

    def get_inputs(self, world_grid: grid.Grid, steps: int) -> List[float]:
        """
        10 inputs:
          0-3  food signal per direction  (1/dist, 0 if none within 3 cells)
          4-7  agent signal per direction
          8    own energy normalised to [0,1]
          9    age fraction of generation length
        """
        inputs = [0.0] * 10
        for i, (dx, dy) in enumerate(DIRECTIONS):
            for dist in range(1, 4):
                nx = (self.x + dx * dist) % world_grid.cols
                ny = (self.y + dy * dist) % world_grid.rows
                if isinstance(world_grid.get_cell(nx, ny), Food):
                    inputs[i] = 1.0 / dist
                    break
            for dist in range(1, 4):
                nx = (self.x + dx * dist) % world_grid.cols
                ny = (self.y + dy * dist) % world_grid.rows
                cell = world_grid.get_cell(nx, ny)
                if isinstance(cell, Agent) and cell is not self:
                    inputs[i + 4] = 1.0 / dist
                    break
        inputs[8] = min(self.energy / 2.0, 1.0)
        inputs[9] = min(self.age / steps, 1.0)
        return inputs

    def step(self, world_grid: grid.Grid, foods: List[Food], cfg: EnvConfig):
        if not self.alive:
            return

        self.age += 1
        self.energy -= cfg.METABOLISM

        outputs = self.net.activate(self.get_inputs(world_grid, cfg.STEPS_PER_GENERATION))
        action = outputs.index(max(outputs))

        if action == 0:  # move forward
            dx, dy = DIRECTIONS[self.direction]
            nx = (self.x + dx) % world_grid.cols
            ny = (self.y + dy) % world_grid.rows
            target = world_grid.get_cell(nx, ny)
            if target is None:
                world_grid.set_cell(self.x, self.y, None)
                self.x, self.y = nx, ny
                world_grid.set_cell(nx, ny, self)
            elif isinstance(target, Food):
                self.energy += target.energy
                self.food_eaten += 1
                foods.remove(target)
                world_grid.set_cell(nx, ny, None)
                world_grid.set_cell(self.x, self.y, None)
                self.x, self.y = nx, ny
                world_grid.set_cell(nx, ny, self)
        elif action == 1:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4
        elif action == 3:  # eat adjacent cell
            for dx, dy in DIRECTIONS:
                nx = (self.x + dx) % world_grid.cols
                ny = (self.y + dy) % world_grid.rows
                cell = world_grid.get_cell(nx, ny)
                if isinstance(cell, Food):
                    self.energy += cell.energy
                    self.food_eaten += 1
                    foods.remove(cell)
                    world_grid.set_cell(nx, ny, None)
                    break

        if self.energy <= 0:
            self.alive = False
            world_grid.set_cell(self.x, self.y, None)

    @property
    def fitness(self) -> float:
        return self.food_eaten * 3.0 + self.age * 0.01

    def draw(self, surface: pg.Surface, px: int, py: int):
        s = EnvConfig.CELL_SIZE - 4
        r = min(255, int(80 + self.energy * 120))
        g = min(255, int(180 - self.energy * 60))
        pg.draw.rect(surface, (r, g, 100), (px + 2, py + 2, s, s))
        cx = px + EnvConfig.CELL_SIZE // 2
        cy = py + EnvConfig.CELL_SIZE // 2
        dx, dy = DIRECTIONS[self.direction]
        ex = cx + dx * (EnvConfig.CELL_SIZE // 2 - 2)
        ey = cy + dy * (EnvConfig.CELL_SIZE // 2 - 2)
        pg.draw.line(surface, (0, 0, 0), (cx, cy), (ex, ey), 2)


class Simulation:
    """Runs one generation: evaluates all agents and returns their fitness scores."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.world_grid = grid.Grid(cfg.WIDTH, cfg.HEIGHT, cfg.CELL_SIZE)
        self.foods: List[Food] = []
        self.agents: List[Agent] = []

    def _reset(self, nets: List[neat.nn.FeedForwardNetwork], genome_ids: List[int]):
        self.world_grid.clear()
        self.foods.clear()
        self.agents.clear()

        n_food = int(self.world_grid.cols * self.world_grid.rows * self.cfg.FOOD_DENSITY)
        for _ in range(n_food):
            pos = self.world_grid.find_empty_cell()
            if pos:
                f = Food(pos[0], pos[1])
                self.foods.append(f)
                self.world_grid.set_cell(pos[0], pos[1], f)

        for net, gid in zip(nets, genome_ids):
            pos = self.world_grid.find_empty_cell()
            if pos:
                a = Agent(pos[0], pos[1], net, gid)
                self.agents.append(a)
                self.world_grid.set_cell(pos[0], pos[1], a)

    def run_generation(
        self,
        nets: List[neat.nn.FeedForwardNetwork],
        genome_ids: List[int],
        screen: Optional[pg.Surface],
        clock: Optional[pg.time.Clock],
    ) -> Dict[int, float]:
        self._reset(nets, genome_ids)

        for tick in range(self.cfg.STEPS_PER_GENERATION):
            if screen:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        raise SystemExit
                    if event.type == pg.KEYDOWN and event.key == pg.K_v:
                        screen = None  # press V to hide display for speed

            if random.random() < self.cfg.FOOD_SPAWN_RATE and len(self.foods) < self.cfg.MAX_FOOD:
                pos = self.world_grid.find_empty_cell()
                if pos:
                    f = Food(pos[0], pos[1])
                    self.foods.append(f)
                    self.world_grid.set_cell(pos[0], pos[1], f)

            for agent in self.agents:
                agent.step(self.world_grid, self.foods, self.cfg)

            if screen:
                screen.fill((15, 15, 20))
                self.world_grid.draw(screen)
                self.world_grid.draw_cells(screen)
                _draw_hud(screen, self.agents, tick, self.cfg)
                pg.display.flip()
                if clock:
                    clock.tick(self.cfg.FPS)

        return {a.genome_id: a.fitness for a in self.agents}


def _draw_hud(screen: pg.Surface, agents: List[Agent], tick: int, cfg: EnvConfig):
    font = pg.font.SysFont("Arial", 15)
    alive = sum(1 for a in agents if a.alive)
    avg_food = sum(a.food_eaten for a in agents) / max(len(agents), 1)
    for i, line in enumerate([
        f"Tick: {tick}/{cfg.STEPS_PER_GENERATION}",
        f"Alive: {alive}/{len(agents)}",
        f"Avg food eaten: {avg_food:.1f}",
        "Press V to hide display (faster)",
    ]):
        screen.blit(font.render(line, True, (220, 220, 220)), (10, 10 + i * 20))


def run_neat(cfg: EnvConfig, visualise: bool = True, max_generations: int = 200):
    neat_cfg = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    population = neat.Population(neat_cfg)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    screen: Optional[pg.Surface] = None
    clock: Optional[pg.time.Clock] = None
    if visualise:
        pg.init()
        screen = pg.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
        pg.display.set_caption("NEAT Evolution Simulation  |  V = toggle display")
        clock = pg.time.Clock()

    sim = Simulation(cfg)

    def eval_genomes(genomes, neat_cfg):
        nets, gids, genome_map = [], [], {}
        for gid, g in genomes:
            g.fitness = 0.0
            nets.append(neat.nn.FeedForwardNetwork.create(g, neat_cfg))
            gids.append(gid)
            genome_map[gid] = g

        for gid, fit in sim.run_generation(nets, gids, screen, clock).items():
            genome_map[gid].fitness = fit

    winner = population.run(eval_genomes, max_generations)
    print(f"\nBest genome fitness: {winner.fitness:.2f}")
    print(f"Nodes: {len(winner.nodes)}  Connections: {len(winner.connections)}")

    if visualise:
        pg.quit()

    return winner, stats


if __name__ == "__main__":
    cfg = EnvConfig()
    # Research: change ONE parameter per run to isolate its effect, e.g.:
    #   cfg.FOOD_DENSITY = 0.02   # scarcity condition
    #   cfg.FOOD_DENSITY = 0.15   # abundance condition
    run_neat(cfg, visualise=True, max_generations=200)
