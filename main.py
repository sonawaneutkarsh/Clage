import pygame as pg
import random
import genome
import grid
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Config:
    WIDTH: int = 800
    HEIGHT: int = 600
    CELL_SIZE: int = 16
    FPS: int = 10
    INIT_ORGS: int = 5
    INIT_FOOD: int = 20
    FOOD_ENERGY: float = 0.5
    METABOLISM: float = 0.01
    REPRODUCTION_THRESHOLD: float = 1.5
    FOOD_SPAWN_RATE: float = 0.05

class Organism:
    def __init__(self, x: int, y: int, genome: Optional[genome.Genome] = None):
        self.x = x
        self.y = y
        self.energy = 1.0
        self.age = 0
        self.genome = genome or genome.Model(input_size=8, output_size=4).genome
        self.direction = random.randint(0, 3)  # 0: up, 1: right, 2: down, 3: left

    def update(self, world: 'World') -> Optional['Organism']:
        self.age += 1
        self.energy -= Config.METABOLISM
        
        # Get inputs from environment
        inputs = self.get_inputs(world)
        
        # Process through neural network
        outputs = self.genome.activate(inputs)
        
        # Execute actions based on outputs
        self.process_outputs(outputs, world)
        
        # Check for death
        if self.energy <= 0:
            return None
        
        # Check for reproduction
        if self.energy >= Config.REPRODUCTION_THRESHOLD:
            return self.reproduce()
        
        return None

    def get_inputs(self, world: 'World') -> List[float]:
        """Get 8 input values for neural network:
        0-3: Food presence in each direction
        4-7: Other organisms in each direction
        """
        inputs = [0.0] * 8
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        
        for i, (dx, dy) in enumerate(directions):
            # Check for food
            for dist in range(1, 4):
                nx, ny = (self.x + dx * dist) % world.grid.cols, (self.y + dy * dist) % world.grid.rows
                if cell := world.grid.get_cell(nx, ny):
                    if isinstance(cell, Food):
                        inputs[i] = 1.0 / dist  # Closer food = stronger signal
                        break
            
            # Check for other organisms
            for dist in range(1, 4):
                nx, ny = (self.x + dx * dist) % world.grid.cols, (self.y + dy * dist) % world.grid.rows
                if cell := world.grid.get_cell(nx, ny):
                    if isinstance(cell, Organism) and cell is not self:
                        inputs[i + 4] = 1.0 / dist
                        break
        
        return inputs

    def process_outputs(self, outputs: List[float], world: 'World'):
        """Process 4 output values from neural network:
        0: Move forward
        1: Turn left
        2: Turn right
        3: Eat
        """
        # Determine strongest output
        action = outputs.index(max(outputs))
        
        if action == 0:  # Move
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
            new_x = (self.x + dx) % world.grid.cols
            new_y = (self.y + dy) % world.grid.rows
            
            # Only move if cell is empty or has food
            if cell := world.grid.get_cell(new_x, new_y):
                if isinstance(cell, Food):
                    self.energy += cell.energy
                    world.foods.remove(cell)
            else:
                world.grid.set_cell(self.x, self.y, None)
                self.x, self.y = new_x, new_y
                world.grid.set_cell(self.x, self.y, self)
        
        elif action == 1:  # Turn left
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # Turn right
            self.direction = (self.direction + 1) % 4
        elif action == 3:  # Eat
            # Check adjacent cells for food
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = (self.x + dx) % world.grid.cols, (self.y + dy) % world.grid.rows
                if cell := world.grid.get_cell(nx, ny):
                    if isinstance(cell, Food):
                        self.energy += cell.energy
                        world.foods.remove(cell)
                        break

    def reproduce(self) -> 'Organism':
        self.energy /= 2
        offspring = Organism(self.x, self.y, self.genome.copy())
        offspring.genome.mutate()
        return offspring

    def draw(self, surface: pg.Surface, x: int, y: int):
        size = Config.CELL_SIZE - 4
        color = (
            min(255, int(100 + self.energy * 100)),
            min(255, int(200 - self.energy * 100)),
            100
        )
        pg.draw.rect(surface, color, (x + 2, y + 2, size, size))
        
        # Draw direction indicator
        center = (x + Config.CELL_SIZE // 2, y + Config.CELL_SIZE // 2)
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
        end = (center[0] + dx * (Config.CELL_SIZE // 2 - 2), 
               center[1] + dy * (Config.CELL_SIZE // 2 - 2))
        pg.draw.line(surface, (0, 0, 0), center, end, 2)

class Food:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.energy = Config.FOOD_ENERGY

    def draw(self, surface: pg.Surface, x: int, y: int):
        size = Config.CELL_SIZE - 8
        pg.draw.rect(surface, (255, 100, 100), (x + 4, y + 4, size, size))

class World:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((Config.WIDTH, Config.HEIGHT))
        pg.display.set_caption("Evolution Simulation")
        self.clock = pg.time.Clock()
        self.grid = grid.Grid(Config.WIDTH, Config.HEIGHT, Config.CELL_SIZE)
        self.organisms: List[Organism] = []
        self.foods: List[Food] = []
        self.initialize()

    def initialize(self):
        """Create initial organisms and food"""
        for _ in range(Config.INIT_ORGS):
            if pos := self.grid.find_empty_cell():
                org = Organism(pos[0], pos[1])
                self.organisms.append(org)
                self.grid.set_cell(pos[0], pos[1], org)

        for _ in range(Config.INIT_FOOD):
            if pos := self.grid.find_empty_cell():
                food = Food(pos[0], pos[1])
                self.foods.append(food)
                self.grid.set_cell(pos[0], pos[1], food)

    def update(self):
        """Update all entities in the world"""
        # Spawn new food occasionally
        if random.random() < Config.FOOD_SPAWN_RATE:
            if pos := self.grid.find_empty_cell():
                food = Food(pos[0], pos[1])
                self.foods.append(food)
                self.grid.set_cell(pos[0], pos[1], food)

        # Update organisms
        new_organisms = []
        dead_organisms = []

        for org in self.organisms:
            # Remove from grid temporarily
            self.grid.set_cell(org.x, org.y, None)
            
            # Update organism
            if baby := org.update(self):
                new_organisms.append(baby)
            
            # Check if organism died
            if org.energy <= 0:
                dead_organisms.append(org)
            else:
                # Put back in grid
                self.grid.set_cell(org.x, org.y, org)

        # Add new organisms
        for baby in new_organisms:
            if self.grid.get_cell(baby.x, baby.y) is None:
                self.organisms.append(baby)
                self.grid.set_cell(baby.x, baby.y, baby)

        # Remove dead organisms
        for org in dead_organisms:
            self.organisms.remove(org)

    def draw(self):
        """Draw the entire world"""
        self.screen.fill((0, 0, 0))
        self.grid.draw(self.screen)
        self.grid.draw_cells(self.screen)
        
        # Display stats
        font = pg.font.SysFont('Arial', 16)
        stats = [
            f"Organisms: {len(self.organisms)}",
            f"Food: {len(self.foods)}",
            f"Avg Energy: {sum(o.energy for o in self.organisms)/len(self.organisms):.2f}" if self.organisms else "0"
        ]
        for i, stat in enumerate(stats):
            text = font.render(stat, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 20))
        
        pg.display.flip()

    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_r:
                        self.__init__()  # Reset simulation

            self.update()
            self.draw()
            self.clock.tick(Config.FPS)

        pg.quit()

if __name__ == "__main__":
    world = World()
    world.run()