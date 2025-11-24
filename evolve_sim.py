import pygame
import random
import hashlib
import genome
from typing import List, Tuple, Optional

# === CONFIG ===
WIDTH, HEIGHT = 1200, 750
GRID_SIZE = 40
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE
FPS = 10
INIT_ORGS = 10
INIT_FOOD = 30
FOOD_ENERGY = 0.5
METABOLISM = 0.02
REPRODUCTION_THRESHOLD = 1.8
MUTATION_RATE = 0.2

# === COLORS ===
BLACK = (0, 0, 0)
FOOD_COLOR = (255, 100, 100)
ORG_COLOR = (100, 200, 255)
VISION_COLOR = (100, 255, 100, 100)  # RGBA for transparency
TEXT_COLOR = (255, 255, 255)

class Food:
    def __init__(self, x: int, y: int, energy: float = FOOD_ENERGY):
        self.x = x
        self.y = y
        self.energy = energy

    def draw(self, screen):
        rect = pygame.Rect(
            self.x * GRID_SIZE + GRID_SIZE * 0.1,
            self.y * GRID_SIZE + GRID_SIZE * 0.1,
            GRID_SIZE * 0.8,
            GRID_SIZE * 0.8
        )
        pygame.draw.rect(screen, FOOD_COLOR, rect)

class Organism:
    _id_counter = 0

    def __init__(self, x: int, y: int, parent_genome: Optional[genome.Genome] = None):
        self.x = x
        self.y = y
        self.health = 1.0
        self.energy = self.health
        self.id = Organism._id_counter
        Organism._id_counter += 1
        
        # Initialize genome
        if parent_genome:
            self.genome = parent_genome.copy()
            self.genome.mutate(MUTATION_RATE)
        else:
            model = genome.Model(input_size=8, output_size=4)
            self.genome = model.genome
        
        # Unique visual identifier
        self.brain_hash = hashlib.sha384(f"{self.id}".encode()).hexdigest()
        self.brain_pixels = self._hash_to_pixels(self.brain_hash)
        self.vision_lines = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W

    def _hash_to_pixels(self, hash_string: str) -> List[Tuple[int, int, int]]:
        pixels = []
        for i in range(0, 96, 6):  # 16 RGB values from 96 hex chars
            r = int(hash_string[i:i+2], 16)
            g = int(hash_string[i+2:i+4], 16)
            b = int(hash_string[i+4:i+6], 16)
            pixels.append((r, g, b))
        return pixels

    def draw(self, screen):
        center_x = self.x * GRID_SIZE + GRID_SIZE // 2
        center_y = self.y * GRID_SIZE + GRID_SIZE // 2
        radius = GRID_SIZE // 2 - 2

        # Draw organism body (color based on energy)
        energy_color = (
            ORG_COLOR[0],
            min(255, ORG_COLOR[1] + int(self.energy * 50)),
            ORG_COLOR[2]
        )
        pygame.draw.circle(screen, energy_color, (center_x, center_y), radius)

        # Draw brain pattern (4x4 grid)
        start_x = center_x - 8
        start_y = center_y - 8
        for i, color in enumerate(self.brain_pixels[:16]):  # Only use first 16 colors
            x = start_x + (i % 4) * 4
            y = start_y + (i // 4) * 4
            pygame.draw.rect(screen, color, (x, y, 4, 4))

        # Draw vision lines (faded to show they're visual guides)
        for dx, dy in self.vision_lines:
            end_x = center_x + dx * GRID_SIZE * 2
            end_y = center_y + dy * GRID_SIZE * 2
            pygame.draw.line(screen, VISION_COLOR, (center_x, center_y), (end_x, end_y), 1)

    def update(self, foods: List['Food'], organisms: List['Organism']) -> Optional['Organism']:
        # Get inputs from environment
        inputs = self.get_inputs(foods, organisms)
        
        # Process through neural network
        outputs = self.genome.activate(inputs)
        
        # Execute actions based on outputs
        self.process_outputs(outputs)
        
        # Energy management
        self.energy -= METABOLISM
        
        # Check for death
        if self.energy <= 0:
            return None
        
        # Check for reproduction
        if self.energy >= REPRODUCTION_THRESHOLD:
            return self.reproduce()
        
        return None

    def get_inputs(self, foods: List['Food'], organisms: List['Organism']) -> List[float]:
        """Get 8 input values:
        0-3: Food presence in each direction
        4-7: Other organisms in each direction
        """
        inputs = [0.0] * 8
        
        for i, (dx, dy) in enumerate(self.vision_lines):
            # Check for food
            for dist in range(1, 4):  # Check 3 cells ahead
                nx, ny = (self.x + dx * dist) % COLS, (self.y + dy * dist) % ROWS
                for food in foods:
                    if food.x == nx and food.y == ny:
                        inputs[i] = 1.0 / dist  # Closer food = stronger signal
                        break
            
            # Check for other organisms
            for dist in range(1, 4):
                nx, ny = (self.x + dx * dist) % COLS, (self.y + dy * dist) % ROWS
                for org in organisms:
                    if org is not self and org.x == nx and org.y == ny:
                        inputs[i + 4] = 1.0 / dist
                        break
        
        return inputs

    def process_outputs(self, outputs: List[float]):
        """Process 4 output values:
        0: Move North
        1: Move East
        2: Move South
        3: Move West
        """
        action = outputs.index(max(outputs))
        dx, dy = self.vision_lines[action]
        self.x = (self.x + dx) % COLS
        self.y = (self.y + dy) % ROWS

    def reproduce(self) -> 'Organism':
        self.energy /= 2
        offspring = Organism(self.x, self.y, self.genome)
        return offspring

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evolution Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)

    # Initialize world
    organisms = [
        Organism(random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
        for _ in range(INIT_ORGS)
    ]
    foods = [
        Food(random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
        for _ in range(INIT_FOOD)
    ]

    running = True
    while running:
        clock.tick(FPS)
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset simulation
                    organisms = [
                        Organism(random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
                        for _ in range(INIT_ORGS)
                    ]
                    foods = [
                        Food(random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
                        for _ in range(INIT_FOOD)
                    ]

        # Update organisms
        new_organisms = []
        dead_organisms = []
        
        for org in organisms:
            if baby := org.update(foods, organisms):
                new_organisms.append(baby)
            if org.energy <= 0:
                dead_organisms.append(org)
        
        # Remove dead organisms and add new ones
        organisms = [o for o in organisms if o not in dead_organisms] + new_organisms

        # Spawn new food occasionally
        if random.random() < 0.03 and len(foods) < INIT_FOOD * 2:
            foods.append(Food(random.randint(0, COLS - 1), random.randint(0, ROWS - 1)))

        # Draw everything
        for food in foods:
            food.draw(screen)
        for org in organisms:
            org.draw(screen)

        # Display stats
        stats = [
            f"Organisms: {len(organisms)}",
            f"Food: {len(foods)}",
            f"Avg Energy: {sum(o.energy for o in organisms)/len(organisms):.2f}" if organisms else "0",
            f"Press R to reset"
        ]
        for i, stat in enumerate(stats):
            text = font.render(stat, True, TEXT_COLOR)
            screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()