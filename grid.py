import pygame as pg
from typing import Optional, Any, List

class Grid:
    def __init__(self, width: int, height: int, cell_size: int):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]

    def get_cell(self, cx: int, cy: int) -> Optional[Any]:
        """Get cell content at grid coordinates"""
        if 0 <= cx < self.cols and 0 <= cy < self.rows:
            return self.grid[cy][cx]
        return None

    def get_cell_xy(self, x: int, y: int) -> Optional[Any]:
        """Get cell content at pixel coordinates"""
        return self.get_cell(x // self.cell_size, y // self.cell_size)

    def set_cell(self, cx: int, cy: int, value: Any) -> bool:
        """Set cell content at grid coordinates"""
        if 0 <= cx < self.cols and 0 <= cy < self.rows:
            self.grid[cy][cx] = value
            return True
        return False

    def set_cell_xy(self, x: int, y: int, value: Any) -> bool:
        """Set cell content at pixel coordinates"""
        return self.set_cell(x // self.cell_size, y // self.cell_size, value)

    def clear(self) -> None:
        """Clear all cells"""
        self.grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]

    def find_empty_cell(self) -> Optional[tuple[int, int]]:
        """Find random empty cell coordinates"""
        empty_cells = [
            (x, y) 
            for y in range(self.rows) 
            for x in range(self.cols) 
            if self.grid[y][x] is None
        ]
        return choice(empty_cells) if empty_cells else None

    def draw(self, surface: pg.Surface) -> None:
        """Draw grid lines"""
        for x in range(0, self.width, self.cell_size):
            pg.draw.line(surface, (40, 40, 40), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pg.draw.line(surface, (40, 40, 40), (0, y), (self.width, y))

    def draw_cells(self, surface: pg.Surface) -> None:
        """Draw all cell contents"""
        for y in range(self.rows):
            for x in range(self.cols):
                if (cell := self.grid[y][x]) is not None:
                    if hasattr(cell, 'draw'):
                        cell.draw(surface, x * self.cell_size, y * self.cell_size)

    def get_neighbors(self, cx: int, cy: int, radius: int = 1) -> List[Any]:
        """Get neighboring cells within radius"""
        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if (cell := self.grid[ny][nx]) is not None:
                        neighbors.append(cell)
        return neighbors