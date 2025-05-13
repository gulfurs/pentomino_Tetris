from constants import RED, GREEN, BLUE, DARK_BLUE, PURPLE, YELLOW

PENTOMINOS = {
    'I': {
        'shape': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        'color': BLUE
    },
    'L': {
        'shape': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)],
        'color': DARK_BLUE
    },
    'P': {
        'shape': [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
        'color': PURPLE
    },
    'T': {
        'shape': [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
        'color': YELLOW
    },
    'V': {
        'shape': [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
        'color': GREEN
    }
}


class Pentomino:
    def __init__(self, shape_name=None):
        import random
        if shape_name is None:
            shape_name = random.choice(list(PENTOMINOS.keys()))
        
        self.shape_name = shape_name
        self.shape = PENTOMINOS[shape_name]['shape']
        self.color = PENTOMINOS[shape_name]['color']
        self.x = 0
        self.y = 0
        self.rotation = 0 
    
    def get_coords(self):
        rotated_shape = self._get_rotated_shape()
        return [(self.x + x, self.y + y) for x, y in rotated_shape]
    
    def _get_rotated_shape(self):
        if self.rotation == 0:
            return self.shape
        elif self.rotation == 1:
            return [(-y, x) for x, y in self.shape]  # 90 degrees
        elif self.rotation == 2:
            return [(-x, -y) for x, y in self.shape]  # 180 degrees
        else:  # rotation == 3
            return [(y, -x) for x, y in self.shape]  # 270 degrees
    
    def rotate(self, clockwise=True):
        if clockwise:
            self.rotation = (self.rotation + 1) % 4
        else:
            self.rotation = (self.rotation - 1) % 4
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy