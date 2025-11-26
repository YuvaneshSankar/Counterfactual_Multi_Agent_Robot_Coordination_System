

import numpy as np
import pybullet as p
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class WarehouseLayout:


    def __init__(
        self,
        width: float = 100.0,
        height: float = 80.0,
        num_shelves: int = 8,
        num_charging_stations: int = 2,
        num_delivery_zones: int = 3,
        client: int = None,
        shelf_width: float = 3.0,
        shelf_height: float = 2.0,
        aisle_width: float = 5.0,
    ):

        self.width = width
        self.height = height
        self.num_shelves = num_shelves
        self.num_charging_stations = num_charging_stations
        self.num_delivery_zones = num_delivery_zones
        self.client = client
        self.shelf_width = shelf_width
        self.shelf_height = shelf_height
        self.aisle_width = aisle_width

        # Storage for layout elements
        self.shelves: List[Dict] = []
        self.charging_stations: List[Dict] = []
        self.delivery_zones: List[Dict] = []
        self.wall_ids: List[int] = []
        self.shelf_ids: List[int] = []

        # Generate layout
        self._generate_layout()

        logger.info(
            f"WarehouseLayout created: {width}x{height}, "
            f"{num_shelves} shelves, {num_charging_stations} charging stations"
        )

    def _generate_layout(self):
        self._create_shelves()
        self._create_charging_stations()
        self._create_delivery_zones()

    def _create_shelves(self):
        margin = 5.0
        usable_width = self.width - 2 * margin
        usable_height = self.height - 2 * margin

        # Arrange shelves in grid
        shelf_grid_x = int(np.sqrt(self.num_shelves))
        shelf_grid_y = (self.num_shelves + shelf_grid_x - 1) // shelf_grid_x

        shelf_spacing_x = usable_width / (shelf_grid_x + 1)
        shelf_spacing_y = usable_height / (shelf_grid_y + 1)

        shelf_idx = 0
        for i in range(shelf_grid_x):
            for j in range(shelf_grid_y):
                if shelf_idx >= self.num_shelves:
                    break

                x = margin + (i + 1) * shelf_spacing_x
                y = margin + (j + 1) * shelf_spacing_y

                # Create shelf as collision box
                shelf = {
                    'x': x - self.shelf_width / 2,
                    'y': y - self.shelf_height / 2,
                    'width': self.shelf_width,
                    'height': self.shelf_height,
                    'id': self._create_shelf_body(x, y)
                }
                self.shelves.append(shelf)
                self.shelf_ids.append(shelf['id'])
                shelf_idx += 1

    def _create_shelf_body(self, x: float, y: float) -> int:

        if self.client is None:
            return -1

        # Create collision shape
        half_width = self.shelf_width / 2
        half_height = self.shelf_height / 2

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_width, half_height, 1.0],
            physicsClientId=self.client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_width, half_height, 1.0],
            rgbaColor=[0.5, 0.5, 0.5, 0.7],
            physicsClientId=self.client
        )

        # Create body
        body_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, 1.0],
            physicsClientId=self.client
        )

        return body_id

    def _create_charging_stations(self):
        for i in range(self.num_charging_stations):
            if i % 2 == 0:
                # Charging stations on left side
                x = 2.0
                y = 5.0 + i * (self.height - 10.0) / max(1, self.num_charging_stations - 1)
            else:
                # Charging stations on right side
                x = self.width - 2.0
                y = 5.0 + i * (self.height - 10.0) / max(1, self.num_charging_stations - 1)

            station = {
                'x': x,
                'y': y,
                'radius': 1.0,
                'id': i,
                'available': True,
            }
            self.charging_stations.append(station)

    def _create_delivery_zones(self):
        margin = 5.0

        if self.num_delivery_zones == 1:
            positions = [(self.width / 2, margin)]
        elif self.num_delivery_zones == 2:
            positions = [
                (margin, self.height / 2),
                (self.width - margin, self.height / 2)
            ]
        elif self.num_delivery_zones == 3:
            positions = [
                (margin, self.height / 2),
                (self.width - margin, self.height / 2),
                (self.width / 2, self.height - margin)
            ]
        else:
            # Grid arrangement for 4+
            grid_x = int(np.sqrt(self.num_delivery_zones))
            grid_y = (self.num_delivery_zones + grid_x - 1) // grid_x

            spacing_x = (self.width - 2*margin) / (grid_x + 1)
            spacing_y = (self.height - 2*margin) / (grid_y + 1)

            positions = []
            zone_idx = 0
            for i in range(grid_x):
                for j in range(grid_y):
                    if zone_idx >= self.num_delivery_zones:
                        break
                    x = margin + (i + 1) * spacing_x
                    y = margin + (j + 1) * spacing_y
                    positions.append((x, y))
                    zone_idx += 1

        for i, (x, y) in enumerate(positions):
            zone = {
                'x': x,
                'y': y,
                'radius': 1.5,
                'id': i,
                'active_deliveries': 0,
            }
            self.delivery_zones.append(zone)

    def build(self):

        if self.client is None:
            logger.warning("PyBullet client not available, skipping physical build")
            return

        # Create walls
        self._create_walls()

    def _create_walls(self):
        wall_thickness = 0.1
        wall_height = 0.5

        # Bottom wall
        self._create_wall(
            self.width / 2, 0,
            self.width + 2*wall_thickness, wall_thickness, wall_height
        )

        # Top wall
        self._create_wall(
            self.width / 2, self.height,
            self.width + 2*wall_thickness, wall_thickness, wall_height
        )

        # Left wall
        self._create_wall(
            0, self.height / 2,
            wall_thickness, self.height + 2*wall_thickness, wall_height
        )

        # Right wall
        self._create_wall(
            self.width, self.height / 2,
            wall_thickness, self.height + 2*wall_thickness, wall_height
        )

    def _create_wall(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        wall_height: float
    ) -> int:

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[width/2, height/2, wall_height/2],
            physicsClientId=self.client
        )

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[width/2, height/2, wall_height/2],
            rgbaColor=[0.3, 0.3, 0.3, 1.0],
            physicsClientId=self.client
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, wall_height/2],
            physicsClientId=self.client
        )

        self.wall_ids.append(body_id)
        return body_id

    def get_nearest_charging_station(self, position: Tuple[float, float]) -> Optional[Dict]:

        available_stations = [s for s in self.charging_stations if s['available']]

        if not available_stations:
            return None

        nearest = min(
            available_stations,
            key=lambda s: np.sqrt((s['x'] - position[0])**2 + (s['y'] - position[1])**2)
        )

        return nearest

    def get_nearest_delivery_zone(self, position: Tuple[float, float]) -> Optional[Dict]:

        if not self.delivery_zones:
            return None

        nearest = min(
            self.delivery_zones,
            key=lambda z: np.sqrt((z['x'] - position[0])**2 + (z['y'] - position[1])**2)
        )

        return nearest

    def is_valid_position(
        self,
        position: Tuple[float, float],
        clearance: float = 1.0
    ) -> bool:

        x, y = position

        # Check bounds
        if x < clearance or x > self.width - clearance:
            return False
        if y < clearance or y > self.height - clearance:
            return False

        # Check shelf collisions
        for shelf in self.shelves:
            shelf_x = shelf['x']
            shelf_y = shelf['y']
            shelf_w = shelf['width']
            shelf_h = shelf['height']

            if (shelf_x - clearance < x < shelf_x + shelf_w + clearance and
                shelf_y - clearance < y < shelf_y + shelf_h + clearance):
                return False

        return True

    def get_free_positions(
        self,
        num_positions: int = 10,
        clearance: float = 2.0
    ) -> List[Tuple[float, float]]:
 
        positions = []
        max_attempts = 1000
        attempts = 0

        while len(positions) < num_positions and attempts < max_attempts:
            x = np.random.uniform(clearance, self.width - clearance)
            y = np.random.uniform(clearance, self.height - clearance)

            if self.is_valid_position((x, y), clearance):
                positions.append((x, y))

            attempts += 1

        return positions

    def get_layout_info(self) -> Dict:
        """Get information about warehouse layout."""
        return {
            'width': self.width,
            'height': self.height,
            'num_shelves': len(self.shelves),
            'num_charging_stations': len(self.charging_stations),
            'num_delivery_zones': len(self.delivery_zones),
            'shelf_width': self.shelf_width,
            'shelf_height': self.shelf_height,
            'aisle_width': self.aisle_width,
        }
