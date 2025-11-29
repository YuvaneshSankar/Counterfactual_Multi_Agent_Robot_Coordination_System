

import numpy as np
import pybullet as p
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PyBulletRenderer:


    def __init__(
        self,
        client: int,
        warehouse_width: float = 100.0,
        warehouse_height: float = 80.0,
    ):

        self.client = client
        self.warehouse_width = warehouse_width
        self.warehouse_height = warehouse_height

        # Visual elements
        self.debug_items: List[int] = []
        self.text_items: List[int] = []

        # Camera state
        self.camera_distance = 150.0
        self.camera_yaw = 0.0
        self.camera_pitch = -60.0
        self.camera_target = [warehouse_width / 2, warehouse_height / 2, 0]

        self._setup_camera()

        logger.info("PyBulletRenderer initialized")

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target,
            physicsClientId=self.client
        )

        # Configure GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)

    def render_robots(self, robots: List) -> None:

        for robot in robots:
            pos = robot.get_position()
            battery = robot.get_battery_level()

            # Battery level color (green to red)
            color = self._battery_to_color(battery)

            # Draw battery indicator above robot
            self._draw_battery_indicator(pos[:2], battery, color)

            # Draw task indicator if robot has task
            if robot.has_task:
                self._draw_task_indicator(pos[:2])

    def render_tasks(self, tasks: List[Tuple]) -> None:

        for task in tasks:
            pickup_loc = task[:2]
            delivery_loc = task[3:5]
            priority = task[2]

            # Draw pickup location (blue)
            self._draw_sphere(pickup_loc, radius=1.0, color=[0, 0, 1, 0.5])

            # Draw delivery location (green)
            self._draw_sphere(delivery_loc, radius=1.0, color=[0, 1, 0, 0.5])

            # Draw line connecting pickup to delivery
            self._draw_line(pickup_loc, delivery_loc, color=[0.5, 0.5, 0.5])

    def render_paths(self, robot_id: int, path: List[np.ndarray]) -> None:

        if len(path) < 2:
            return

        # Color based on robot ID
        color = self._robot_id_to_color(robot_id)

        # Draw path segments
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            self._draw_line(start, end, color=color, width=2.0)

    def render_communication_links(self, robots: List, connectivity: Dict) -> None:

        for robot in robots:
            robot_id = robot.robot_id
            pos_i = robot.get_position()[:2]

            for connected_id in connectivity.get(robot_id, []):
                if connected_id < len(robots):
                    pos_j = robots[connected_id].get_position()[:2]

                    # Draw communication link (dashed line)
                    self._draw_line(
                        pos_i, pos_j,
                        color=[0.5, 0.5, 1, 0.3],
                        width=1.0
                    )

    def render_collision_warnings(self, collisions: List[Tuple[int, int]], robots: List) -> None:

        for robot_id1, robot_id2 in collisions:
            if robot_id1 < len(robots) and robot_id2 < len(robots):
                pos1 = robots[robot_id1].get_position()[:2]
                pos2 = robots[robot_id2].get_position()[:2]

                # Draw red warning sphere at midpoint
                midpoint = (np.array(pos1) + np.array(pos2)) / 2
                self._draw_sphere(midpoint, radius=0.5, color=[1, 0, 0, 0.8])

    def render_charging_stations(self, stations: List[Dict]) -> None:

        for station in stations:
            pos = (station['x'], station['y'])
            available = station.get('available', True)

            # Green if available, red if occupied
            color = [0, 1, 0, 0.6] if available else [1, 0, 0, 0.6]

            self._draw_sphere(pos, radius=1.5, color=color)

    def render_text_overlay(self, text: str, position: Tuple[float, float]) -> None:

        text_id = p.addUserDebugText(
            text,
            textPosition=[position[0], position[1], 2.0],
            textColorRGB=[1, 1, 1],
            textSize=1.0,
            physicsClientId=self.client
        )
        self.text_items.append(text_id)

    def clear_debug_items(self) -> None:
        for item_id in self.debug_items:
            try:
                p.removeUserDebugItem(item_id, physicsClientId=self.client)
            except:
                pass

        for text_id in self.text_items:
            try:
                p.removeUserDebugItem(text_id, physicsClientId=self.client)
            except:
                pass

        self.debug_items = []
        self.text_items = []

    def update_camera(
        self,
        distance: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        target: Optional[List[float]] = None
    ) -> None:

        if distance is not None:
            self.camera_distance = distance
        if yaw is not None:
            self.camera_yaw = yaw
        if pitch is not None:
            self.camera_pitch = pitch
        if target is not None:
            self.camera_target = target

        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target,
            physicsClientId=self.client
        )

    def follow_robot(self, robot) -> None:

        pos = robot.get_position()
        self.update_camera(target=[pos[0], pos[1], 0])

    def _draw_sphere(
        self,
        position: Tuple[float, float],
        radius: float = 1.0,
        color: List[float] = [1, 0, 0, 0.5]
    ) -> int:
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            physicsClientId=self.client
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=[position[0], position[1], 0.5],
            physicsClientId=self.client
        )

        self.debug_items.append(body_id)
        return body_id

    def _draw_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: List[float] = [1, 0, 0],
        width: float = 2.0
    ) -> int:
        line_id = p.addUserDebugLine(
            lineFromXYZ=[start[0], start[1], 0.5],
            lineToXYZ=[end[0], end[1], 0.5],
            lineColorRGB=color[:3],
            lineWidth=width,
            physicsClientId=self.client
        )

        self.debug_items.append(line_id)
        return line_id

    def _draw_battery_indicator(
        self,
        position: np.ndarray,
        battery: float,
        color: List[float]
    ) -> None:
        # Draw bar
        bar_length = 2.0 * (battery / 100.0)
        bar_height = 0.2

        # Background bar (gray)
        p.addUserDebugLine(
            lineFromXYZ=[position[0] - 1.0, position[1], 3.0],
            lineToXYZ=[position[0] + 1.0, position[1], 3.0],
            lineColorRGB=[0.3, 0.3, 0.3],
            lineWidth=5.0,
            physicsClientId=self.client
        )

        # Battery level bar (colored)
        line_id = p.addUserDebugLine(
            lineFromXYZ=[position[0] - 1.0, position[1], 3.0],
            lineToXYZ=[position[0] - 1.0 + bar_length, position[1], 3.0],
            lineColorRGB=color[:3],
            lineWidth=5.0,
            physicsClientId=self.client
        )

        self.debug_items.append(line_id)

    def _draw_task_indicator(self, position: np.ndarray) -> None:
        # Draw small circle above robot
        circle_id = p.addUserDebugLine(
            lineFromXYZ=[position[0], position[1], 3.5],
            lineToXYZ=[position[0], position[1], 3.5],
            lineColorRGB=[1, 1, 0],
            lineWidth=10.0,
            physicsClientId=self.client
        )

        self.debug_items.append(circle_id)

    def _battery_to_color(self, battery: float) -> List[float]:
        # Green (high) to red (low)
        normalized = battery / 100.0
        red = 1.0 - normalized
        green = normalized
        blue = 0.0

        return [red, green, blue, 1.0]

    def _robot_id_to_color(self, robot_id: int) -> List[float]:
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]

        return colors[robot_id % len(colors)]
