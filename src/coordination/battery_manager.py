

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BatteryManager:


    def __init__(
        self,
        num_robots: int = 5,
        battery_capacity: float = 100.0,
        charge_threshold: float = 30.0,
        critical_threshold: float = 10.0,
    ):

        self.num_robots = num_robots
        self.battery_capacity = battery_capacity
        self.charge_threshold = charge_threshold
        self.critical_threshold = critical_threshold

        # Battery tracking
        self.battery_levels: Dict[int, float] = {i: battery_capacity for i in range(num_robots)}
        self.charging_robots: Dict[int, Optional[int]] = {i: None for i in range(num_robots)}

        # Statistics
        self.total_energy_consumed = 0.0
        self.total_energy_charged = 0.0
        self.charging_events = 0

        logger.info(
            f"BatteryManager initialized: capacity={battery_capacity}, "
            f"charge_threshold={charge_threshold}"
        )

    def update_battery_levels(self, robots: List):

        for robot in robots:
            self.battery_levels[robot.robot_id] = robot.get_battery_level()

    def get_battery_status(self, robot_id: int) -> Dict:

        battery = self.battery_levels.get(robot_id, 0)
        percent = (battery / self.battery_capacity) * 100

        status = 'critical' if battery < self.critical_threshold else \
                 'low' if battery < self.charge_threshold else \
                 'normal'

        return {
            'battery_level': battery,
            'battery_percent': percent,
            'status': status,
            'charging': self.charging_robots[robot_id] is not None,
        }

    def should_charge(self, robot_id: int) -> bool:

        battery = self.battery_levels.get(robot_id, 0)
        return battery < self.charge_threshold

    def get_robots_needing_charge(self) -> List[int]:

        robots = []
        for robot_id, battery in self.battery_levels.items():
            if battery < self.charge_threshold:
                robots.append(robot_id)

        return robots

    def route_to_charging_station(
        self,
        robot_id: int,
        warehouse_layout,
    ) -> Optional[Tuple[float, float]]:

        station = warehouse_layout.get_nearest_charging_station((0, 0))

        if station is None:
            return None

        self.charging_robots[robot_id] = station['id']
        return (station['x'], station['y'])

    def start_charging(self, robot_id: int, station_id: int):

        self.charging_robots[robot_id] = station_id
        self.charging_events += 1
        logger.info(f"Robot {robot_id} started charging at station {station_id}")

    def stop_charging(self, robot_id: int):

        self.charging_robots[robot_id] = None

    def update_charge(self, robot_id: int, charge_amount: float):

        current = self.battery_levels.get(robot_id, 0)
        self.battery_levels[robot_id] = min(current + charge_amount, self.battery_capacity)
        self.total_energy_charged += charge_amount

    def predict_battery_runtime(
        self,
        robot_id: int,
        mean_velocity: float = 1.0,
    ) -> float:

        battery = self.battery_levels.get(robot_id, 0)

        # Discharge rate is velocity-dependent (from config)
        discharge_rate = 0.1 * (1.0 + mean_velocity)

        runtime = battery / discharge_rate if discharge_rate > 0 else 0

        return runtime

    def get_fleet_energy_status(self) -> Dict:

        batteries = list(self.battery_levels.values())
        charging_count = sum(1 for s in self.charging_robots.values() if s is not None)

        return {
            'total_robots': self.num_robots,
            'avg_battery': np.mean(batteries),
            'min_battery': np.min(batteries),
            'max_battery': np.max(batteries),
            'robots_charging': charging_count,
            'robots_low_battery': len(self.get_robots_needing_charge()),
            'total_energy_consumed': self.total_energy_consumed,
            'total_energy_charged': self.total_energy_charged,
        }

    def optimize_charging_order(self, warehouse_layout) -> List[int]:

        robots_to_charge = self.get_robots_needing_charge()

        # Sort by battery level (lowest first)
        robots_to_charge.sort(
            key=lambda rid: self.battery_levels[rid]
        )

        return robots_to_charge

    def get_statistics(self) -> Dict:
        """Get battery management statistics."""
        return {
            'charging_events': self.charging_events,
            'total_energy_consumed': self.total_energy_consumed,
            'total_energy_charged': self.total_energy_charged,
            'avg_battery_level': np.mean(list(self.battery_levels.values())),
            'energy_efficiency': (
                self.total_energy_charged / self.total_energy_consumed
                if self.total_energy_consumed > 0 else 0
            ),
        }

    def reset(self):
        """Reset battery manager."""
        for robot_id in range(self.num_robots):
            self.battery_levels[robot_id] = self.battery_capacity
            self.charging_robots[robot_id] = None

        self.total_energy_consumed = 0.0
        self.total_energy_charged = 0.0
        self.charging_events = 0


class PowerOptimizer:


    def __init__(self):
        """Initialize power optimizer."""
        self.optimization_history: List[Dict] = []

    def optimize_velocity(
        self,
        current_battery: float,
        distance_to_target: float,
        time_budget: float,
    ) -> float:

        # Minimum velocity to reach goal
        min_velocity = distance_to_target / time_budget if time_budget > 0 else 0

        # Battery-limited velocity (higher battery allows higher speed)
        battery_factor = np.sqrt(current_battery / 100.0)  # Non-linear relationship
        max_velocity = 2.0 * battery_factor

        # Recommended velocity between min and max
        recommended = max(min_velocity, max_velocity * 0.5)
        recommended = min(recommended, max_velocity)

        return recommended

    def estimate_energy_cost(
        self,
        path_length: float,
        avg_velocity: float,
    ) -> float:

        # Energy = distance * velocity_factor
        # Higher velocity uses more energy (quadratic relationship)
        velocity_factor = 0.1 * (1.0 + avg_velocity)
        energy_cost = path_length * velocity_factor

        return energy_cost
