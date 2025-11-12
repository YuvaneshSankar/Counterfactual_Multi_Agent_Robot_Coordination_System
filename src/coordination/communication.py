"""
Communication Module - Inter-Robot Message Passing

Enables robots to share information about tasks, positions, and intentions.
Implements message queues and broadcast protocols.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RobotCommunicationNetwork:
    """
    Communication network for multi-robot system.

    Features:
    - Message broadcasting
    - Communication ranges
    - Message queue management
    - Latency simulation
    """

    def __init__(
        self,
        num_robots: int = 5,
        communication_range: float = 30.0,
        max_messages_per_robot: int = 100,
    ):
        """
        Initialize communication network.

        Args:
            num_robots: Number of robots
            communication_range: Maximum communication distance
            max_messages_per_robot: Max queued messages per robot
        """
        self.num_robots = num_robots
        self.communication_range = communication_range
        self.max_messages = max_messages_per_robot

        # Message queues per robot
        self.message_queues: Dict[int, List[Dict]] = {i: [] for i in range(num_robots)}

        # Communication graph
        self.connectivity: Dict[int, List[int]] = {i: [] for i in range(num_robots)}

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0

        logger.info(
            f"RobotCommunicationNetwork initialized: {num_robots} robots, "
            f"range={communication_range}m"
        )

    def update_connectivity(self, robots: List):
        """
        Update communication connectivity based on robot positions.

        Args:
            robots: List of robot objects
        """
        self.connectivity = {i: [] for i in range(self.num_robots)}

        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                pos_i = robots[i].get_position()[:2]
                pos_j = robots[j].get_position()[:2]

                distance = np.linalg.norm(pos_i - pos_j)

                if distance <= self.communication_range:
                    self.connectivity[i].append(j)
                    self.connectivity[j].append(i)

    def broadcast_message(
        self,
        sender_id: int,
        message_type: str,
        payload: Dict,
    ):
        """
        Broadcast message to all connected robots.

        Args:
            sender_id: ID of sending robot
            message_type: Type of message
            payload: Message payload dictionary
        """
        message = {
            'sender': sender_id,
            'type': message_type,
            'payload': payload,
            'timestamp': 0,  # Could track real time
        }

        # Send to all connected robots
        for receiver_id in self.connectivity.get(sender_id, []):
            if len(self.message_queues[receiver_id]) < self.max_messages:
                self.message_queues[receiver_id].append(message)
                self.messages_sent += 1

    def send_message(
        self,
        sender_id: int,
        receiver_id: int,
        message_type: str,
        payload: Dict,
    ) -> bool:
        """
        Send direct message to specific robot.

        Args:
            sender_id: Sender robot ID
            receiver_id: Receiver robot ID
            message_type: Message type
            payload: Message payload

        Returns:
            True if message sent successfully
        """
        # Check if in communication range
        if receiver_id not in self.connectivity.get(sender_id, []):
            return False

        message = {
            'sender': sender_id,
            'type': message_type,
            'payload': payload,
            'timestamp': 0,
        }

        if len(self.message_queues[receiver_id]) < self.max_messages:
            self.message_queues[receiver_id].append(message)
            self.messages_received += 1
            return True

        return False

    def get_messages(self, robot_id: int, message_type: Optional[str] = None) -> List[Dict]:
        """
        Get all messages for a robot.

        Args:
            robot_id: Robot ID
            message_type: Filter by message type (optional)

        Returns:
            List of messages
        """
        messages = self.message_queues[robot_id]

        if message_type is not None:
            messages = [m for m in messages if m['type'] == message_type]

        return messages.copy()

    def clear_messages(self, robot_id: int):
        """Clear all messages for a robot."""
        self.message_queues[robot_id] = []

    def get_connectivity_matrix(self) -> np.ndarray:
        """
        Get communication connectivity matrix.

        Returns:
            Connectivity matrix (num_robots x num_robots)
        """
        matrix = np.zeros((self.num_robots, self.num_robots), dtype=np.uint8)

        for i in range(self.num_robots):
            for j in self.connectivity.get(i, []):
                matrix[i, j] = 1

        return matrix


class MessageTypes:
    """Standard message types for robot communication."""

    # Status messages
    ROBOT_STATUS = 'robot_status'
    TASK_UPDATE = 'task_update'
    BATTERY_STATUS = 'battery_status'

    # Coordination messages
    TASK_OFFER = 'task_offer'
    TASK_ACCEPT = 'task_accept'
    TASK_COMPLETE = 'task_complete'

    # Safety messages
    COLLISION_WARNING = 'collision_warning'
    EMERGENCY_STOP = 'emergency_stop'

    # Navigation messages
    POSITION_UPDATE = 'position_update'
    TRAJECTORY_SHARE = 'trajectory_share'


class DecentralizedCoordinator:
    """
    Decentralized coordination using local communication.

    Robots make decisions based on local information from nearby robots.
    """

    def __init__(self, robot_id: int, communication_network: RobotCommunicationNetwork):
        """
        Initialize decentralized coordinator.

        Args:
            robot_id: This robot's ID
            communication_network: Shared communication network
        """
        self.robot_id = robot_id
        self.network = communication_network

        # Local knowledge
        self.known_robots: Dict[int, Dict] = {}
        self.known_tasks: List[Tuple] = []

    def update_local_knowledge(self):
        """Update local knowledge from messages."""
        messages = self.network.get_messages(self.robot_id)

        for msg in messages:
            if msg['type'] == MessageTypes.ROBOT_STATUS:
                payload = msg['payload']
                self.known_robots[msg['sender']] = {
                    'position': payload.get('position'),
                    'velocity': payload.get('velocity'),
                    'battery': payload.get('battery'),
                }

            elif msg['type'] == MessageTypes.TASK_UPDATE:
                payload = msg['payload']
                self.known_tasks = payload.get('tasks', [])

        self.network.clear_messages(self.robot_id)

    def broadcast_status(self, robot):
        """Broadcast robot status to neighbors."""
        payload = {
            'position': robot.get_position().tolist(),
            'velocity': robot.get_velocity().tolist(),
            'battery': robot.get_battery_level(),
            'has_task': robot.has_task,
        }

        self.network.broadcast_message(
            self.robot_id,
            MessageTypes.ROBOT_STATUS,
            payload
        )

    def get_nearby_robots(self) -> Dict[int, Dict]:
        """Get information about nearby robots."""
        return self.known_robots.copy()

    def get_nearest_neighbor(self) -> Optional[Tuple[int, Dict]]:
        """Get nearest known robot."""
        if not self.known_robots:
            return None

        min_distance = float('inf')
        nearest = None

        for robot_id, info in self.known_robots.items():
            if info['position'] is not None:
                distance = np.linalg.norm(np.array(info['position'][:2]))
                if distance < min_distance:
                    min_distance = distance
                    nearest = (robot_id, info)

        return nearest
