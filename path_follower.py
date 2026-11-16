#!/usr/bin/env python3
"""Fast, simultaneous Task 3B follower for three robots carrying payload_gamma."""

import math
import numpy as np
import rclpy

from hb_description.srv import AttachPayload, DetachPayload
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ROBOTS = ("r1", "r2", "r3")
PAYLOAD = "payload_gamma"

SLOTS = {
    "r1": (0.0, -0.1827),
    "r2": (-0.1796, 0.0),
    "r3": (0.0, 0.1827),
}

YAWS = {
    "r1": math.pi / 2.0,
    "r2": 0.0,
    "r3": -math.pi / 2.0,
}

DESTINATION = (1.2192, 1.2192)

CONTROL_HZ = 50.0
MAX_WHEEL_SPEED = 30.0

# Task 3A-style fast holonomic controller.
MAX_LINEAR_SPEED = 0.50
MAX_ANGULAR_SPEED = 1.8
LINEAR_GAIN = 3.0
ANGULAR_GAIN = 3.0
BRAKING_ACCEL = 1.2

# Precision only in the final docking region.
DOCK_APPROACH_SPEED = 0.07
FINAL_APPROACH_DISTANCE = 0.15

WAYPOINT_TOLERANCE = 0.035
DOCK_TOLERANCE = 0.015
CARRY_TOLERANCE = 0.025
BACKAWAY_TOLERANCE = 0.015
YAW_TOLERANCE = math.radians(2.0)

# Carry the rigid formation more gently than the free robots approach.
CARRY_MAX_SPEED = 0.30
BACKAWAY_MAX_SPEED = 0.30

DOCK_RETRY_STEP = 0.002
DOCK_RETRY_MAX_IN = 0.120
DOCK_RETRY_MAX_OUT = 0.030
SERVICE_RETRY_INTERVAL = 0.5
LEAD = 0.10

LATCHED = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

IK_MATRIX = np.array([
    [33.9617805, 19.6078431, 2.5145098],
    [-33.9617805, 19.6078431, 2.5145098],
    [0.0, -39.2156863, 2.5145098],
], dtype=float)


# ---------------------------------------------------------------------------
# GEOMETRY AND PATH UTILITIES
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def path_to_traj(msg):
    """Convert Path into [(relative_time, x, y), ...]."""
    start = Time.from_msg(msg.header.stamp)
    result = []

    for pose in msg.poses:
        stamp = Time.from_msg(pose.header.stamp)
        t = (stamp - start).nanoseconds * 1e-9

        result.append((
            float(t),
            float(pose.pose.position.x),
            float(pose.pose.position.y),
        ))

    return result


def position_at(traj, t):
    if not traj:
        return None

    if t <= traj[0][0]:
        return traj[0][1], traj[0][2]

    for a, b in zip(traj, traj[1:]):
        if t <= b[0]:
            u = (t - a[0]) / max(1e-6, b[0] - a[0])

            return (
                a[1] + u * (b[1] - a[1]),
                a[2] + u * (b[2] - a[2]),
            )

    return traj[-1][1], traj[-1][2]


def arena_to_body(vx, vy, yaw):
    # Same coordinate conversion used by the Task 3A follower.
    return (
        math.cos(yaw) * vx + math.sin(yaw) * vy,
        -math.sin(yaw) * vx + math.cos(yaw) * vy,
    )


def body_to_wheels(vx, vy, wz):
    wheels = IK_MATRIX @ np.array(
        [vx, vy, wz],
        dtype=float,
    )

    peak = float(np.max(np.abs(wheels)))

    if peak > MAX_WHEEL_SPEED:
        wheels *= MAX_WHEEL_SPEED / peak

    return wheels.tolist()


# ---------------------------------------------------------------------------
# TASK 3A-STYLE FAST CONTROLLER
# ---------------------------------------------------------------------------

class FastController:
    """Distance-proportional velocity with braking and heading correction."""

    def calculate(
        self,
        pose,
        target,
        target_yaw,
        v_limit,
        slow_final=False,
    ):
        x, y, yaw = pose

        dx = target[0] - x
        dy = target[1] - y

        distance = math.hypot(dx, dy)

        if distance < 0.001:
            vx = 0.0
            vy = 0.0

        else:
            desired_speed = min(
                v_limit,
                LINEAR_GAIN * distance,
            )

            # Braking profile prevents high-speed overshoot.
            braking_speed = math.sqrt(
                max(0.0, 2.0 * BRAKING_ACCEL * distance)
            )

            desired_speed = min(
                desired_speed,
                braking_speed,
            )

            # Slow down only for the final docking approach.
            if (
                slow_final
                and distance < FINAL_APPROACH_DISTANCE
            ):
                desired_speed = min(
                    desired_speed,
                    DOCK_APPROACH_SPEED,
                )

            vx = desired_speed * dx / distance
            vy = desired_speed * dy / distance

        yaw_error = wrap(target_yaw - yaw)

        wz = clamp(
            ANGULAR_GAIN * yaw_error,
            -MAX_ANGULAR_SPEED,
            MAX_ANGULAR_SPEED,
        )

        body_vx, body_vy = arena_to_body(
            vx,
            vy,
            yaw,
        )

        wheels = body_to_wheels(
            body_vx,
            body_vy,
            wz,
        )

        return wheels, distance, yaw_error


# ---------------------------------------------------------------------------
# TEAM FOLLOWER
# ---------------------------------------------------------------------------

class TeamFollower(Node):

    def __init__(self):
        super().__init__("task_3b_team")

        self.pose = {}

        self.plans = {
            r: None for r in ROBOTS
        }

        self.carry = None

        self.phase = "waiting"
        self.phase_start = None

        self.waypoint = {
            r: 0 for r in ROBOTS
        }

        self.arrived = {
            r: False for r in ROBOTS
        }

        self.attached = {
            r: False for r in ROBOTS
        }

        self.detached = {
            r: False for r in ROBOTS
        }

        self.dock_bias = {
            r: 0.0 for r in ROBOTS
        }

        self.pending = {
            r: False for r in ROBOTS
        }

        self.last_request = {
            r: -math.inf for r in ROBOTS
        }

        self.controller = {
            r: FastController() for r in ROBOTS
        }

        self.wheel_pubs = {}
        self.attach_clients = {}
        self.detach_clients = {}

        for robot in ROBOTS:

            self.wheel_pubs[robot] = self.create_publisher(
                Float64MultiArray,
                f"/{robot}/wheel_commands",
                10,
            )

            self.create_subscription(
                Odometry,
                f"/{robot}/odom",
                lambda msg, r=robot: self.odom_callback(r, msg),
                10,
            )

            self.create_subscription(
                Path,
                f"/{robot}/plan",
                lambda msg, r=robot: self.plan_callback(r, msg),
                LATCHED,
            )

            self.attach_clients[robot] = self.create_client(
                AttachPayload,
                f"/{robot}/attach_payload",
            )

            self.detach_clients[robot] = self.create_client(
                DetachPayload,
                f"/{robot}/detach_payload",
            )

        self.create_subscription(
            Path,
            "/carry_plan",
            self.carry_callback,
            LATCHED,
        )

        self.create_timer(
            1.0 / CONTROL_HZ,
            self.tick,
        )

        self.get_logger().info(
            "TASK 3B FAST TEAM FOLLOWER INITIALIZED"
        )

        self.get_logger().info(
            f"Controller: {MAX_LINEAR_SPEED:.2f} m/s max, "
            f"{CONTROL_HZ:.0f} Hz, "
            f"wheel limit {MAX_WHEEL_SPEED:.0f} rad/s"
        )

        self.get_logger().info(
            "All three robots will approach their docking slots simultaneously."
        )

    # -----------------------------------------------------------------------
    # INPUT CALLBACKS
    # -----------------------------------------------------------------------

    def odom_callback(self, robot, msg):

        p = msg.pose.pose.position

        self.pose[robot] = (
            float(p.x),
            float(p.y),
            yaw_from_quat(msg.pose.pose.orientation),
        )

    def plan_callback(self, robot, msg):

        if msg.poses and self.plans[robot] is None:
            self.plans[robot] = path_to_traj(msg)
            self.try_start()

    def carry_callback(self, msg):

        if msg.poses and self.carry is None:
            self.carry = path_to_traj(msg)
            self.try_start()

    def try_start(self):

        if (
            self.phase == "waiting"
            and self.carry
            and all(self.plans[r] for r in ROBOTS)
        ):

            self.phase = "approach"

            self.get_logger().info(
                "All plans received. SIMULTANEOUS APPROACH STARTED."
            )

    # -----------------------------------------------------------------------
    # WHEEL COMMANDS
    # -----------------------------------------------------------------------

    def send(self, robot, wheels=(0.0, 0.0, 0.0)):

        self.wheel_pubs[robot].publish(
            Float64MultiArray(
                data=[float(v) for v in wheels]
            )
        )

    def stop_all(self):

        for robot in ROBOTS:
            self.send(robot)

    def drive(
        self,
        robot,
        target,
        target_yaw,
        v_limit,
        slow_final=False,
    ):

        wheels, distance, yaw_error = (
            self.controller[robot].calculate(
                self.pose[robot],
                target,
                target_yaw,
                v_limit,
                slow_final=slow_final,
            )
        )

        self.send(robot, wheels)

        return distance, yaw_error

    def dock_target(self, robot):
        """Final planned docking pose, adjusted by retry bias."""

        final = self.plans[robot][-1]

        tx = final[1]
        ty = final[2]

        sx, sy = SLOTS[robot]

        length = math.hypot(sx, sy)
        bias = self.dock_bias[robot]

        return (
            tx - bias * sx / length,
            ty - bias * sy / length,
        )

    # -----------------------------------------------------------------------
    # ASYNCHRONOUS ATTACH / DETACH SERVICES
    # -----------------------------------------------------------------------

    def request_service(self, robot, attach):

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.pending[robot]:
            return

        if (
            now - self.last_request[robot]
            < SERVICE_RETRY_INTERVAL
        ):
            return

        client = (
            self.attach_clients[robot]
            if attach
            else self.detach_clients[robot]
        )

        if not client.service_is_ready():
            return

        request_type = (
            AttachPayload.Request
            if attach
            else DetachPayload.Request
        )

        request = request_type()

        request.payload_name = PAYLOAD 

        self.pending[robot] = True
        self.last_request[robot] = now

        future = client.call_async(request)

        future.add_done_callback(
            lambda f, r=robot, a=attach:
                self.service_callback(r, a, f)
        )

    def service_callback(self, robot, attach, future):

        self.pending[robot] = False

        try:
            response = future.result()

        except Exception as exc:

            self.get_logger().error(
                f"{robot} "
                f"{'attach' if attach else 'detach'} "
                f"service error: {exc}"
            )

            return

        if response is not None and response.success:

            if attach:

                self.attached[robot] = True

                self.get_logger().info(
                    f"{robot}: ATTACHED {PAYLOAD} "
                    f"(distance {response.distance:.3f} m)"
                )

            else:

                self.detached[robot] = True

                self.get_logger().info(
                    f"{robot}: detached {PAYLOAD}"
                )

            return

        message = (
            response.message
            if response is not None
            else "no response"
        )

        self.get_logger().warning(
            f"{robot}: {message}"
        )

        if attach and response is not None:

            if "Not close enough" in message:

                self.dock_bias[robot] = min(
                    DOCK_RETRY_MAX_IN,
                    self.dock_bias[robot] + DOCK_RETRY_STEP,
                )

            elif "Too close" in message:

                self.dock_bias[robot] = max(
                    -DOCK_RETRY_MAX_OUT,
                    self.dock_bias[robot] - DOCK_RETRY_STEP,
                )

            elif "Not facing" in message:

                # Heading controller continues aligning.
                pass

    # -----------------------------------------------------------------------
    # APPROACH PHASE
    # -----------------------------------------------------------------------

    def approach_tick(self):
        """Follow all three approach paths in parallel."""

        for robot in ROBOTS:

            if self.arrived[robot]:
                self.send(robot)
                continue

            plan = self.plans[robot]

            index = min(
                self.waypoint[robot],
                len(plan) - 1,
            )

            final = index == len(plan) - 1

            if final:

                target = self.dock_target(robot)

            else:

                target = (
                    plan[index][1],
                    plan[index][2],
                )

            target_yaw = (
                YAWS[robot]
                if final
                else -math.pi / 2.0
            )

            tolerance = (
                DOCK_TOLERANCE
                if final
                else WAYPOINT_TOLERANCE
            )

            distance, yaw_error = self.drive(
                robot,
                target,
                target_yaw,
                MAX_LINEAR_SPEED,
                slow_final=final,
            )

            if final:

                if (
                    distance <= tolerance
                    and abs(yaw_error) <= YAW_TOLERANCE
                ):

                    self.arrived[robot] = True
                    self.send(robot)

                    self.get_logger().info(
                        f"{robot}: docked and aligned"
                    )

            elif distance <= tolerance:

                self.waypoint[robot] = min(
                    self.waypoint[robot] + 1,
                    len(plan) - 1,
                )

        if all(self.arrived.values()):

            self.phase = "attaching"

            self.get_logger().info(
                "ALL THREE DOCKED. REQUESTING THREE ATTACHMENTS."
            )

    # -----------------------------------------------------------------------
    # ATTACHING PHASE
    # -----------------------------------------------------------------------

    def attaching_tick(self):

        for robot in ROBOTS:

            if self.attached[robot]:
                self.send(robot)
                continue

            target = self.dock_target(robot)

            distance, yaw_error = self.drive(
                robot,
                target,
                YAWS[robot],
                DOCK_APPROACH_SPEED,
                slow_final=False,
            )

            if (
                distance <= DOCK_TOLERANCE
                and abs(yaw_error) <= YAW_TOLERANCE
            ):

                self.send(robot)

                self.request_service(
                    robot,
                    attach=True,
                )

        if all(self.attached.values()):

            self.phase = "carry"
            self.phase_start = self.get_clock().now()

            self.get_logger().info(
                "ALL THREE ATTACHED: CARRYING payload_gamma."
            )

    # -----------------------------------------------------------------------
    # CARRY PHASE
    # -----------------------------------------------------------------------

    def elapsed(self):

        return (
            self.get_clock().now() - self.phase_start
        ).nanoseconds * 1e-9

    def payload_estimate(self):

        estimates = [
            (
                self.pose[r][0] - SLOTS[r][0],
                self.pose[r][1] - SLOTS[r][1],
            )
            for r in ROBOTS
        ]

        return (
            sum(p[0] for p in estimates) / len(estimates),
            sum(p[1] for p in estimates) / len(estimates),
        )

    def carry_tick(self):

        end_time = self.carry[-1][0]

        t = min(
            self.elapsed() + LEAD,
            end_time,
        )

        payload_target = position_at(
            self.carry,
            t,
        )

        payload_now = self.payload_estimate()

        at_destination = (
            self.elapsed() >= end_time
            and math.dist(
                payload_now,
                DESTINATION,
            ) <= 0.04
        )

        if at_destination:

            self.phase = "detaching"
            self.stop_all()

            self.get_logger().info(
                "Payload reached destination. DETACHING."
            )

            return

        for robot in ROBOTS:

            sx, sy = SLOTS[robot]

            target = (
                payload_target[0] + sx,
                payload_target[1] + sy,
            )

            self.drive(
                robot,
                target,
                YAWS[robot],
                CARRY_MAX_SPEED,
                slow_final=False,
            )

    # -----------------------------------------------------------------------
    # DETACHING PHASE
    # -----------------------------------------------------------------------

    def detaching_tick(self):

        self.stop_all()

        for robot in ROBOTS:

            if not self.detached[robot]:

                self.request_service(
                    robot,
                    attach=False,
                )

        if all(self.detached.values()):

            self.phase = "backaway"

            self.get_logger().info(
                "ALL THREE RELEASED payload_gamma. BACKING AWAY."
            )

    # -----------------------------------------------------------------------
    # BACKAWAY PHASE
    # -----------------------------------------------------------------------

    def backaway_tick(self):

        complete = []

        for robot in ROBOTS:

            sx, sy = SLOTS[robot]

            length = math.hypot(sx, sy)

            target = (
                DESTINATION[0]
                + sx
                + 0.13 * sx / length,

                DESTINATION[1]
                + sy
                + 0.13 * sy / length,
            )

            distance, yaw_error = self.drive(
                robot,
                target,
                YAWS[robot],
                BACKAWAY_MAX_SPEED,
                slow_final=False,
            )

            complete.append(
                distance <= BACKAWAY_TOLERANCE
                and abs(yaw_error) <= YAW_TOLERANCE
            )

            if complete[-1]:
                self.send(robot)

        if all(complete):

            self.phase = "done"
            self.stop_all()

            self.get_logger().info(
                "TASK 3B COMPLETE."
            )

    # -----------------------------------------------------------------------
    # MAIN CONTROL LOOP
    # -----------------------------------------------------------------------

    def tick(self):

        if self.phase == "waiting":
            self.stop_all()
            return

        if len(self.pose) != len(ROBOTS):
            self.stop_all()
            return

        if self.phase == "approach":
            self.approach_tick()

        elif self.phase == "attaching":
            self.attaching_tick()

        elif self.phase == "carry":
            self.carry_tick()

        elif self.phase == "detaching":
            self.detaching_tick()

        elif self.phase == "backaway":
            self.backaway_tick()

        else:
            self.stop_all()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = TeamFollower()

    try:
        rclpy.spin(node)

    except (
        KeyboardInterrupt,
        ExternalShutdownException,
    ):
        pass

    finally:

        try:
            node.stop_all()

        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()