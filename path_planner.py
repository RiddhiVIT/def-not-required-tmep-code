#!/usr/bin/env python3
"""Task 3B planner: find payload_gamma and plan three approaches and one carry."""

import heapq
import math
import os
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

TEAM_ID = 3978
ROBOTS = ("r1", "r2", "r3")
STARTS = {"r1": (0.8969, 2.2616), "r2": (1.2192, 2.2616),
          "r3": (1.5415, 2.2616)}
DESTINATION = (1.1992, 1.2092)
PAYLOAD_PADS = ((0.4992, 0.3992), (1.9392, 0.3992),
                (0.3592, 1.2192), (2.0792, 1.2192))
SLOTS = {"r1": (0.0, -0.1827), "r2": (-0.1796, 0.0),
         "r3": (0.0, 0.1827)}
ARENA_SIZE = 2.4384
ARENA_LEFT, ARENA_TOP, ARENA_PIXELS = 304, 24, 671
GRID_CELLS = 244
CELL_SIZE = ARENA_SIZE / GRID_CELLS
CAMERA_URL = os.environ.get("HE_CAMERA_URL", "http://127.0.0.1:8080/stream")
SAND_LOW, SAND_HIGH = (10, 40, 170), (30, 120, 255)
PLAN_CELL = 0.025
ROBOT_RADIUS = 0.084
SAFETY_MARGIN = 0.066
PLAN_SPEED = 0.10
CARRY_SPEED = 0.14
DOCK_SPEED = 0.07
START_DELAY = 2.0
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def read_frame(url=CAMERA_URL):
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no camera frame; is task3a.launch.py running?")
    return frame


def crop_arena(frame):
    return frame[ARENA_TOP:ARENA_TOP + ARENA_PIXELS,
                 ARENA_LEFT:ARENA_LEFT + ARENA_PIXELS]


def pixel_to_world(u, v):
    return ((u - ARENA_LEFT) * ARENA_SIZE / ARENA_PIXELS,
            (v - ARENA_TOP) * ARENA_SIZE / ARENA_PIXELS)


def find_payload(frame):
    """Return payload_gamma centre in /odom from its bright violet top."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # The violet cap is antialiased and much of it sits around saturation
    # 140-170; a 170 cutoff fragments it under the 3x3 opening.
    mask = cv2.inRange(hsv, (125, 140, 100), (155, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 350 <= area <= 1800 and 20 <= w <= 60 and 20 <= h <= 60:
            candidates.append((area, centres[i]))
    if not candidates:
        raise RuntimeError("payload_gamma not found in camera frame")
    _, (u, v) = max(candidates, key=lambda item: item[0])
    x, y = pixel_to_world(float(u), float(v))
    correction = 0.93  # top is raised; project radially back toward camera axis
    x = DESTINATION[0] + correction * (x - DESTINATION[0])
    y = DESTINATION[1] + correction * (y - DESTINATION[1])
    # Residual centroid bias measured against the physics contact point.  The
    # violet bevel is asymmetric in the overhead render; these millimetres
    # matter because the solenoid contact band is only 6 mm wide.
    x -= 0.0020
    y += 0.0042
    return x, y


def frame_to_occupancy(frame):
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    obstacles = cv2.bitwise_not(cv2.inRange(hsv, SAND_LOW, SAND_HIGH))
    obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_OPEN,
                                 np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(obstacles, contours, -1, 255, cv2.FILLED)
    small = cv2.resize(obstacles, (GRID_CELLS, GRID_CELLS),
                       interpolation=cv2.INTER_AREA)
    return small > 0


def clear_circle(grid, xy, radius):
    rows, cols = np.indices(grid.shape)
    x = (cols + 0.5) * CELL_SIZE
    y = (rows + 0.5) * CELL_SIZE
    grid[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = False


def block_circle(grid, xy, radius):
    rows, cols = np.indices(grid.shape)
    x = (cols + 0.5) * CELL_SIZE
    y = (rows + 0.5) * CELL_SIZE
    grid[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = True


def remove_drivable(frame, occupied, robots, goals):
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    coloured = ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 50)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(coloured, 8)
    floor = np.zeros_like(coloured, dtype=np.uint8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 1500:
            floor[labels == i] = 1
    floor = cv2.resize(floor, (GRID_CELLS, GRID_CELLS),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    occupied[floor] = False

    # The coloured rings, black outlines and shadows of the four payload pads
    # and the centre hexagon are paint, not collision geometry.  Free their
    # full painted footprints, then put the three payloads that are not being
    # carried back into the map as compact physical obstacles.  Without this
    # step the inflated black outlines can join nearby pillars into a false
    # wall on some random layouts.
    for pad in PAYLOAD_PADS:
        clear_circle(occupied, pad, 0.24)
    clear_circle(occupied, DESTINATION, 0.41)
    for payload in PAYLOAD_PADS[:3]:
        block_circle(occupied, payload, 0.105)

    # Clear the complete camera silhouettes of the robots and start plates.
    # Inflation below will add the real chassis clearance back around them.
    for p in [*STARTS.values(), *robots]:
        clear_circle(occupied, p, 0.17)
    for p in goals:
        clear_circle(occupied, p, 0.15)
    return occupied


def planning_grid(occupied, left=ROBOT_RADIUS + SAFETY_MARGIN,
                  right=None, up=None, down=None):
    """Configuration-space grid for an asymmetric footprint around a point."""
    right = left if right is None else right
    up = left if up is None else up
    down = up if down is None else down
    n = int(round(ARENA_SIZE / PLAN_CELL))
    kl = int(math.ceil(left / CELL_SIZE)); kr = int(math.ceil(right / CELL_SIZE))
    ku = int(math.ceil(up / CELL_SIZE)); kd = int(math.ceil(down / CELL_SIZE))
    if abs(left - right) < 1e-9 and abs(left - up) < 1e-9 and abs(left - down) < 1e-9:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kl + kr + 1, ku + kd + 1))
    else:
        kernel = np.ones((ku + kd + 1, kl + kr + 1), np.uint8)
    inflated = cv2.dilate(occupied.astype(np.uint8), kernel,
                          anchor=(kl, ku)) > 0
    centre = (np.arange(n) + 0.5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    blocked = inflated[np.ix_(fine, fine)].T
    blocked[centre < left, :] = True
    blocked[centre > ARENA_SIZE - right, :] = True
    blocked[:, centre < up] = True
    blocked[:, centre > ARENA_SIZE - down] = True
    return blocked


def to_cell(p):
    n = int(round(ARENA_SIZE / PLAN_CELL))
    return (min(max(int(p[0] / PLAN_CELL), 0), n - 1),
            min(max(int(p[1] / PLAN_CELL), 0), n - 1))


def to_world(c):
    return ((c[0] + 0.5) * PLAN_CELL, (c[1] + 0.5) * PLAN_CELL)


def nearest_free(blocked, p, max_r=0.35):
    c = to_cell(p)
    if not blocked[c]:
        return c
    r = int(math.ceil(max_r / PLAN_CELL)); cells = []
    for i in range(max(0, c[0] - r), min(blocked.shape[0], c[0] + r + 1)):
        for j in range(max(0, c[1] - r), min(blocked.shape[1], c[1] + r + 1)):
            if not blocked[i, j]: cells.append((i, j))
    return min(cells, key=lambda q: math.dist(to_world(q), p)) if cells else None


def plan_route(blocked, start, goal, speed=PLAN_SPEED):
    source, target = nearest_free(blocked, start), nearest_free(blocked, goal)
    if source is None or target is None: return None
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]
    frontier = [(math.dist(source, target), 0.0, source)]
    parent, cost = {source: None}, {source: 0.0}
    while frontier:
        _, g, cur = heapq.heappop(frontier)
        if g > cost.get(cur, math.inf) + 1e-9: continue
        if cur == target: break
        for di, dj in moves:
            nxt = (cur[0] + di, cur[1] + dj)
            if not (0 <= nxt[0] < blocked.shape[0] and
                    0 <= nxt[1] < blocked.shape[1]) or blocked[nxt]: continue
            if di and dj and (blocked[cur[0] + di, cur[1]] or
                              blocked[cur[0], cur[1] + dj]): continue
            ng = g + math.hypot(di, dj)
            if ng < cost.get(nxt, math.inf):
                cost[nxt], parent[nxt] = ng, cur
                heapq.heappush(frontier, (ng + math.dist(nxt, target), ng, nxt))
    if target not in parent: return None
    cells, cur = [], target
    while cur is not None: cells.append(cur); cur = parent[cur]
    points = [to_world(c) for c in reversed(cells)]
    points[0], points[-1] = start, goal

    def clear(a, b):
        steps = max(2, int(math.ceil(math.dist(a, b) / (0.4 * PLAN_CELL))))
        return all(not blocked[to_cell((a[0] + (b[0] - a[0]) * k / steps,
                                               a[1] + (b[1] - a[1]) * k / steps))]
                   for k in range(steps + 1))
    short, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not clear(points[i], points[j]): j -= 1
        short.append(points[j]); i = j
    plan, t = [(0.0, *short[0])], 0.0
    for a, b in zip(short, short[1:]):
        t += math.dist(a, b) / speed; plan.append((t, *b))
    return plan


def carry_grid(occupied, payload):
    free = occupied.copy(); clear_circle(free, payload, 0.38)
    clear_circle(free, DESTINATION, 0.42)
    left = right = up = down = 0.27
    kl = int(math.ceil(left / CELL_SIZE)); kr = int(math.ceil(right / CELL_SIZE))
    ku = int(math.ceil(up / CELL_SIZE)); kd = int(math.ceil(down / CELL_SIZE))
    yy, xx = np.indices((ku + kd + 1, kl + kr + 1))
    dx = (xx - kl) * CELL_SIZE; dy = (yy - ku) * CELL_SIZE
    footprint = (abs(dx) <= 0.12) & (abs(dy) <= 0.12)
    for slot in SLOTS.values():
        footprint |= ((dx - slot[0]) ** 2 + (dy - slot[1]) ** 2 <= 0.08 ** 2)
    inflated = cv2.dilate(free.astype(np.uint8), footprint.astype(np.uint8),
                          anchor=(kl, ku)) > 0
    n = int(round(ARENA_SIZE / PLAN_CELL)); centre = (np.arange(n) + .5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    blocked = inflated[np.ix_(fine, fine)].T
    blocked[centre < left, :] = True; blocked[centre > ARENA_SIZE - right, :] = True
    blocked[:, centre < up] = True; blocked[:, centre > ARENA_SIZE - down] = True
    return blocked


def plan_carry(blocked, payload):
    return plan_route(blocked, payload, DESTINATION, CARRY_SPEED)


def append_straight(plan, goal, speed=DOCK_SPEED):
    """Finish a route with a slow radial docking motion."""
    if plan is None:
        return None
    _, x, y = plan[-1]
    return [*plan, (plan[-1][0] + math.dist((x, y), goal) / speed,
                    goal[0], goal[1])]


def to_path_msg(plan, start):
    msg = Path(); msg.header.frame_id = "odom"; msg.header.stamp = start.to_msg()
    for t, x, y in plan:
        pose = PoseStamped(); pose.header.frame_id = "odom"
        pose.header.stamp = (start + Duration(seconds=float(t))).to_msg()
        pose.pose.position.x = float(x); pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0; msg.poses.append(pose)
    return msg


def delay_plan(plan, delay):
    return [(t + delay, x, y) for t, x, y in plan]


def sample(plan, t):
    return position_at(plan, t)


def position_at(plan, t):
    if t <= plan[0][0]: return plan[0][1:]
    for a, b in zip(plan, plan[1:]):
        if t <= b[0]:
            u = (t - a[0]) / max(1e-6, b[0] - a[0])
            return (a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]))
    return plan[-1][1:]


def schedule(plans):
    """Add start waits until no pair of robot centres comes within 14 cm."""
    out = {}
    for robot in ROBOTS:
        raw = plans[robot]; delay = 0.0
        while True:
            candidate = delay_plan(raw, delay)
            end = max([candidate[-1][0], *[p[-1][0] for p in out.values()]])
            collision = any(
                math.dist(sample(candidate, t), sample(other, t)) < 0.20
                for other in out.values() for t in np.arange(0.0, end + 0.05, 0.05))
            if not collision: break
            delay += 0.25
            if delay > 30.0: raise RuntimeError("could not schedule collision-free approaches")
        out[robot] = candidate
    return out


class PathPlanner(Node):
    def __init__(self):
        super().__init__("task_3b_path_planner")
        self.robot_xy = {}; self.done = False
        for robot in ROBOTS:
            self.create_subscription(Odometry, f"/{robot}/odom",
                                     lambda msg, r=robot: self.on_odom(r, msg), 10)
        self.approach_pub = {r: self.create_publisher(Path, f"/{r}/plan", LATCHED)
                             for r in ROBOTS}
        self.carry_pub = self.create_publisher(Path, "/carry_plan", LATCHED)
        self.create_timer(0.25, self.tick)

    def on_odom(self, robot, msg):
        p = msg.pose.pose.position; self.robot_xy[robot] = (p.x, p.y)

    def tick(self):
        if self.done or len(self.robot_xy) != 3: return
        try:
            frame = read_frame(); payload = find_payload(frame)
            docks = {r: (payload[0] + SLOTS[r][0], payload[1] + SLOTS[r][1])
                     for r in ROBOTS}
            pre_docks = {}
            for r in ROBOTS:
                sx, sy = SLOTS[r]
                length = math.hypot(sx, sy)
                pre_docks[r] = (docks[r][0] + 0.12 * sx / length,
                                docks[r][1] + 0.12 * sy / length)
            occupied = remove_drivable(frame, frame_to_occupancy(frame),
                                       list(self.robot_xy.values()),
                                       [*pre_docks.values(), *docks.values(),
                                        DESTINATION])
            # The violet crate can merge into its coloured pad when the floor
            # is freed.  Re-block the crate while routing to the three outer
            # staging points; only each final radial docking leg enters this
            # exclusion zone.
            rows, cols = np.indices(occupied.shape)
            ox = (cols + .5) * CELL_SIZE; oy = (rows + .5) * CELL_SIZE
            occupied[(ox - payload[0]) ** 2 + (oy - payload[1]) ** 2
                     <= 0.15 ** 2] = True
            for pre_dock in pre_docks.values():
                clear_circle(occupied, pre_dock, 0.05)
            grid = planning_grid(occupied)
            approach = {
                r: append_straight(
                    plan_route(grid, self.robot_xy[r], pre_docks[r]), docks[r])
                for r in ROBOTS
            }
            missing = [r for r, plan in approach.items() if plan is None]
            if missing:
                raise RuntimeError("no approach route for: " + ", ".join(missing))
            # The follower docks one robot at a time, so geometric paths are
            # sufficient and do not need a simultaneous time schedule.
            carry = plan_carry(carry_grid(occupied, payload), payload)
            if carry is None: raise RuntimeError("no route for payload carry")
            start = self.get_clock().now() + Duration(seconds=START_DELAY)
            for r in ROBOTS: self.approach_pub[r].publish(to_path_msg(approach[r], start))
            self.carry_pub.publish(to_path_msg(carry, start))
            self.done = True
            self.get_logger().info("payload_gamma at (%.3f, %.3f)" % payload)
            for r in ROBOTS: self.get_logger().info(f"published /{r}/plan")
            self.get_logger().info("published /carry_plan")
        except Exception as exc:
            self.get_logger().error(str(exc))


def main():
    rclpy.init(); node = PathPlanner()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally: node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__": main()