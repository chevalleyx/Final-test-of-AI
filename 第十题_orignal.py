import math
from dataclasses import dataclass
from typing import List, Tuple

INF = 1e18

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: int
    ready: float
    due: float


def load_instance(path: str):
    """
    读取附件中的 CVRPTW 实例。
    数据格式:
    VEHICLE
    NUMBER CAPACITY
      10 150
    CUSTOMER
    CUST NO. XCOORD. YCOORD. DEMAND READY TIME DUE DATE
    ...
    """
    tokens: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tokens.extend(line.strip().split())

    i = 0
    # 找到 VEHICLE
    while i < len(tokens) and tokens[i].upper() != "VEHICLE":
        i += 1
    if i == len(tokens):
        raise ValueError("Input missing 'VEHICLE'")
    i += 1

    # 找到 NUMBER
    while i < len(tokens) and tokens[i].upper() != "NUMBER":
        i += 1
    if i == len(tokens):
        raise ValueError("Input missing 'NUMBER'")
    i += 1

    # 找到 CAPACITY
    while i < len(tokens) and tokens[i].upper() != "CAPACITY":
        i += 1
    if i == len(tokens):
        raise ValueError("Input missing 'CAPACITY'")
    i += 1

    vehicle_number = int(tokens[i]); i += 1
    capacity = int(tokens[i]); i += 1

    # 找到 CUSTOMER
    while i < len(tokens) and tokens[i].upper() != "CUSTOMER":
        i += 1
    if i == len(tokens):
        raise ValueError("Input missing 'CUSTOMER'")
    i += 1

    # 跳到第一行客户数据（第一个客户编号）
    while i < len(tokens) and not tokens[i].isdigit():
        i += 1

    customers: List[Customer] = []
    while i + 5 < len(tokens):
        cid = int(tokens[i]); x = float(tokens[i + 1]); y = float(tokens[i + 2])
        demand = int(tokens[i + 3]); ready = float(tokens[i + 4]); due = float(tokens[i + 5])
        customers.append(Customer(cid, x, y, demand, ready, due))
        i += 6

    customers.sort(key=lambda c: c.id)
    return vehicle_number, capacity, customers


def compute_distance_matrix(customers: List[Customer]) -> List[List[int]]:
    """欧式距离向下取整。"""
    n = len(customers)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = customers[i].x - customers[j].x
            dy = customers[i].y - customers[j].y
            d = int(math.hypot(dx, dy))
            dist[i][j] = dist[j][i] = d
    return dist


def simulate_route(route: List[int],
                   customers: List[Customer],
                   dist: List[List[int]],
                   capacity: int) -> Tuple[bool, float, float]:
    """
    检查一条路径的可行性，并返回 (是否可行, 路径距离, 完成时间)。
    route 形如 [0, i1, i2, ..., 0]。
    """
    if route[0] != 0 or route[-1] != 0:
        raise ValueError("Route must start and end at depot 0")

    time = 0.0
    load = 0
    total_dist = 0.0

    for idx in range(len(route) - 1):
        i = route[idx]
        j = route[idx + 1]
        total_dist += dist[i][j]
        time += dist[i][j]
        cust_j = customers[j]

        # 等待窗口开始
        if time < cust_j.ready:
            time = cust_j.ready

        # 时间窗约束
        if time > cust_j.due + 1e-6:
            return False, INF, time

        # 容量约束
        if j != 0:
            load += cust_j.demand
            if load > capacity:
                return False, INF, time

    return True, total_dist, time


def total_distance(routes: List[List[int]],
                   customers: List[Customer],
                   dist: List[List[int]],
                   capacity: int) -> float:
    total = 0.0
    feasible = True
    for r in routes:
        f, d, _ = simulate_route(r, customers, dist, capacity)
        if not f:
            feasible = False
        total += d
    return total if feasible else INF


def initial_solution(customers: List[Customer],
                     dist: List[List[int]],
                     vehicle_number: int,
                     capacity: int) -> List[List[int]]:
    """
    构造阶段：按 DUE TIME 升序依次插入客户，
    每次在所有已有路径/新路径中选择“最省距离”的插入位置（插入启发式）。
    """
    n = len(customers)
    unserved = set(range(1, n))  # 客户 1..n-1
    order = sorted(unserved, key=lambda i: (customers[i].due, customers[i].ready))

    routes: List[List[int]] = []

    for cid in order:
        if cid not in unserved:
            continue

        best_delta = None
        best_route_idx = None
        best_new_route = None

        # 尝试插入到已有路径
        for r_idx, route in enumerate(routes):
            base_dist = simulate_route(route, customers, dist, capacity)[1]
            for pos in range(1, len(route)):  # 在 route[pos-1] 和 route[pos] 之间插入
                new_route = route[:pos] + [cid] + route[pos:]
                feasible, new_dist, _ = simulate_route(new_route, customers, dist, capacity)
                if not feasible:
                    continue
                delta = new_dist - base_dist
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_route_idx = r_idx
                    best_new_route = new_route

        # 尝试开新车
        if len(routes) < vehicle_number:
            new_route = [0, cid, 0]
            feasible, new_dist, _ = simulate_route(new_route, customers, dist, capacity)
            if feasible:
                delta = new_dist
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_route_idx = None
                    best_new_route = new_route

        # 应用最优插入
        if best_new_route is None:
            # 理论上不太会发生，如发生则强制单独一条路径
            new_route = [0, cid, 0]
            routes.append(new_route)
        else:
            if best_route_idx is None:
                routes.append(best_new_route)
            else:
                routes[best_route_idx] = best_new_route

        unserved.remove(cid)

    return routes


def two_opt(route: List[int],
            customers: List[Customer],
            dist: List[List[int]],
            capacity: int) -> List[int]:
    """
    单条路径的 2-opt 邻域搜索（只接受改进解）。
    """
    best_route = route
    best_dist = simulate_route(route, customers, dist, capacity)[1]
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                feasible, new_dist, _ = simulate_route(new_route, customers, dist, capacity)
                if feasible and new_dist + 1e-6 < best_dist:
                    best_route, best_dist = new_route, new_dist
                    improved = True
                    break
            if improved:
                break

    return best_route


def intra_route_2opt_all(routes: List[List[int]],
                         customers: List[Customer],
                         dist: List[List[int]],
                         capacity: int) -> List[List[int]]:
    return [two_opt(r, customers, dist, capacity) for r in routes]


def relocate_move(routes: List[List[int]],
                  customers: List[Customer],
                  dist: List[List[int]],
                  capacity: int):
    """
    在所有路径之间做 relocate（把一个顾客从某条路径移动到另一条路径），
    找到单次最优改进操作。
    """
    base_total = total_distance(routes, customers, dist, capacity)
    best_delta = 0.0
    best_routes = None

    n_routes = len(routes)
    for from_r in range(n_routes):
        for to_r in range(n_routes):
            for i in range(1, len(routes[from_r]) - 1):
                cid = routes[from_r][i]
                if cid == 0:
                    continue
                for j in range(1, len(routes[to_r])):
                    if from_r == to_r and (j == i or j == i + 1):
                        continue

                    new_routes = [list(r) for r in routes]
                    # 从 from_r 中删除
                    new_routes[from_r].pop(i)
                    if len(new_routes[from_r]) < 2:
                        continue

                    # 插入到 to_r 中
                    insert_pos = j
                    if from_r == to_r and j > i:
                        insert_pos -= 1
                    new_routes[to_r].insert(insert_pos, cid)

                    # 检查所有路径的可行性和新距离
                    ok = True
                    new_total = 0.0
                    for r in new_routes:
                        f, d, _ = simulate_route(r, customers, dist, capacity)
                        if not f:
                            ok = False
                            break
                        new_total += d
                    if not ok:
                        continue

                    delta = new_total - base_total
                    if delta < best_delta - 1e-6:
                        best_delta = delta
                        best_routes = new_routes

    if best_routes is None:
        return routes, False
    else:
        return best_routes, True


def local_search(routes: List[List[int]],
                 customers: List[Customer],
                 dist: List[List[int]],
                 capacity: int,
                 max_iter: int = 50) -> List[List[int]]:
    """
    简单局部搜索：先对每条路做 2-opt，再反复执行最优 relocate 直到没有改进。
    """
    routes = intra_route_2opt_all(routes, customers, dist, capacity)
    improved = True
    it = 0
    while improved and it < max_iter:
        it += 1
        routes, improved = relocate_move(routes, customers, dist, capacity)
        if improved:
            routes = intra_route_2opt_all(routes, customers, dist, capacity)
    return routes


def compute_schedule(route: List[int],
                     customers: List[Customer],
                     dist: List[List[int]],
                     capacity: int):
    """
    计算一条路径上每个客户的到达时间和装载量。
    返回列表 [(cust_id, load, time), ...]
    """
    time = 0.0
    load = 0
    schedule = []
    for idx, cid in enumerate(route):
        if idx > 0:
            prev = route[idx - 1]
            time += dist[prev][cid]
        cust = customers[cid]
        if time < cust.ready:
            time = cust.ready
        if cid != 0:
            load += cust.demand
        schedule.append((cid, load, time))
    return schedule


def solve(path: str):
    vehicle_number, capacity, customers = load_instance(path)
    dist = compute_distance_matrix(customers)

    # 构造初始解
    routes = initial_solution(customers, dist, vehicle_number, capacity)
    # 局部搜索改进
    routes = local_search(routes, customers, dist, capacity, max_iter=50)

    total = total_distance(routes, customers, dist, capacity)

    # 输出结果，格式参考题目示例
    print(f"Total distance: {total:.0f}")
    for k, r in enumerate(routes):
        print(f"Route for vehicle {k}:")
        schedule = compute_schedule(r, customers, dist, capacity)
        # schedule[0] 是 depot 0
        print("  0", end="")
        for cid, load, t in schedule[1:]:
            print(f" -> {cid} Load({load}) Time({int(t)})", end="")
        print()
        feasible, d, _ = simulate_route(r, customers, dist, capacity)
        print(f"Distance of the route: {int(d)}  Feasible: {feasible}")
        print()


if __name__ == "__main__":
    # 默认使用 data.txt，可以在命令行传入其它路径
    import sys
    instance_path = sys.argv[1] if len(sys.argv) > 1 else "data.txt"
    solve(instance_path)
