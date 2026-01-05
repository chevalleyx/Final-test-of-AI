import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
        for j in range(n):
            dx = customers[i].x - customers[j].x
            dy = customers[i].y - customers[j].y
            dist[i][j] = int(math.sqrt(dx*dx + dy*dy))  # floor
    return dist


def simulate_route(route: List[int],
                   customers: List[Customer],
                   dist: List[List[int]],
                   capacity: int) -> Tuple[bool, float, float]:
    """
    判断路径是否满足：
    - 容量约束
    - 时间窗约束（允许提前到达并等待）
    返回: (feasible, distance, finish_time)
    """
    time = 0.0
    load = 0
    total_dist = 0.0

    for idx in range(len(route) - 1):
        i = route[idx]
        j = route[idx + 1]
        total_dist += dist[i][j]
        time += dist[i][j]

        cust = customers[j]
        # 等待到 ready
        if time < cust.ready:
            time = cust.ready
        # 超过 due 不可行
        if time > cust.due:
            return False, INF, INF
        # 更新载重
        load += cust.demand
        if load > capacity:
            return False, INF, INF

    return True, total_dist, time


def route_distance(route: List[int],
                   customers: List[Customer],
                   dist: List[List[int]],
                   capacity: int) -> Optional[float]:
    feasible, d, _ = simulate_route(route, customers, dist, capacity)
    return d if feasible else None



def normalize_route(route: List[int]) -> List[int]:
    """
    规范化路线：确保只在首尾出现 depot=0，去掉内部多余的 0（例如 [0,0,5,...,0]）。
    """
    if not route:
        return [0, 0]
    core = [cid for cid in route[1:-1] if cid != 0]
    return [0] + core + [0]


def normalize_routes(routes: List[List[int]]) -> List[List[int]]:
    out = []
    for r in routes:
        nr = normalize_route(r)
        if len(nr) > 2:
            out.append(nr)
    return out
def total_distance(routes: List[List[int]],
                   customers: List[Customer],
                   dist: List[List[int]],
                   capacity: int) -> float:
    """
    计算总距离，并强制检查“覆盖约束”：
    - 客户 1..N-1 必须各出现一次（不允许缺失/重复）
    """
    n = len(customers)
    seen = []
    for r in routes:
        for cid in r[1:-1]:
            if cid != 0:
                seen.append(cid)
    if len(seen) != n - 1 or len(set(seen)) != n - 1:
        return INF

    total = 0.0
    for r in routes:
        d = route_distance(r, customers, dist, capacity)
        if d is None:
            return INF
        total += d
    return total
def best_insertion_position(routes: List[List[int]],
                            cid: int,
                            customers: List[Customer],
                            dist: List[List[int]],
                            capacity: int,
                            vehicle_number: int):
    """
    在所有已有路线（以及可用的新车）中，寻找插入 cid 的最小增量位置。
    返回 (delta, route_idx or None(表示新车), new_route)
    """
    best = None  # (delta, idx, new_route)

    for r_idx, route in enumerate(routes):
        base_d = route_distance(route, customers, dist, capacity)
        if base_d is None:
            continue
        for pos in range(1, len(route)):  # 插入在 pos-1 与 pos 之间
            new_route = route[:pos] + [cid] + route[pos:]
            nd = route_distance(new_route, customers, dist, capacity)
            if nd is None:
                continue
            delta = nd - base_d
            if best is None or delta < best[0]:
                best = (delta, r_idx, new_route)

    # 允许开新车
    if len(routes) < vehicle_number:
        new_route = [0, cid, 0]
        nd = route_distance(new_route, customers, dist, capacity)
        if nd is not None:
            delta = nd
            if best is None or delta < best[0]:
                best = (delta, None, new_route)

    return best


def initial_solution_random(customers: List[Customer],
                            dist: List[List[int]],
                            vehicle_number: int,
                            capacity: int,
                            seed: int = 0,
                            noise: float = 50.0) -> Optional[List[List[int]]]:
    """
    构造阶段（随机多次重启用）：
    仍然偏向 due time 小的客户先插入，但加入 noise 打破确定性，避免陷入差的初始解。
    """
    random.seed(seed)
    n = len(customers)
    ids = list(range(1, n))  # 客户 1..n-1
    if noise > 0:
        ids.sort(key=lambda i: customers[i].due + random.random() * noise)
    else:
        ids.sort(key=lambda i: (customers[i].due, customers[i].ready))

    routes: List[List[int]] = []
    for cid in ids:
        best = best_insertion_position(routes, cid, customers, dist, capacity, vehicle_number)
        if best is None:
            # 仍无法插入：若车数不够就开新车，否则失败
            if len(routes) < vehicle_number:
                routes.append([0, cid, 0])
            else:
                return None
        else:
            _, idx, new_route = best
            if idx is None:
                routes.append(new_route)
            else:
                routes[idx] = new_route

    # 清理空路线（理论上不会出现）
    routes = [r for r in routes if len(r) > 2]
    return routes


def two_opt(route: List[int],
            customers: List[Customer],
            dist: List[List[int]],
            capacity: int) -> List[int]:
    """单条路径的 2-opt 邻域搜索（只接受改进解）。"""
    best_route = route
    best_dist = route_distance(route, customers, dist, capacity) or INF
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for k in range(i + 1, len(best_route) - 1):
                new_route = best_route[:i] + best_route[i:k+1][::-1] + best_route[k+1:]
                nd = route_distance(new_route, customers, dist, capacity)
                if nd is None:
                    continue
                if nd + 1e-9 < best_dist:
                    best_dist = nd
                    best_route = new_route
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
                  capacity: int) -> Tuple[List[List[int]], bool]:
    """
    最优 relocate：把一个客户从一条路径移动到另一条（或同一条）某位置。
    这里用“增量评价”：只重算受影响的1~2条路径距离，比全量 total_distance 更快。
    """
    # 预计算每条路径距离
    route_ds = []
    for r in routes:
        d = route_distance(r, customers, dist, capacity)
        if d is None:
            return routes, False
        route_ds.append(d)

    base_total = sum(route_ds)
    best_delta = 0.0
    best_move = None  # (from_r, to_r, i, j, new_ra, new_rb, remove_from)

    n_routes = len(routes)
    for from_r in range(n_routes):
        ra = routes[from_r]
        da = route_ds[from_r]
        for i in range(1, len(ra) - 1):
            cid = ra[i]

            for to_r in range(n_routes):
                rb = routes[to_r]
                db = route_ds[to_r]

                for j in range(1, len(rb)):  # 插入到 j 之前
                    if from_r == to_r and (j == i or j == i + 1):
                        continue

                    if from_r == to_r:
                        tmp = ra[:i] + ra[i+1:]
                        new_r = tmp[:j] + [cid] + tmp[j:]
                        nd = route_distance(new_r, customers, dist, capacity)
                        if nd is None:
                            continue
                        delta = nd - da
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_move = (from_r, to_r, i, j, new_r, None, False)
                    else:
                        new_ra = ra[:i] + ra[i+1:]
                        new_rb = rb[:j] + [cid] + rb[j:]

                        # 如果 from 路线被移空，就把它删掉（不再计入）
                        remove_from = (len(new_ra) <= 2)

                        nda = 0.0
                        if not remove_from:
                            nda = route_distance(new_ra, customers, dist, capacity)
                            if nda is None:
                                continue
                        ndb = route_distance(new_rb, customers, dist, capacity)
                        if ndb is None:
                            continue

                        if remove_from:
                            delta = ndb - (da + db)
                        else:
                            delta = (nda + ndb) - (da + db)

                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_move = (from_r, to_r, i, j, new_ra, new_rb, remove_from)

    if best_move is None:
        return routes, False

    from_r, to_r, i, j, new_ra, new_rb, remove_from = best_move
    new_routes = [r[:] for r in routes]
    if from_r == to_r:
        new_routes[from_r] = new_ra  # type: ignore
    else:
        if remove_from:
            new_routes[from_r] = [0, 0]  # 占位，后面过滤掉
        else:
            new_routes[from_r] = new_ra  # type: ignore
        new_routes[to_r] = new_rb  # type: ignore

    new_routes = [r for r in new_routes if len(r) > 2]
    new_routes = normalize_routes(new_routes)
    return new_routes, True
def swap_move(routes: List[List[int]],
              customers: List[Customer],
              dist: List[List[int]],
              capacity: int) -> Tuple[List[List[int]], bool]:
    """
    交换两条路径中的两个客户（swap）。
    """
    base_total = total_distance(routes, customers, dist, capacity)
    best_delta = 0.0
    best_routes = None

    n_routes = len(routes)
    route_ds = [route_distance(r, customers, dist, capacity) for r in routes]

    for a in range(n_routes):
        for b in range(a + 1, n_routes):
            ra = routes[a]; rb = routes[b]
            da = route_ds[a] if route_ds[a] is not None else INF
            db = route_ds[b] if route_ds[b] is not None else INF

            for i in range(1, len(ra) - 1):
                ca = ra[i]
                for j in range(1, len(rb) - 1):
                    cb = rb[j]
                    new_ra = ra[:i] + [cb] + ra[i+1:]
                    new_rb = rb[:j] + [ca] + rb[j+1:]

                    nda = route_distance(new_ra, customers, dist, capacity)
                    if nda is None:
                        continue
                    ndb = route_distance(new_rb, customers, dist, capacity)
                    if ndb is None:
                        continue

                    new_total = base_total - da - db + nda + ndb
                    delta = new_total - base_total
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_routes = [r[:] for r in routes]
                        best_routes[a] = new_ra
                        best_routes[b] = new_rb

    if best_routes is None:
        return routes, False
    return normalize_routes(best_routes), True


def two_opt_star_move(routes: List[List[int]],
                      customers: List[Customer],
                      dist: List[List[int]],
                      capacity: int) -> Tuple[List[List[int]], bool]:
    """
    2-opt*（跨路径）：在两条路径上分别选切点，交换后半段（tail exchange）。
    对 VRP/CVRPTW 通常很有效。
    """
    base_total = total_distance(routes, customers, dist, capacity)
    best_delta = 0.0
    best_routes = None

    n_routes = len(routes)
    route_ds = [route_distance(r, customers, dist, capacity) for r in routes]

    for a in range(n_routes):
        for b in range(a + 1, n_routes):
            ra = routes[a]; rb = routes[b]
            da = route_ds[a] if route_ds[a] is not None else INF
            db = route_ds[b] if route_ds[b] is not None else INF

            for i in range(1, len(ra) - 1):
                for j in range(1, len(rb) - 1):
                    # 交换 tail（去掉末尾 0 再接回 0）
                    new_ra = ra[:i] + rb[j:-1] + [0]
                    new_rb = rb[:j] + ra[i:-1] + [0]

                    # 可能出现 [0,0]，直接丢弃
                    cand_routes = [r[:] for r in routes]
                    cand_routes[a] = new_ra
                    cand_routes[b] = new_rb
                    cand_routes = [r for r in cand_routes if len(r) > 2]

                    new_total = total_distance(cand_routes, customers, dist, capacity)
                    if new_total >= INF:
                        continue
                    delta = new_total - base_total
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_routes = cand_routes

    if best_routes is None:
        return routes, False
    return normalize_routes(best_routes), True


def vns_local_search(routes: List[List[int]],
                     customers: List[Customer],
                     dist: List[List[int]],
                     capacity: int,
                     max_iter: int = 120) -> List[List[int]]:
    """
    变量邻域搜索（VNS 风格）：
    intra 2-opt → relocate → swap → 2-opt* 循环，直到没有改进或迭代上限。
    """
    routes = normalize_routes(intra_route_2opt_all(routes, customers, dist, capacity))
    it = 0
    improved = True
    while improved and it < max_iter:
        it += 1
        improved = False

        routes, imp = relocate_move(routes, customers, dist, capacity)
        if imp:
            routes = normalize_routes(intra_route_2opt_all(routes, customers, dist, capacity))
            improved = True
            continue

        routes, imp = swap_move(routes, customers, dist, capacity)
        if imp:
            routes = normalize_routes(intra_route_2opt_all(routes, customers, dist, capacity))
            improved = True
            continue

        routes, imp = two_opt_star_move(routes, customers, dist, capacity)
        if imp:
            routes = normalize_routes(intra_route_2opt_all(routes, customers, dist, capacity))
            improved = True
            continue

    return routes


def ruin_and_recreate(routes: List[List[int]],
                      customers: List[Customer],
                      dist: List[List[int]],
                      capacity: int,
                      vehicle_number: int,
                      remove_k: int) -> Optional[List[List[int]]]:
    """
    简单 LNS（Ruin & Recreate）：
    - 随机移除 remove_k 个客户
    - 用“最小增量插入”重新插回去
    """
    all_customers = [cid for r in routes for cid in r[1:-1]]
    if not all_customers:
        return None
    remove_k = max(1, min(remove_k, len(all_customers) - 1))
    removed = set(random.sample(all_customers, remove_k))

    new_routes: List[List[int]] = []
    for r in routes:
        nr = [0] + [cid for cid in r[1:-1] if cid not in removed] + [0]
        if len(nr) > 2:
            new_routes.append(nr)

    # 重新插入：按 due 排序（可改为随机/混合）
    rem_list = list(removed)
    rem_list.sort(key=lambda i: customers[i].due)

    for cid in rem_list:
        best = best_insertion_position(new_routes, cid, customers, dist, capacity, vehicle_number)
        if best is None:
            return None
        _, idx, new_route = best
        if idx is None:
            new_routes.append(new_route)
        else:
            new_routes[idx] = new_route

    return normalize_routes(new_routes)


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
        load += cust.demand
        schedule.append((cid, load, time))
    return schedule


def solve(path: str,
          restarts: int = 10,
          lns_iters: int = 60,
          noise: float = 50.0,
          seed_base: int = 0):
    vehicle_number, capacity, customers = load_instance(path)
    dist = compute_distance_matrix(customers)

    best_routes = None
    best_cost = INF

    for s in range(seed_base, seed_base + restarts):
        routes = initial_solution_random(customers, dist, vehicle_number, capacity, seed=s, noise=noise)
        if routes is None:
            continue

        routes = vns_local_search(routes, customers, dist, capacity, max_iter=120)
        cur_cost = total_distance(routes, customers, dist, capacity)

        # LNS 迭代：只接受改进（你也可以加 SA/TABU）
        for _ in range(lns_iters):
            k = random.randint(3, 8)
            cand = ruin_and_recreate(routes, customers, dist, capacity, vehicle_number, remove_k=k)
            if cand is None:
                continue
            cand = vns_local_search(cand, customers, dist, capacity, max_iter=60)
            cand_cost = total_distance(cand, customers, dist, capacity)
            if cand_cost + 1e-9 < cur_cost:
                routes = cand
                cur_cost = cand_cost

        if cur_cost + 1e-9 < best_cost:
            best_cost = cur_cost
            best_routes = routes

    if best_routes is None:
        raise RuntimeError("Failed to construct a feasible solution.")

    best_routes = normalize_routes(best_routes)

    # 输出结果
    print(f"Total distance: {best_cost:.0f}")
    for k, r in enumerate(best_routes):
        print(f"Route for vehicle {k}:")
        schedule = compute_schedule(r, customers, dist, capacity)
        print("  0", end="")
        for cid, load, t in schedule[1:]:
            print(f" -> {cid} Load({load}) Time({int(t)})", end="")
        print()
        feasible, d, _ = simulate_route(r, customers, dist, capacity)
        print(f"Distance of the route: {int(d)}  Feasible: {feasible}")
        print()


if __name__ == "__main__":
    import sys
    instance_path = sys.argv[1] if len(sys.argv) > 1 else "data.txt"
    # 可选参数：restarts / lns_iters / noise
    # 例如：python 第十题_improved.py data.txt 12 120 80
    restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    lns_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    noise = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0
    solve(instance_path, restarts=restarts, lns_iters=lns_iters, noise=noise)
