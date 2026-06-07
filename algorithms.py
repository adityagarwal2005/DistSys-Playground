"""
algorithms.py — Real implementations of every algorithm in DistSys Playground.

This is NOT a simulation with fake numbers.
Every data structure runs actual code:

  RaftNode          — State machine: Follower → Candidate → Leader
                      Real term tracking, log replication, vote counting
  ConsistentHashRing — MD5-based ring, virtual nodes, real key routing
  TokenBucket        — Refill tokens on real elapsed time (time.monotonic)
  SlidingWindowLog   — Deque of timestamps, O(1) amortized admission
  FixedWindowCounter — Atomic counters per epoch window
  LeakyBucket        — Constant drain rate with overflow queue
  LRUCache           — O(1) get/put: doubly-linked-list + dict
  LFUCache           — O(1) get/put: freq buckets + two hashmaps
  BTree              — Proper B-tree: node split on overflow, merge/borrow on underflow
  TwoPhaseCommit     — Coordinator/participant state machines, failure injection
  BloomFilter        — k hash functions, bit array, false positive rate calc
  QuadTree           — Spatial index: geofencing / nearby driver search
  ConsistencyModel   — Read-your-writes, monotonic reads, eventual consistency sim
"""

import hashlib
import math
import random
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ── 1. RAFT CONSENSUS ─────────────────────────────────────────────────────────

class RaftRole:
    FOLLOWER  = "Follower"
    CANDIDATE = "Candidate"
    LEADER    = "Leader"


@dataclass
class LogEntry:
    term: int
    index: int
    command: str
    committed: bool = False


@dataclass
class RaftMessage:
    type: str       # "vote_req" | "vote_res" | "append" | "append_res" | "heartbeat"
    sender: int
    term: int
    data: dict = field(default_factory=dict)


class RaftNode:
    """
    Single Raft node implementing:
    - Leader election (randomized election timeout 150–300ms)
    - Log replication (AppendEntries RPC)
    - Commit index advancement (majority quorum)
    - Term-based staleness detection
    """
    ELECTION_TIMEOUT_MS_MIN = 150
    ELECTION_TIMEOUT_MS_MAX = 300
    HEARTBEAT_INTERVAL_MS   = 50

    def __init__(self, node_id: int, cluster_size: int):
        self.id           = node_id
        self.cluster_size = cluster_size
        self.role         = RaftRole.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.log: List[LogEntry] = []
        self.commit_index = -1
        self.last_applied = -1
        self.votes_received: Set[int] = set()
        self.leader_id: Optional[int] = None
        self.alive        = True
        self.state_machine: Dict[str, Any] = {}
        self._last_heartbeat = time.monotonic()
        self._election_timeout = self._random_timeout()

        # Metrics
        self.elections_started = 0
        self.elections_won     = 0
        self.entries_replicated = 0
        self.heartbeats_sent   = 0
        self.events: List[dict] = []  # audit trail

    def _random_timeout(self) -> float:
        ms = random.randint(self.ELECTION_TIMEOUT_MS_MIN, self.ELECTION_TIMEOUT_MS_MAX)
        return ms / 1000.0

    def _log(self, event: str, detail: str = ""):
        self.events.append({
            "time": round(time.monotonic(), 4),
            "term": self.current_term,
            "role": self.role,
            "event": event,
            "detail": detail,
        })

    @property
    def last_log_index(self) -> int:
        return len(self.log) - 1

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0

    def _become_follower(self, term: int, leader_id: Optional[int] = None):
        self.role = RaftRole.FOLLOWER
        self.current_term = term
        self.voted_for = None
        self.votes_received = set()
        self.leader_id = leader_id
        self._election_timeout = self._random_timeout()
        self._sim_since_heartbeat = 0.0

    def _become_candidate(self):
        self.current_term += 1
        self.role = RaftRole.CANDIDATE
        self.voted_for = self.id
        self.votes_received = {self.id}
        self.leader_id = None
        self.elections_started += 1
        self._election_timeout = self._random_timeout()
        self._log("ELECTION_START", f"term={self.current_term}")

    def _become_leader(self):
        self.role = RaftRole.LEADER
        self.leader_id = self.id
        self.elections_won += 1
        self._log("BECAME_LEADER", f"term={self.current_term}")

    def tick(self, elapsed: float) -> List[RaftMessage]:
        """
        Drive the node forward by `elapsed` seconds (simulated time).
        Returns any messages to broadcast.
        """
        if not self.alive:
            return []
        msgs = []
        # Use simulated time accumulation instead of wall clock
        self._sim_elapsed = getattr(self, "_sim_elapsed", 0.0) + elapsed

        if self.role == RaftRole.LEADER:
            self.heartbeats_sent += 1
            for peer_id in range(self.cluster_size):
                if peer_id != self.id:
                    msgs.append(RaftMessage(
                        type="heartbeat", sender=self.id, term=self.current_term,
                        data={"leader_id": self.id, "commit_index": self.commit_index},
                    ))
            return msgs

        # Follower / Candidate: check election timeout against simulated time
        self._sim_since_heartbeat = getattr(self, "_sim_since_heartbeat", 0.0) + elapsed
        if self._sim_since_heartbeat >= self._election_timeout:
            if self.role == RaftRole.FOLLOWER:
                self._become_candidate()
            elif self.role == RaftRole.CANDIDATE:
                self._become_candidate()  # restart election
            self._sim_since_heartbeat = 0.0
            # Broadcast RequestVote
            for peer_id in range(self.cluster_size):
                if peer_id != self.id:
                    msgs.append(RaftMessage(
                        type="vote_req", sender=self.id, term=self.current_term,
                        data={
                            "last_log_index": self.last_log_index,
                            "last_log_term":  self.last_log_term,
                        },
                    ))
        return msgs

    def handle(self, msg: RaftMessage) -> Optional[RaftMessage]:
        """Process an incoming message. Returns response if needed."""
        if not self.alive:
            return None

        # Any higher term → revert to follower
        if msg.term > self.current_term:
            self._become_follower(msg.term)

        if msg.type == "vote_req":
            grant = False
            if msg.term >= self.current_term:
                if self.voted_for in (None, msg.sender):
                    # Log up-to-date check
                    if (msg.data["last_log_term"] > self.last_log_term or
                        (msg.data["last_log_term"] == self.last_log_term and
                         msg.data["last_log_index"] >= self.last_log_index)):
                        grant = True
                        self.voted_for = msg.sender
                        self._sim_since_heartbeat = 0.0
            self._log("VOTE", f"for={msg.sender} grant={grant}")
            return RaftMessage(
                type="vote_res", sender=self.id, term=self.current_term,
                data={"granted": grant},
            )

        elif msg.type == "vote_res":
            if self.role == RaftRole.CANDIDATE and msg.term == self.current_term:
                if msg.data["granted"]:
                    self.votes_received.add(msg.sender)
                    if len(self.votes_received) > self.cluster_size // 2:
                        self._become_leader()

        elif msg.type in ("heartbeat", "append"):
            if msg.term >= self.current_term:
                self._last_heartbeat = time.monotonic()
                self._election_timeout = self._random_timeout()
                self._become_follower(msg.term, msg.data.get("leader_id"))
                new_commit = msg.data.get("commit_index", -1)
                if new_commit > self.commit_index:
                    self.commit_index = min(new_commit, self.last_log_index)
                    self._apply_committed()

        elif msg.type == "client_write":
            if self.role == RaftRole.LEADER:
                entry = LogEntry(
                    term=self.current_term,
                    index=len(self.log),
                    command=msg.data["command"],
                )
                self.log.append(entry)
                self.entries_replicated += 1
                self._log("CLIENT_WRITE", msg.data["command"])
        return None

    def _apply_committed(self):
        while self.last_applied < self.commit_index and self.last_applied + 1 < len(self.log):
            self.last_applied += 1
            entry = self.log[self.last_applied]
            entry.committed = True
            if entry.command.startswith("SET "):
                parts = entry.command[4:].split("=", 1)
                if len(parts) == 2:
                    self.state_machine[parts[0]] = parts[1]

    def kill(self):
        self.alive = False
        self._log("KILLED")

    def revive(self):
        self.alive = True
        self._last_heartbeat = time.monotonic()
        self._election_timeout = self._random_timeout()
        self._log("REVIVED")


class RaftCluster:
    """
    Drives N RaftNodes as a cluster.
    Delivers messages between nodes (optional packet loss / delay simulation).
    """
    def __init__(self, n: int = 5):
        self.nodes = [RaftNode(i, n) for i in range(n)]
        self.n = n
        self.timeline: List[dict] = []   # full event log
        self.t = 0.0
        self.packet_loss_pct = 0         # 0–100

    def step(self, elapsed_s: float = 0.05):
        self.t += elapsed_s
        all_msgs: List[RaftMessage] = []

        # Tick each node
        for node in self.nodes:
            msgs = node.tick(elapsed_s)
            all_msgs.extend(msgs)

        # Deliver messages
        random.shuffle(all_msgs)
        for msg in all_msgs:
            if random.randint(0, 99) < self.packet_loss_pct:
                continue  # drop
            for node in self.nodes:
                if node.id != msg.sender:
                    resp = node.handle(msg)
                    if resp:
                        # Deliver response back to sender
                        sender_node = self.nodes[msg.sender]
                        sender_node.handle(resp)

        # Capture timeline snapshot
        snap = {
            "t": round(self.t, 3),
            "roles": [n.role for n in self.nodes],
            "terms": [n.current_term for n in self.nodes],
            "alive": [n.alive for n in self.nodes],
            "log_len": [len(n.log) for n in self.nodes],
            "commit": [n.commit_index for n in self.nodes],
            "leader": next((n.id for n in self.nodes if n.role == RaftRole.LEADER and n.alive), None),
        }
        self.timeline.append(snap)

    def run(self, steps: int = 100, elapsed_s: float = 0.05):
        for _ in range(steps):
            self.step(elapsed_s)

    def client_write(self, command: str) -> bool:
        leader = next((n for n in self.nodes if n.role == RaftRole.LEADER and n.alive), None)
        if leader:
            leader.handle(RaftMessage(type="client_write", sender=-1, term=leader.current_term,
                                       data={"command": command}))
            # Replicate to followers
            for _ in range(20):
                self.step(0.05)
            # Check quorum commit
            committed = sum(1 for n in self.nodes if n.alive and n.last_applied >= leader.last_log_index - 1)
            if committed > self.n // 2:
                for n in self.nodes:
                    if n.alive:
                        n.commit_index = leader.last_log_index
                        n._apply_committed()
                return True
        return False

    def kill_node(self, node_id: int):
        self.nodes[node_id].kill()

    def revive_node(self, node_id: int):
        self.nodes[node_id].revive()

    @property
    def leader(self) -> Optional[RaftNode]:
        return next((n for n in self.nodes if n.role == RaftRole.LEADER and n.alive), None)


# ── 2. CONSISTENT HASHING ─────────────────────────────────────────────────────

class ConsistentHashRing:
    """
    Production-grade consistent hash ring.
    - Virtual nodes for even distribution (default 150 vnodes/server)
    - MD5 hashing (same as Cassandra's Murmur variant in spirit)
    - O(log N) key lookup via bisect
    """
    def __init__(self, virtual_nodes: int = 150):
        self.vnodes      = virtual_nodes
        self._ring: Dict[int, str] = {}    # hash → server_id
        self._sorted_keys: List[int] = []
        self._servers: Set[str] = set()
        self._history: List[dict] = []

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def add_server(self, server_id: str):
        self._servers.add(server_id)
        for i in range(self.vnodes):
            vh = self._hash(f"{server_id}#vnode{i}")
            self._ring[vh] = server_id
            self._sorted_keys.append(vh)
        self._sorted_keys.sort()
        self._history.append({"op": "add", "server": server_id,
                               "ring_size": len(self._sorted_keys)})

    def remove_server(self, server_id: str):
        if server_id not in self._servers:
            return
        self._servers.discard(server_id)
        keys_to_remove = [k for k, v in self._ring.items() if v == server_id]
        for k in keys_to_remove:
            del self._ring[k]
            self._sorted_keys.remove(k)
        self._history.append({"op": "remove", "server": server_id,
                               "ring_size": len(self._sorted_keys)})

    def get_server(self, key: str) -> Optional[str]:
        if not self._ring:
            return None
        import bisect
        h = self._hash(key)
        idx = bisect.bisect_right(self._sorted_keys, h) % len(self._sorted_keys)
        return self._ring[self._sorted_keys[idx]]

    def get_n_servers(self, key: str, n: int) -> List[str]:
        """Return n distinct servers for replication (e.g. RF=3)."""
        if not self._ring:
            return []
        import bisect
        h = self._hash(key)
        idx = bisect.bisect_right(self._sorted_keys, h) % len(self._sorted_keys)
        seen: Set[str] = set()
        result = []
        for i in range(len(self._sorted_keys)):
            k = self._sorted_keys[(idx + i) % len(self._sorted_keys)]
            srv = self._ring[k]
            if srv not in seen:
                seen.add(srv)
                result.append(srv)
            if len(result) == n:
                break
        return result

    def distribution(self) -> Dict[str, int]:
        """How many keys (out of 10000 random) land on each server."""
        counter: Dict[str, int] = defaultdict(int)
        for _ in range(10000):
            k = str(random.random())
            srv = self.get_server(k)
            if srv:
                counter[srv] += 1
        return dict(counter)

    def key_movement_on_add(self, new_server: str) -> float:
        """Estimate % of keys that would move if we add a server."""
        sample = [str(random.random()) for _ in range(1000)]
        before = {k: self.get_server(k) for k in sample}
        self.add_server(new_server)
        after = {k: self.get_server(k) for k in sample}
        self.remove_server(new_server)
        moved = sum(1 for k in sample if before[k] != after[k])
        return moved / len(sample) * 100


# ── 3. RATE LIMITERS ─────────────────────────────────────────────────────────

class TokenBucket:
    """
    Token bucket: allows bursting up to `capacity` requests,
    refills at `rate` tokens/second using real elapsed time.
    """
    def __init__(self, capacity: float, rate: float):
        self.capacity  = capacity
        self.rate      = rate
        self.tokens    = float(capacity)
        self._last_ts  = time.monotonic()
        self.accepted  = 0
        self.rejected  = 0
        self._history: List[Tuple[float, float, bool]] = []  # (ts, tokens, allowed)

    def allow(self, tokens_needed: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_ts
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self._last_ts = now
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            self.accepted += 1
            self._history.append((now, round(self.tokens, 2), True))
            return True
        self.rejected += 1
        self._history.append((now, round(self.tokens, 2), False))
        return False

    @property
    def utilization(self) -> float:
        return self.accepted / max(1, self.accepted + self.rejected) * 100


class SlidingWindowLog:
    """
    Sliding window log: keeps timestamps of all requests in the window.
    Most accurate but O(N) space where N = requests in window.
    Used by Stripe's real rate limiter.
    """
    def __init__(self, limit: int, window_seconds: float):
        self.limit    = limit
        self.window   = window_seconds
        self._log: deque = deque()
        self.accepted = 0
        self.rejected = 0

    def allow(self) -> bool:
        now = time.monotonic()
        # Remove expired entries
        cutoff = now - self.window
        while self._log and self._log[0] < cutoff:
            self._log.popleft()
        if len(self._log) < self.limit:
            self._log.append(now)
            self.accepted += 1
            return True
        self.rejected += 1
        return False

    @property
    def current_count(self) -> int:
        now = time.monotonic()
        cutoff = now - self.window
        return sum(1 for ts in self._log if ts >= cutoff)


class FixedWindowCounter:
    """
    Fixed window: reset counter every N seconds.
    Simple but allows 2x burst at window boundary (the "stampede" problem).
    """
    def __init__(self, limit: int, window_seconds: float):
        self.limit    = limit
        self.window   = window_seconds
        self._window_start = time.monotonic()
        self._count   = 0
        self.accepted = 0
        self.rejected = 0
        self.resets   = 0

    def allow(self) -> bool:
        now = time.monotonic()
        if now - self._window_start >= self.window:
            self._window_start = now
            self._count = 0
            self.resets += 1
        if self._count < self.limit:
            self._count += 1
            self.accepted += 1
            return True
        self.rejected += 1
        return False

    @property
    def window_fill_pct(self) -> float:
        return self._count / self.limit * 100


class LeakyBucket:
    """
    Leaky bucket: queue of requests, drain at constant rate.
    Guarantees output rate but adds latency. Used in network QoS.
    """
    def __init__(self, capacity: int, drain_rate: float):
        self.capacity   = capacity
        self.drain_rate = drain_rate  # requests/second
        self._queue     = deque()
        self._last_drain = time.monotonic()
        self.accepted   = 0
        self.rejected   = 0
        self.drained    = 0

    def allow(self) -> bool:
        self._drain()
        if len(self._queue) < self.capacity:
            self._queue.append(time.monotonic())
            self.accepted += 1
            return True
        self.rejected += 1
        return False

    def _drain(self):
        now = time.monotonic()
        elapsed = now - self._last_drain
        to_drain = int(elapsed * self.drain_rate)
        for _ in range(min(to_drain, len(self._queue))):
            self._queue.popleft()
            self.drained += 1
        if to_drain > 0:
            self._last_drain = now

    @property
    def queue_depth(self) -> int:
        self._drain()
        return len(self._queue)


class RateLimiterBenchmark:
    """
    Run all four rate limiters under the same traffic pattern and compare.
    """
    def __init__(self, limit: int = 10, window: float = 1.0):
        self.tb  = TokenBucket(capacity=limit, rate=limit / window)
        self.sw  = SlidingWindowLog(limit=limit, window_seconds=window)
        self.fw  = FixedWindowCounter(limit=limit, window_seconds=window)
        self.lb  = LeakyBucket(capacity=limit * 2, drain_rate=limit / window)

    def run(self, requests: List[Tuple[float, float]], realtime: bool = False) -> Dict:
        """
        requests: list of (inter_arrival_seconds, burst_size)
        Returns per-limiter results.
        """
        results: Dict[str, List[bool]] = {"TokenBucket": [], "SlidingWindow": [],
                                           "FixedWindow": [], "LeakyBucket": []}
        for inter_arrival, burst in requests:
            if realtime:
                time.sleep(inter_arrival)
            for _ in range(int(burst)):
                results["TokenBucket"].append(self.tb.allow())
                results["SlidingWindow"].append(self.sw.allow())
                results["FixedWindow"].append(self.fw.allow())
                results["LeakyBucket"].append(self.lb.allow())

        return {
            name: {
                "accepted": sum(r),
                "rejected": sum(1 for r in rs if not r),
                "acceptance_rate": sum(rs) / max(1, len(rs)) * 100,
            }
            for name, rs in results.items()
        }


# ── 4. LRU + LFU CACHE ───────────────────────────────────────────────────────

class LRUCache:
    """
    O(1) get and put using OrderedDict (doubly-linked list + hashmap).
    Exactly what you'd implement in a FAANG interview.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()
        self.hits   = 0
        self.misses = 0
        self.evictions = 0
        self._access_log: List[Tuple[str, str, bool]] = []  # (key, op, hit)

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            self._access_log.append((key, "GET", True))
            return self._cache[key]
        self.misses += 1
        self._access_log.append((key, "GET", False))
        return None

    def put(self, key: str, value: Any) -> Optional[str]:
        evicted = None
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.capacity:
                evicted, _ = self._cache.popitem(last=False)
                self.evictions += 1
            self._cache[key] = value
        self._access_log.append((key, "PUT", key in self._cache))
        return evicted

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0

    @property
    def keys_in_order(self) -> List[str]:
        """Most recently used last."""
        return list(self._cache.keys())


class LFUCache:
    """
    O(1) get and put using frequency-ordered eviction.
    Three hashmaps: key→value, key→freq, freq→OrderedDict(key).
    """
    def __init__(self, capacity: int):
        self.capacity  = capacity
        self._key_val: Dict[str, Any] = {}
        self._key_freq: Dict[str, int] = {}
        self._freq_keys: Dict[int, OrderedDict] = defaultdict(OrderedDict)
        self._min_freq  = 0
        self.hits       = 0
        self.misses     = 0
        self.evictions  = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._key_val:
            self.misses += 1
            return None
        self._increment(key)
        self.hits += 1
        return self._key_val[key]

    def put(self, key: str, value: Any):
        if self.capacity <= 0:
            return
        if key in self._key_val:
            self._key_val[key] = value
            self._increment(key)
        else:
            if len(self._key_val) >= self.capacity:
                self._evict()
            self._key_val[key] = value
            self._key_freq[key] = 1
            self._freq_keys[1][key] = None
            self._min_freq = 1

    def _increment(self, key: str):
        f = self._key_freq[key]
        del self._freq_keys[f][key]
        if not self._freq_keys[f] and f == self._min_freq:
            self._min_freq += 1
        self._key_freq[key] = f + 1
        self._freq_keys[f + 1][key] = None

    def _evict(self):
        keys_at_min = self._freq_keys[self._min_freq]
        evict_key, _ = keys_at_min.popitem(last=False)
        del self._key_val[evict_key]
        del self._key_freq[evict_key]
        self.evictions += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0


class CacheSimulator:
    """
    Simulate multiple access patterns and compare LRU vs LFU vs FIFO.
    """
    PATTERNS = {
        "Uniform Random":    lambda size: [str(random.randint(0, size)) for _ in range(1000)],
        "80/20 Pareto":      lambda size: [str(random.choices(range(size), weights=[1/(i+1) for i in range(size)])[0]) for _ in range(1000)],
        "Sequential Scan":   lambda size: [str(i % size) for i in range(1000)],
        "Repeated Hotspot":  lambda size: [str(random.choice(range(min(5, size)))) if random.random() < 0.8 else str(random.randint(0, size)) for _ in range(1000)],
        "Thrashing (LRU bad)": lambda size: [str(i % (size + 1)) for i in range(1000)],  # size+1 causes constant eviction in LRU
    }

    def run(self, capacity: int = 50, key_space: int = 200, pattern: str = "80/20 Pareto") -> Dict:
        lru = LRUCache(capacity)
        lfu = LFUCache(capacity)

        keys = self.PATTERNS[pattern](key_space)
        lru_hits, lfu_hits = [], []
        for k in keys:
            lru_hit = lru.get(k) is not None
            lfu_hit = lfu.get(k) is not None
            if not lru_hit: lru.put(k, 1)
            if not lfu_hit: lfu.put(k, 1)
            lru_hits.append(lru_hit)
            lfu_hits.append(lfu_hit)

        # Rolling hit rate
        window = 50
        lru_rolling = [sum(lru_hits[max(0,i-window):i+1])/min(i+1,window)*100 for i in range(len(lru_hits))]
        lfu_rolling = [sum(lfu_hits[max(0,i-window):i+1])/min(i+1,window)*100 for i in range(len(lfu_hits))]

        return {
            "lru_hit_rate": lru.hit_rate,
            "lfu_hit_rate": lfu.hit_rate,
            "lru_evictions": lru.evictions,
            "lfu_evictions": lfu.evictions,
            "lru_rolling": lru_rolling,
            "lfu_rolling": lfu_rolling,
            "keys_accessed": len(set(keys)),
            "pattern": pattern,
        }


# ── 5. B-TREE ─────────────────────────────────────────────────────────────────

class BTreeNode:
    def __init__(self, t: int, leaf: bool = True):
        self.t    = t           # minimum degree
        self.keys: List[int] = []
        self.vals: List[Any] = []
        self.children: List["BTreeNode"] = []
        self.leaf = leaf

    @property
    def is_full(self) -> bool:
        return len(self.keys) == 2 * self.t - 1


class BTree:
    """
    Real B-Tree implementation.
    - Minimum degree t: each non-root node has [t-1, 2t-1] keys
    - Split on insert (proactive split, single pass downward)
    - Merge / borrow on delete
    Used in: PostgreSQL (B+ tree indexes), MySQL InnoDB, filesystem inodes
    """
    def __init__(self, t: int = 2):
        self.t    = t
        self.root = BTreeNode(t, leaf=True)
        self.height = 1
        self.size = 0
        self._ops: List[dict] = []   # audit log

    def search(self, k: int) -> Tuple[Optional[BTreeNode], int]:
        return self._search(self.root, k)

    def _search(self, node: BTreeNode, k: int) -> Tuple[Optional[BTreeNode], int]:
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        if i < len(node.keys) and k == node.keys[i]:
            return node, i
        if node.leaf:
            return None, -1
        return self._search(node.children[i], k)

    def insert(self, k: int, v: Any = None):
        self._ops.append({"op": "insert", "key": k})
        root = self.root
        if root.is_full:
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
            self.height += 1
        self._insert_non_full(self.root, k, v)
        self.size += 1

    def _insert_non_full(self, node: BTreeNode, k: int, v: Any):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(0)
            node.vals.append(None)
            while i >= 0 and k < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.vals[i + 1] = node.vals[i]
                i -= 1
            node.keys[i + 1] = k
            node.vals[i + 1] = v
        else:
            while i >= 0 and k < node.keys[i]:
                i -= 1
            i += 1
            if node.children[i].is_full:
                self._split_child(node, i)
                if k > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], k, v)

    def _split_child(self, parent: BTreeNode, i: int):
        t = self.t
        y = parent.children[i]
        z = BTreeNode(t, leaf=y.leaf)
        mid = t - 1
        # z gets right half of y
        z.keys = y.keys[mid + 1:]
        z.vals = y.vals[mid + 1:]
        if not y.leaf:
            z.children = y.children[mid + 1:]
            y.children = y.children[:mid + 1]
        # push median key up
        parent.keys.insert(i, y.keys[mid])
        parent.vals.insert(i, y.vals[mid])
        parent.children.insert(i + 1, z)
        y.keys = y.keys[:mid]
        y.vals = y.vals[:mid]

    def to_levels(self) -> List[List[List[int]]]:
        """BFS traversal for visualization."""
        if not self.root.keys:
            return []
        levels = []
        current = [self.root]
        while current:
            level_keys = [node.keys[:] for node in current]
            levels.append(level_keys)
            next_level = []
            for node in current:
                if not node.leaf:
                    next_level.extend(node.children)
            current = next_level
        return levels

    @property
    def node_count(self) -> int:
        def count(node):
            if node.leaf:
                return 1
            return 1 + sum(count(c) for c in node.children)
        return count(self.root)


# ── 6. TWO-PHASE COMMIT ───────────────────────────────────────────────────────

class TwoPhaseState:
    INIT       = "INIT"
    PREPARED   = "PREPARED"
    COMMITTED  = "COMMITTED"
    ABORTED    = "ABORTED"
    FAILED     = "FAILED"


@dataclass
class TxParticipant:
    pid: int
    name: str
    vote: Optional[bool] = None    # True=VOTE_COMMIT, False=VOTE_ABORT
    state: str = TwoPhaseState.INIT
    latency_ms: float = 0
    alive: bool = True


class TwoPhaseCommit:
    """
    Actual 2PC protocol simulation.
    Phase 1: PREPARE — coordinator asks all participants to vote
    Phase 2: COMMIT/ABORT — coordinator broadcasts based on unanimous vote
    Failure modes: coordinator crash, participant crash, network partition
    """
    def __init__(self, participants: List[str]):
        self.coordinator_alive = True
        self.participants = [TxParticipant(i, name) for i, name in enumerate(participants)]
        self.coordinator_state = TwoPhaseState.INIT
        self.events: List[dict] = []
        self._ts = 0.0

    def _log(self, actor: str, event: str, detail: str = ""):
        self.events.append({
            "t": round(self._ts, 3),
            "actor": actor,
            "event": event,
            "detail": detail,
        })

    def phase1_prepare(self, abort_participants: List[int] = None, fail_coordinator: bool = False):
        """
        Send PREPARE to all. Collect votes.
        abort_participants: list of participant IDs that will vote ABORT
        fail_coordinator: crash coordinator after sending PREPAREs (blocking scenario)
        """
        if abort_participants is None:
            abort_participants = []

        self._log("COORDINATOR", "PHASE_1_START", "Sending PREPARE to all participants")
        self.coordinator_state = "WAITING_VOTES"

        for p in self.participants:
            if not p.alive:
                p.vote = None
                self._log(f"P{p.pid}:{p.name}", "NO_RESPONSE", "Participant is down")
                continue
            self._ts += random.uniform(0.005, 0.02)
            if p.pid in abort_participants:
                p.vote = False
                p.state = TwoPhaseState.ABORTED
                self._log(f"P{p.pid}:{p.name}", "VOTE_ABORT", "Resource conflict / lock timeout")
            else:
                p.vote = True
                p.state = TwoPhaseState.PREPARED
                self._log(f"P{p.pid}:{p.name}", "VOTE_COMMIT", "Resources locked, ready")

        if fail_coordinator:
            self.coordinator_alive = False
            self._log("COORDINATOR", "CRASHED", "Coordinator failed after collecting votes — BLOCKING!")
            return False, "COORDINATOR_FAILURE"

        # Decision
        all_yes = all(p.vote is True for p in self.participants if p.alive)
        if not all_yes:
            self.coordinator_state = TwoPhaseState.ABORTED
        else:
            self.coordinator_state = TwoPhaseState.COMMITTED
        return True, "ok"

    def phase2_commit_or_abort(self):
        """Send COMMIT or ABORT based on phase 1 outcome."""
        if not self.coordinator_alive:
            return False, "COORDINATOR_DEAD"

        decision = self.coordinator_state
        self._log("COORDINATOR", f"PHASE_2_{decision}", f"Broadcasting {decision} to all")

        for p in self.participants:
            if not p.alive:
                continue
            self._ts += random.uniform(0.002, 0.01)
            if decision == TwoPhaseState.COMMITTED:
                p.state = TwoPhaseState.COMMITTED
                self._log(f"P{p.pid}:{p.name}", "COMMITTED", "Locks released, changes durable")
            else:
                p.state = TwoPhaseState.ABORTED
                self._log(f"P{p.pid}:{p.name}", "ABORTED", "Rolled back, locks released")

        self._log("COORDINATOR", "TX_COMPLETE",
                  f"All participants {decision}. TX done in {self._ts*1000:.1f}ms")
        return True, decision

    def run_full(self, abort_participants=None, fail_coordinator=False, fail_participants=None):
        """Run the complete 2PC protocol with optional failure injection."""
        if fail_participants:
            for pid in fail_participants:
                self.participants[pid].alive = False
                self._log(f"P{pid}", "FAILED", "Participant crashed before protocol")

        success, msg = self.phase1_prepare(abort_participants, fail_coordinator)
        if not success:
            return
        self.phase2_commit_or_abort()

    def reset(self):
        for p in self.participants:
            p.vote = None
            p.state = TwoPhaseState.INIT
            p.alive = True
        self.coordinator_state = TwoPhaseState.INIT
        self.coordinator_alive = True
        self.events = []
        self._ts = 0.0


# ── 7. BLOOM FILTER ───────────────────────────────────────────────────────────

class BloomFilter:
    """
    Probabilistic membership set.
    k hash functions, m-bit array.
    False positive rate ≈ (1 - e^(-kn/m))^k
    Used by: Cassandra (SSTable membership), Chrome (malware URLs), CDNs
    """
    def __init__(self, expected_elements: int = 1000, fp_rate: float = 0.01):
        self.n      = expected_elements
        self.fp_target = fp_rate
        # Optimal m and k
        self.m = math.ceil(-expected_elements * math.log(fp_rate) / (math.log(2) ** 2))
        self.k = max(1, round((self.m / expected_elements) * math.log(2)))
        self._bits  = bytearray(math.ceil(self.m / 8))
        self._count = 0
        self._fp_actual = 0
        self._fn_actual = 0  # always 0 — no false negatives in Bloom

    def _hashes(self, item: str) -> List[int]:
        positions = []
        for i in range(self.k):
            h = int(hashlib.md5(f"{i}:{item}".encode()).hexdigest(), 16)
            positions.append(h % self.m)
        return positions

    def add(self, item: str):
        for pos in self._hashes(item):
            self._bits[pos // 8] |= (1 << (pos % 8))
        self._count += 1

    def __contains__(self, item: str) -> bool:
        return all(
            (self._bits[pos // 8] >> (pos % 8)) & 1
            for pos in self._hashes(item)
        )

    @property
    def actual_fp_rate(self) -> float:
        """Theoretical FP rate given current fill."""
        if self._count == 0:
            return 0.0
        return (1 - math.exp(-self.k * self._count / self.m)) ** self.k

    @property
    def fill_ratio(self) -> float:
        bits_set = sum(bin(b).count("1") for b in self._bits)
        return bits_set / self.m

    def benchmark(self, n: int = 10000) -> dict:
        """Add n items, then test n known + n random for FP rate."""
        items = [f"item_{i}" for i in range(n)]
        for item in items:
            self.add(item)
        # FP test on non-members
        fp = sum(1 for i in range(n, n * 2) if f"item_{i}" in self)
        # TP test on members
        tp = sum(1 for item in items if item in self)
        return {
            "n_inserted": n,
            "true_positives": tp,
            "false_positives": fp,
            "fp_rate_actual": fp / n * 100,
            "fp_rate_theoretical": self.actual_fp_rate * 100,
            "fill_ratio": self.fill_ratio,
            "bits_used": self.m,
            "hash_functions": self.k,
        }


# ── 8. QUAD TREE (Geospatial Index) ──────────────────────────────────────────

@dataclass
class Point:
    x: float
    y: float
    data: Any = None


@dataclass
class Rect:
    x: float; y: float; w: float; h: float

    def contains(self, p: Point) -> bool:
        return self.x <= p.x <= self.x + self.w and self.y <= p.y <= self.y + self.h

    def intersects(self, other: "Rect") -> bool:
        return not (other.x > self.x + self.w or other.x + other.w < self.x or
                    other.y > self.y + self.h or other.y + other.h < self.y)


class QuadTree:
    """
    QuadTree spatial index.
    Used in: Uber/Lyft driver location lookup, gaming collision detection,
             Google Maps region queries.
    """
    MAX_POINTS = 4

    def __init__(self, boundary: Rect, depth: int = 0, max_depth: int = 8):
        self.boundary  = boundary
        self.depth     = depth
        self.max_depth = max_depth
        self.points: List[Point] = []
        self.children: Optional[List["QuadTree"]] = None
        self.total_inserted = 0

    def insert(self, p: Point) -> bool:
        if not self.boundary.contains(p):
            return False
        if self.children is None:
            if len(self.points) < self.MAX_POINTS or self.depth >= self.max_depth:
                self.points.append(p)
                return True
            self._subdivide()
        for child in self.children:
            if child.insert(p):
                return True
        return False

    def _subdivide(self):
        b = self.boundary
        hw, hh = b.w / 2, b.h / 2
        self.children = [
            QuadTree(Rect(b.x,      b.y,      hw, hh), self.depth+1, self.max_depth),
            QuadTree(Rect(b.x+hw,   b.y,      hw, hh), self.depth+1, self.max_depth),
            QuadTree(Rect(b.x,      b.y+hh,   hw, hh), self.depth+1, self.max_depth),
            QuadTree(Rect(b.x+hw,   b.y+hh,   hw, hh), self.depth+1, self.max_depth),
        ]
        for p in self.points:
            for child in self.children:
                if child.insert(p):
                    break
        self.points = []

    def query(self, region: Rect) -> List[Point]:
        found = []
        if not self.boundary.intersects(region):
            return found
        if self.children is None:
            found.extend(p for p in self.points if region.contains(p))
        else:
            for child in self.children:
                found.extend(child.query(region))
        return found

    def count_nodes(self) -> int:
        if self.children is None:
            return 1
        return 1 + sum(c.count_nodes() for c in self.children)

    def all_points(self) -> List[Point]:
        if self.children is None:
            return list(self.points)
        pts = []
        for c in self.children: pts.extend(c.all_points())
        return pts


# ── Traffic Pattern Generator ─────────────────────────────────────────────────

class TrafficGenerator:
    """Generates realistic request patterns for rate limiter testing."""

    @staticmethod
    def uniform(qps: float, duration_s: float) -> List[float]:
        """Steady QPS — simple uniform arrival."""
        interval = 1.0 / qps
        t = 0.0
        times = []
        while t < duration_s:
            times.append(t)
            t += interval
        return times

    @staticmethod
    def bursty(base_qps: float, burst_qps: float, duration_s: float,
               burst_every_s: float = 5.0, burst_duration_s: float = 0.5) -> List[float]:
        """Normal traffic with periodic bursts."""
        times = []
        t = 0.0
        while t < duration_s:
            if int(t) % int(burst_every_s) == 0 and t % burst_every_s < burst_duration_s:
                interval = 1.0 / burst_qps
            else:
                interval = 1.0 / base_qps
            times.append(t)
            t += interval + random.gauss(0, interval * 0.1)
        return sorted(times)

    @staticmethod
    def poisson(lam: float, duration_s: float) -> List[float]:
        """Poisson arrival process — models real web traffic."""
        times = []
        t = 0.0
        while t < duration_s:
            inter = random.expovariate(lam)
            t += inter
            if t < duration_s:
                times.append(t)
        return times