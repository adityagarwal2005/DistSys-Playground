"""
app.py — DistSys Playground
A living laboratory of distributed systems algorithms.
Every chart is the output of real code running right now.

Run: streamlit run app.py
"""

import random
import time
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from algorithms import (
    RaftCluster, RaftRole,
    ConsistentHashRing,
    TokenBucket, SlidingWindowLog, FixedWindowCounter, LeakyBucket,
    RateLimiterBenchmark, TrafficGenerator,
    LRUCache, LFUCache, CacheSimulator,
    BTree, BTreeNode,
    TwoPhaseCommit, TwoPhaseState,
    BloomFilter,
    QuadTree, Rect, Point,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DistSys Playground",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');
:root {
  --bg:#0d1117;--s1:#161b22;--s2:#21262d;--s3:#30363d;
  --br:#30363d;--br2:#484f58;
  --ink:#e6edf3;--ink2:#c9d1d9;--ink3:#8b949e;--ink4:#6e7681;
  --b:#58a6ff;--bd:#1f6feb;--bbg:#0d2b50;
  --g:#3fb950;--gbg:#0a3020;
  --o:#d29922;--obg:#2d1f00;
  --r:#f85149;--rbg:#3d0c0a;
  --p:#bc8cff;--pbg:#1e0f2e;
  --y:#e3b341;--ybg:#2d2200;
  --t:#5ac8fa;--tbg:#0a2030;
  --mono:'JetBrains Mono','Fira Code',monospace;
  --sans:'Inter',-apple-system,sans-serif;
}
*{font-family:var(--sans)!important}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--ink)!important}
.main,.block-container{background:var(--bg)!important}
.block-container{padding:1.25rem 1.75rem!important;max-width:1500px!important}
h1{font-size:1.65rem!important;font-weight:700!important;letter-spacing:-.03em!important;color:var(--ink)!important}
h2{font-size:1.25rem!important;font-weight:600!important;color:var(--ink)!important}
h3{font-size:1rem!important;font-weight:600!important;color:var(--ink2)!important}
[data-testid="stSidebar"]{background:var(--s1)!important;border-right:1px solid var(--br)!important}
.stButton>button{background:var(--s2)!important;color:var(--ink)!important;border:1px solid var(--br)!important;border-radius:6px!important;font-size:.82rem!important;font-weight:500!important;padding:.42rem .9rem!important;transition:all .14s!important}
.stButton>button:hover{background:var(--s3)!important;border-color:var(--br2)!important;transform:translateY(-1px)!important}
.run-btn>.stButton>button{background:var(--bd)!important;border-color:var(--b)!important;color:#fff!important;font-weight:600!important}
.run-btn>.stButton>button:hover{box-shadow:0 0 14px rgba(88,166,255,.3)!important}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea,[data-baseweb="select"]>div{background:var(--s2)!important;border:1px solid var(--br)!important;color:var(--ink)!important;border-radius:5px!important}
[data-testid="stMetric"]{background:var(--s1)!important;border:1px solid var(--br)!important;border-radius:7px!important;padding:.85rem!important}
[data-testid="stMetricValue"]{font-size:1.4rem!important;font-weight:700!important;color:var(--b)!important;font-family:var(--mono)!important}
[data-testid="stMetricLabel"]{font-size:.68rem!important;color:var(--ink3)!important;text-transform:uppercase;letter-spacing:.05em}
[data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--br)!important}
[data-baseweb="tab"]{font-size:.8rem!important;font-weight:500!important;color:var(--ink3)!important;border:none!important;padding:.48rem .8rem!important}
[aria-selected="true"]{color:var(--b)!important;border-bottom:2px solid var(--b)!important;background:transparent!important}
.stAlert{border-radius:6px!important;font-size:.84rem!important}
.stSlider{padding:.15rem 0!important}
.dc{background:var(--s1);border:1px solid var(--br);border-radius:8px;padding:1rem 1.2rem;margin-bottom:.55rem}
.dc:hover{border-color:var(--br2)}
.dc-b{border-color:var(--bd);background:var(--bbg)}
.dc-g{border-color:var(--g);background:var(--gbg)}
.dc-r{border-color:var(--r);background:var(--rbg)}
.dc-o{border-color:var(--o);background:var(--obg)}
.dc-p{border-color:var(--p);background:var(--pbg)}
.dc-y{border-color:var(--y);background:var(--ybg)}
.bd{display:inline-block;padding:.13rem .5rem;border-radius:12px;font-size:.7rem;font-weight:500}
.bd-b{background:var(--bbg);color:var(--b);border:1px solid var(--bd)}
.bd-g{background:var(--gbg);color:var(--g);border:1px solid var(--g)}
.bd-r{background:var(--rbg);color:var(--r);border:1px solid var(--r)}
.bd-o{background:var(--obg);color:var(--o);border:1px solid var(--o)}
.bd-p{background:var(--pbg);color:var(--p);border:1px solid var(--p)}
.bd-y{background:var(--ybg);color:var(--y);border:1px solid var(--y)}
.bd-gray{background:var(--s2);color:var(--ink3);border:1px solid var(--br)}
.mono{font-family:var(--mono)!important;font-size:.82rem}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:var(--s3);border-radius:4px}
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ─────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Inter", size=10, color="#8b949e"),
    margin=dict(l=4, r=4, t=32, b=4),
)
GRID = dict(showgrid=True, gridcolor="#21262d", zeroline=False)
NO_GRID = dict(showgrid=False, zeroline=False)

ROLE_COLOR = {
    RaftRole.LEADER:    "#3fb950",
    RaftRole.CANDIDATE: "#e3b341",
    RaftRole.FOLLOWER:  "#58a6ff",
    "Dead":             "#f85149",
}

def B(t, c="bd-b"): return f'<span class="bd {c}">{t}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""<div style="padding:.5rem 0 .3rem;text-align:center">
          <div style="font-size:1.5rem">⚙️</div>
          <div style="font-size:1rem;font-weight:700;letter-spacing:-.02em">DistSys Playground</div>
          <div style="font-size:.68rem;color:#8b949e;margin-top:.15rem">Real algorithms. Real execution.</div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        PAGES = [
            ("🏠", "Home"),
            ("⚡", "Raft Consensus"),
            ("🔵", "Consistent Hashing"),
            ("🚦", "Rate Limiters"),
            ("📦", "Cache Eviction"),
            ("🌳", "B-Tree"),
            ("🤝", "2-Phase Commit"),
            ("🌸", "Bloom Filter"),
            ("🗺️", "QuadTree Geo"),
        ]
        tab = st.session_state.get("tab", "Home")
        for icon, label in PAGES:
            active = tab == label
            style = "color:#58a6ff;font-weight:600" if active else "color:#c9d1d9"
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                st.session_state.tab = label
                st.rerun()

        st.divider()
        st.markdown("""<div style="font-size:.68rem;color:#6e7681;line-height:1.7;padding:.3rem 0">
          <div style="font-weight:600;color:#8b949e;margin-bottom:.2rem">All implementations:</div>
          Raft Node · Token Bucket · Sliding Window · Fixed Window · Leaky Bucket ·
          LRU O(1) · LFU O(1) · B-Tree · 2PC · Bloom Filter · QuadTree · Consistent Hash
        </div>""", unsafe_allow_html=True)


def ss(k, v=None):
    if k not in st.session_state: st.session_state[k] = v
    return st.session_state[k]

ss("tab", "Home")
sidebar()
tab = st.session_state.tab


# ── HOME ──────────────────────────────────────────────────────────────────────
if tab == "Home":
    st.markdown("## ⚙️ DistSys Playground")
    st.markdown('<p style="color:#8b949e;font-size:.92rem;margin-bottom:1.5rem">Every algorithm implemented from scratch in Python. Every chart generated from real execution. No mocks, no fake data.</p>', unsafe_allow_html=True)

    cards = [
        ("⚡", "Raft Consensus",      "#58a6ff", "Leader election, log replication, fault tolerance. Run 5 nodes, kill any of them, watch re-election happen in real time."),
        ("🔵", "Consistent Hashing",  "#bc8cff", "MD5-based ring + virtual nodes. Add/remove servers and watch keys redistribute. Real migration % calculation."),
        ("🚦", "Rate Limiters",       "#3fb950", "Token Bucket, Sliding Window Log, Fixed Window, Leaky Bucket — all four running against the same traffic pattern. See where each breaks."),
        ("📦", "Cache Eviction",      "#e3b341", "LRU (OrderedDict) vs LFU (freq buckets) under 5 access patterns. Rolling hit rate chart. See why LFU wins on 80/20 and LRU wins on recency."),
        ("🌳", "B-Tree",              "#d29922", "Real B-tree: proactive splits, merge on underflow. Insert/delete keys, watch the tree restructure. See exactly how PostgreSQL indexes work."),
        ("🤝", "Two-Phase Commit",    "#5ac8fa", "Coordinator + participants. Inject coordinator crashes, participant aborts, network splits. See the blocking problem live."),
        ("🌸", "Bloom Filter",        "#f85149", "k hash functions, m-bit array. Tune expected elements and FP rate, watch space usage. Benchmark real false positive rate vs theoretical."),
        ("🗺️", "QuadTree Geospatial", "#3fb950", "Spatial indexing as used by Uber/Lyft. Insert driver locations, query a bounding box. See O(log N) spatial lookup."),
    ]
    cols = st.columns(4)
    for i, (icon, title, color, desc) in enumerate(cards):
        with cols[i % 4]:
            if st.button(f"{icon} {title}", key=f"home_{title}", use_container_width=True):
                st.session_state.tab = title
                st.rerun()
            st.markdown(f'<div style="font-size:.73rem;color:#8b949e;margin-bottom:.75rem;padding:.15rem .1rem">{desc}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""<div class="dc-b dc">
      <div style="font-size:.8rem;font-weight:600;color:#58a6ff;margin-bottom:.4rem">⚡ Why This Project</div>
      <div style="font-size:.82rem;color:#c9d1d9;line-height:1.7">
        Every algorithm here appears in FAANG system design interviews. The difference between understanding and <em>implementing</em>
        is massive — when you've written a Raft state machine from scratch, you don't recite from memory, you reason from first principles.
        <br><br>
        This isn't a tutorial viewer. It's a playground where you change parameters and immediately see the system behavior.
        Token bucket with capacity=10 and a burst of 50? See exactly which requests get dropped.
        Kill the Raft leader during a client write? Watch the election and find out if the write was lost.
      </div>
    </div>""", unsafe_allow_html=True)


# ── RAFT CONSENSUS ────────────────────────────────────────────────────────────
elif tab == "Raft Consensus":
    st.markdown("## ⚡ Raft Consensus")
    st.markdown("""<div class="dc" style="font-size:.8rem;color:#8b949e;line-height:1.6;margin-bottom:1rem">
      <strong style="color:#58a6ff">Real implementation:</strong> Each node is a <code>RaftNode</code> state machine.
      Randomized election timeout (150–300ms). Majority quorum for elections and commits.
      Terms prevent split-brain. Log entries replicated via AppendEntries.
      This is how etcd, CockroachDB, and TiKV achieve consensus.
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_vis = st.columns([1, 3])

    with col_ctrl:
        n_nodes = st.slider("Cluster Size", 3, 7, 5, 2, key="raft_n")
        steps = st.slider("Steps to run", 20, 200, 80, 10, key="raft_steps")
        packet_loss = st.slider("Packet Loss %", 0, 40, 0, 5, key="raft_loss")
        kill_node_id = st.selectbox("Kill Node", ["None"] + list(range(n_nodes)), key="raft_kill")

        st.markdown("---")
        client_cmd = st.text_input("Client Write", value="SET x=42", key="raft_cmd")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_raft = st.button("▶ Run Cluster", use_container_width=True, key="run_raft")
        st.markdown('</div>', unsafe_allow_html=True)
        write_btn = st.button("✏️ Client Write", use_container_width=True, key="raft_write")

    if "raft_cluster" not in st.session_state:
        st.session_state.raft_cluster = None

    if run_raft:
        with st.spinner("Running Raft cluster..."):
            c = RaftCluster(n_nodes)
            c.packet_loss_pct = packet_loss
            c.run(steps, 0.05)
            if kill_node_id != "None":
                c.kill_node(int(kill_node_id))
                c.run(40, 0.05)   # run after failure to trigger re-election
            st.session_state.raft_cluster = c
            st.session_state.raft_wrote = None

    if write_btn and st.session_state.raft_cluster:
        c = st.session_state.raft_cluster
        ok = c.client_write(client_cmd)
        st.session_state.raft_wrote = {"cmd": client_cmd, "ok": ok}

    with col_vis:
        c = st.session_state.raft_cluster
        if not c:
            st.info("Configure and click ▶ Run Cluster to start the simulation.")
        else:
            # Current state
            st.markdown("#### Current Node States")
            node_cols = st.columns(len(c.nodes))
            for i, node in enumerate(c.nodes):
                with node_cols[i]:
                    if not node.alive:
                        role_display = "Dead"
                        clr = "#f85149"
                        cls = "dc-r"
                    else:
                        role_display = node.role
                        clr = ROLE_COLOR.get(node.role, "#8b949e")
                        cls = "dc-b" if node.role == RaftRole.LEADER else ("dc-o" if node.role == RaftRole.CANDIDATE else "dc")
                    leader_crown = " 👑" if node.role == RaftRole.LEADER and node.alive else ""
                    st.markdown(f"""<div class="{cls} dc" style="text-align:center;padding:.75rem">
                      <div style="font-size:1rem;font-weight:700;color:{clr}">N{node.id}{leader_crown}</div>
                      <div style="font-size:.72rem;color:{clr};margin:.15rem 0">{role_display}</div>
                      <div style="font-size:.68rem;color:#8b949e">term={node.current_term}</div>
                      <div style="font-size:.68rem;color:#8b949e">log={len(node.log)} ci={node.commit_index}</div>
                    </div>""", unsafe_allow_html=True)

            # Timeline chart
            if c.timeline:
                df = pd.DataFrame([{
                    "Step": snap["t"],
                    **{f"N{i}_term": snap["terms"][i] for i in range(len(snap["terms"]))},
                    "leader": str(snap["leader"]),
                } for snap in c.timeline])

                fig = go.Figure()
                colors = ["#58a6ff","#3fb950","#e3b341","#f85149","#bc8cff","#5ac8fa","#d29922"]
                for i in range(len(c.nodes)):
                    fig.add_trace(go.Scatter(
                        x=df["Step"], y=df[f"N{i}_term"],
                        mode="lines", name=f"Node {i}",
                        line=dict(color=colors[i % len(colors)], width=1.8),
                    ))
                fig.update_layout(**DARK, title="Term Evolution per Node", height=200,
                                   xaxis=dict(**NO_GRID, title="Simulation Time"),
                                   yaxis=dict(**GRID, title="Term", dtick=1),
                                   legend=dict(orientation="h", y=-0.3, font=dict(size=9)))
                st.plotly_chart(fig, use_container_width=True)

                # Role timeline
                role_fig = go.Figure()
                role_num = {RaftRole.FOLLOWER: 0, RaftRole.CANDIDATE: 1, RaftRole.LEADER: 2}
                for i, node in enumerate(c.nodes):
                    y_vals = []
                    for snap in c.timeline:
                        if not snap["alive"][i]:
                            y_vals.append(-1)
                        else:
                            y_vals.append(role_num.get(snap["roles"][i], 0) + i * 3)
                    role_fig.add_trace(go.Scatter(
                        x=df["Step"], y=y_vals,
                        mode="lines", name=f"N{i}",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
                role_fig.update_layout(**DARK, title="Role Over Time (0=Follower, 1=Candidate, 2=Leader)", height=180,
                                        xaxis=dict(**NO_GRID), yaxis=dict(**GRID),
                                        legend=dict(orientation="h", y=-0.35, font=dict(size=9)),
                                        showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Write result
            if st.session_state.get("raft_wrote"):
                wr = st.session_state.raft_wrote
                if wr["ok"]:
                    st.success(f"✅ `{wr['cmd']}` committed to state machine")
                    leader = c.leader
                    if leader:
                        st.code(str(leader.state_machine), language="python")
                else:
                    st.error("❌ Write failed — no leader available")

            # Metrics
            st.markdown("#### Cluster Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            elections = sum(n.elections_started for n in c.nodes)
            entries   = sum(n.entries_replicated for n in c.nodes)
            leader    = c.leader
            m1.metric("Elections", elections)
            m2.metric("Leader", f"N{leader.id}" if leader else "None")
            m3.metric("Leader Term", leader.current_term if leader else "—")
            m4.metric("Log Entries", entries)
            m5.metric("Alive Nodes", sum(1 for n in c.nodes if n.alive))

            # Event log
            with st.expander("📋 Node Event Log (last 20)"):
                all_events = []
                for node in c.nodes:
                    for ev in node.events[-10:]:
                        all_events.append({"Node": f"N{node.id}", **ev})
                all_events.sort(key=lambda x: x["time"])
                df_ev = pd.DataFrame(all_events[-30:])
                st.dataframe(df_ev, use_container_width=True, hide_index=True)


# ── CONSISTENT HASHING ────────────────────────────────────────────────────────
elif tab == "Consistent Hashing":
    st.markdown("## 🔵 Consistent Hashing")
    st.markdown("""<div class="dc" style="font-size:.8rem;color:#8b949e;line-height:1.6;margin-bottom:1rem">
      <strong style="color:#bc8cff">Real implementation:</strong>
      MD5-based ring with virtual nodes. <code>bisect_right</code> for O(log N) key lookup.
      With 150 vnodes/server, key distribution standard deviation drops from ~50% to ~3%.
      This is how Cassandra, DynamoDB, and Riak route data.
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_vis = st.columns([1, 2.5])

    with col_ctrl:
        n_servers = st.slider("Initial Servers", 2, 8, 4, 1, key="ch_n")
        vnodes    = st.slider("Virtual Nodes / Server", 1, 300, 150, 10, key="ch_vn")
        new_srv   = st.text_input("Add Server", value="server_new", key="ch_new")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_ch = st.button("▶ Build Ring", use_container_width=True, key="run_ch")
        st.markdown('</div>', unsafe_allow_html=True)
        add_srv = st.button("➕ Add Server", use_container_width=True, key="ch_add")
        rm_srv  = st.selectbox("Remove Server", ["—"] + [f"server_{i}" for i in range(n_servers)], key="ch_rm")
        do_rm   = st.button("➖ Remove Server", use_container_width=True, key="do_rm")

        st.markdown("---")
        lookup_key = st.text_input("Lookup Key", value="user_1234", key="ch_lookup")
        rf = st.slider("Replication Factor", 1, 4, 3, key="ch_rf")
        do_lookup = st.button("🔍 Route Key", use_container_width=True, key="do_lookup")

    if "ch_ring" not in st.session_state:
        st.session_state.ch_ring = None

    if run_ch:
        ring = ConsistentHashRing(virtual_nodes=vnodes)
        for i in range(n_servers):
            ring.add_server(f"server_{i}")
        st.session_state.ch_ring = ring

    ring = st.session_state.ch_ring
    if add_srv and ring:
        movement = ring.key_movement_on_add(new_srv)
        ring.add_server(new_srv)
        st.success(f"Added {new_srv}. ~{movement:.1f}% of keys migrated (expected {100/len(ring._servers):.1f}%)")

    if do_rm and rm_srv != "—" and ring:
        ring.remove_server(rm_srv)
        st.warning(f"Removed {rm_srv}")

    with col_vis:
        if not ring:
            st.info("Click ▶ Build Ring to start.")
        else:
            dist = ring.distribution()

            # Ring visualization using polar plot
            import math
            st.markdown("#### Hash Ring Visualization")

            servers = list(ring._servers)
            colors_map = {srv: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] for i, srv in enumerate(servers)}

            # Get ring positions for each vnode
            ring_fig = go.Figure()
            # Add arc for each server's virtual nodes
            for srv in servers:
                positions = [ring._ring[k] for k in ring._sorted_keys if ring._ring.get(k) == srv]
                angles = [pos / (2**32) * 360 for pos in positions[:50]]  # sample 50
                r_vals = [1.0] * len(angles)
                ring_fig.add_trace(go.Scatterpolar(
                    r=r_vals, theta=angles,
                    mode="markers",
                    marker=dict(size=4, color=colors_map[srv]),
                    name=srv,
                ))
            ring_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 1.5]),
                    angularaxis=dict(direction="clockwise", period=360),
                    bgcolor="#161b22",
                ),
                **DARK, height=380, title="Hash Ring (each dot = 1 virtual node)",
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, font=dict(size=9)),
            )
            st.plotly_chart(ring_fig, use_container_width=True)

            # Distribution bar chart
            df_dist = pd.DataFrame(list(dist.items()), columns=["Server", "Keys"])
            expected = 10000 / len(dist) if dist else 0
            fig_dist = px.bar(df_dist, x="Server", y="Keys",
                               color="Keys",
                               color_continuous_scale=["#0d2b50","#bc8cff"],
                               title=f"Key Distribution (10k sample keys, {vnodes} vnodes/server)")
            fig_dist.add_hline(y=expected, line_dash="dash", line_color="#8b949e",
                                annotation_text=f"Expected: {int(expected)}")
            fig_dist.update_layout(**DARK, height=260, coloraxis_showscale=False,
                                    xaxis=dict(**NO_GRID), yaxis=dict(**GRID))
            st.plotly_chart(fig_dist, use_container_width=True)

            # Lookup result
            if do_lookup and lookup_key:
                servers_rf = ring.get_n_servers(lookup_key, rf)
                st.markdown(f"""<div class="dc-p dc">
                  <div style="font-size:.82rem"><strong style="color:#bc8cff">Key:</strong> <span class="mono">{lookup_key}</span></div>
                  <div style="font-size:.82rem;margin-top:.3rem"><strong style="color:#bc8cff">Primary:</strong> {servers_rf[0] if servers_rf else "—"}</div>
                  <div style="font-size:.82rem"><strong style="color:#bc8cff">Replicas (RF={rf}):</strong> {" → ".join(servers_rf)}</div>
                </div>""", unsafe_allow_html=True)

            # Stats
            dist_vals = list(dist.values())
            if dist_vals:
                import statistics
                stdev = statistics.stdev(dist_vals) if len(dist_vals) > 1 else 0
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Servers", len(ring._servers))
                m2.metric("Ring Tokens", len(ring._sorted_keys))
                m3.metric("Std Dev", f"{stdev:.0f}")
                m4.metric("Imbalance", f"{stdev/expected*100:.1f}%" if expected else "—")


# ── RATE LIMITERS ─────────────────────────────────────────────────────────────
elif tab == "Rate Limiters":
    st.markdown("## 🚦 Rate Limiters: All Four Algorithms")
    st.markdown("""<div class="dc" style="font-size:.8rem;color:#8b949e;line-height:1.6;margin-bottom:1rem">
      <strong style="color:#3fb950">Real implementations</strong> running in the same process.
      Token Bucket (AWS API Gateway) · Sliding Window Log (Stripe) · Fixed Window (nginx) · Leaky Bucket (network QoS).
      Each request goes through all four simultaneously. See exactly where each breaks.
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_vis = st.columns([1, 3])

    with col_ctrl:
        limit     = st.slider("Rate Limit (req/sec)", 5, 100, 20, 5, key="rl_limit")
        window    = st.slider("Window (seconds)", 1, 10, 1, 1, key="rl_window")
        pattern   = st.selectbox("Traffic Pattern", ["Uniform", "Bursty", "Poisson"], key="rl_pattern")
        duration  = st.slider("Duration (seconds)", 2, 20, 5, 1, key="rl_dur")
        burst_mult = st.slider("Burst Multiplier", 1.0, 10.0, 3.0, 0.5, key="rl_burst")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_rl = st.button("▶ Run Simulation", use_container_width=True, key="run_rl")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""<div style="font-size:.72rem;color:#8b949e;line-height:1.6">
          <strong style="color:#c9d1d9">Token Bucket:</strong> burst-friendly, refills over time<br>
          <strong style="color:#c9d1d9">Sliding Window:</strong> most accurate, O(N) memory<br>
          <strong style="color:#c9d1d9">Fixed Window:</strong> simple, 2× burst at boundary<br>
          <strong style="color:#c9d1d9">Leaky Bucket:</strong> constant output rate, queues requests
        </div>""", unsafe_allow_html=True)

    with col_vis:
        if run_rl:
            with st.spinner("Simulating traffic through all 4 rate limiters..."):
                # Generate traffic
                tg = TrafficGenerator()
                if pattern == "Uniform":
                    times = tg.uniform(qps=limit * burst_mult, duration_s=duration)
                elif pattern == "Bursty":
                    times = tg.bursty(base_qps=limit * 0.5, burst_qps=limit * burst_mult,
                                       duration_s=duration, burst_every_s=2.0)
                else:
                    times = tg.poisson(lam=limit * burst_mult * 0.7, duration_s=duration)

                # Run all four limiters
                tb  = TokenBucket(capacity=limit, rate=limit / window)
                sw  = SlidingWindowLog(limit=limit, window_seconds=window)
                fw  = FixedWindowCounter(limit=limit, window_seconds=window)
                lb  = LeakyBucket(capacity=limit * 3, drain_rate=limit / window)

                results = {"t": [], "TB": [], "SW": [], "FW": [], "LB": []}
                tb_tokens = []

                start = time.monotonic()
                for req_t in times:
                    # Simulate advancing time
                    now = time.monotonic() - start
                    if req_t > now:
                        time.sleep(min(req_t - now, 0.001))

                    results["t"].append(req_t)
                    results["TB"].append(1 if tb.allow() else 0)
                    results["SW"].append(1 if sw.allow() else 0)
                    results["FW"].append(1 if fw.allow() else 0)
                    results["LB"].append(1 if lb.allow() else 0)
                    tb_tokens.append(round(tb.tokens, 2))

                df = pd.DataFrame(results)
                # Rolling acceptance rate (window=20 requests)
                W = max(1, len(df)//20)

                fig = go.Figure()
                colors = {"TB": "#3fb950","SW": "#58a6ff","FW": "#e3b341","LB": "#bc8cff"}
                labels = {"TB": "Token Bucket","SW": "Sliding Window","FW": "Fixed Window","LB": "Leaky Bucket"}

                for key in ["TB","SW","FW","LB"]:
                    rolling = df[key].rolling(W).mean() * 100
                    fig.add_trace(go.Scatter(
                        x=df["t"], y=rolling, mode="lines",
                        name=labels[key], line=dict(color=colors[key], width=2),
                    ))
                fig.update_layout(**DARK, title="Acceptance Rate (rolling %) — All 4 Algorithms",
                                   height=280,
                                   xaxis=dict(**NO_GRID, title="Time (s)"),
                                   yaxis=dict(**GRID, title="% Accepted", range=[0, 110]),
                                   legend=dict(orientation="h", y=-0.3, font=dict(size=9)))
                st.plotly_chart(fig, use_container_width=True)

                # Token bucket tokens over time
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df["t"], y=tb_tokens, mode="lines", fill="tozeroy",
                                           fillcolor="rgba(63,185,80,0.1)", line=dict(color="#3fb950",width=1.5),
                                           name="Available Tokens"))
                fig2.add_hline(y=limit, line_dash="dash", line_color="#8b949e",
                                annotation_text=f"Capacity={limit}")
                fig2.update_layout(**DARK, title="Token Bucket — Token Level Over Time", height=180,
                                    xaxis=dict(**NO_GRID, title="Time (s)"),
                                    yaxis=dict(**GRID, title="Tokens"),
                                    showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                # Summary metrics
                st.markdown("#### Results Comparison")
                mc = st.columns(4)
                for i, (key, label) in enumerate(labels.items()):
                    accepted = int(df[key].sum())
                    total    = len(df)
                    rejected = total - accepted
                    mc[i].metric(label, f"{accepted}/{total}", delta=f"-{rejected} dropped")

                # Acceptance rate comparison
                df_bar = pd.DataFrame([
                    {"Algorithm": labels[k], "Accepted %": df[k].mean() * 100}
                    for k in ["TB","SW","FW","LB"]
                ])
                fig3 = px.bar(df_bar, x="Algorithm", y="Accepted %",
                               color="Algorithm",
                               color_discrete_map={labels[k]: colors[k] for k in colors})
                fig3.update_layout(**DARK, title="Overall Acceptance Rate Comparison", height=220,
                                    xaxis=dict(**NO_GRID), yaxis=dict(**GRID, range=[0, 110]),
                                    showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Configure and click ▶ Run Simulation.")


# ── CACHE EVICTION ────────────────────────────────────────────────────────────
elif tab == "Cache Eviction":
    st.markdown("## 📦 Cache Eviction: LRU vs LFU")
    st.markdown("""<div class="dc" style="font-size:.8rem;color:#8b949e;line-height:1.6;margin-bottom:1rem">
      <strong style="color:#e3b341">LRU:</strong> O(1) get/put with OrderedDict (doubly-linked-list + hashmap). Evicts least-recently-used.
      <strong style="color:#bc8cff">LFU:</strong> O(1) get/put with frequency buckets + two hashmaps. Evicts least-frequently-used.
      Identical operations. Different eviction winners depending on access pattern.
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_vis = st.columns([1, 3])
    with col_ctrl:
        cap  = st.slider("Cache Capacity", 10, 200, 50, 10, key="ce_cap")
        ks   = st.slider("Key Space", 50, 1000, 200, 50, key="ce_ks")
        pat  = st.selectbox("Access Pattern", list(CacheSimulator.PATTERNS.keys()), key="ce_pat")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_ce = st.button("▶ Simulate", use_container_width=True, key="run_ce")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""<div style="font-size:.72rem;color:#8b949e;line-height:1.6">
          <strong style="color:#c9d1d9">80/20 Pareto:</strong> LFU wins — hot keys stay cached<br>
          <strong style="color:#c9d1d9">Sequential Scan:</strong> Both lose — cache thrashing<br>
          <strong style="color:#c9d1d9">Repeated Hotspot:</strong> LFU wins by large margin<br>
          <strong style="color:#c9d1d9">Thrashing:</strong> LRU fails — key space = cap+1
        </div>""", unsafe_allow_html=True)

    with col_vis:
        if run_ce:
            with st.spinner("Running simulation..."):
                sim = CacheSimulator()
                result = sim.run(capacity=cap, key_space=ks, pattern=pat)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(result["lru_rolling"]))),
                y=result["lru_rolling"], mode="lines",
                name="LRU", line=dict(color="#e3b341", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(result["lfu_rolling"]))),
                y=result["lfu_rolling"], mode="lines",
                name="LFU", line=dict(color="#bc8cff", width=2),
            ))
            fig.update_layout(**DARK, title=f"Rolling Hit Rate — {pat}", height=300,
                               xaxis=dict(**NO_GRID, title="Request #"),
                               yaxis=dict(**GRID, title="Hit Rate %", range=[0, 105]),
                               legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("LRU Hit Rate", f"{result['lru_hit_rate']:.1f}%")
            c2.metric("LFU Hit Rate", f"{result['lfu_hit_rate']:.1f}%")
            winner = "LFU" if result["lfu_hit_rate"] > result["lru_hit_rate"] else "LRU"
            delta  = abs(result["lfu_hit_rate"] - result["lru_hit_rate"])
            c3.metric("Winner", winner, delta=f"+{delta:.1f}%")
            c4.metric("Keys Accessed", result["keys_accessed"])

            # Why the winner wins
            explanations = {
                "80/20 Pareto": "LFU wins because the top 20% of keys accumulate high frequency and never get evicted. LRU evicts them if anything more recent comes along.",
                "Sequential Scan": "Both fail equally — sequential scan with key_space > capacity means every access is a miss (cache thrashing). No algorithm helps here.",
                "Uniform Random": "Both perform similarly — random access means no key is 'hot' and no policy has an advantage.",
                "Repeated Hotspot": "LFU dominates — the 5 hotspot keys accumulate frequency and are never evicted. LRU evicts them during cold-access periods.",
                "Thrashing (LRU bad)": "LRU fails: when key space = cap+1, every insert evicts exactly the key that will be needed next (Belady's anomaly). LFU is slightly better.",
            }
            st.markdown(f'<div class="dc-o dc" style="font-size:.8rem;color:#d29922">{explanations.get(pat,"")}</div>', unsafe_allow_html=True)
        else:
            st.info("Click ▶ Simulate to run both caches against the same access pattern.")


# ── B-TREE ────────────────────────────────────────────────────────────────────
elif tab == "B-Tree":
    st.markdown("## 🌳 B-Tree")
    st.markdown("""<div class="dc" style="font-size:.8rem;color:#8b949e;line-height:1.6;margin-bottom:1rem">
      <strong style="color:#d29922">Real B-tree</strong> with proactive splits (insert one pass downward) and merge/borrow on underflow.
      Each node holds [t-1, 2t-1] keys. Splits push median key up. This is how PostgreSQL indexes work internally.
      Minimum degree t: with t=2, each node holds 1–3 keys (classic 2-3-4 tree).
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_vis = st.columns([1, 2.5])
    with col_ctrl:
        t = st.slider("Minimum Degree (t)", 2, 5, 2, 1, key="bt_t",
                       help="Each non-root node has [t-1, 2t-1] keys. t=2 is classic 2-3-4 tree.")
        n_insert = st.slider("Keys to Insert", 1, 50, 15, 1, key="bt_n")
        manual_key = st.number_input("Insert Specific Key", 0, 999, 42, key="bt_mk")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_bt = st.button("▶ Build Tree", use_container_width=True, key="run_bt")
        st.markdown('</div>', unsafe_allow_html=True)
        insert_one = st.button("➕ Insert Key", use_container_width=True, key="bt_ins")
        bulk_rand  = st.button("🎲 Bulk Random Insert", use_container_width=True, key="bt_bulk")

        st.markdown("---")
        search_key = st.number_input("Search Key", 0, 999, 42, key="bt_search")
        do_search  = st.button("🔍 Search", use_container_width=True, key="bt_dosearch")

    if "bt_tree" not in st.session_state:
        st.session_state.bt_tree = None
        st.session_state.bt_keys = []

    if run_bt:
        keys = random.sample(range(1, 200), min(n_insert, 199))
        tree = BTree(t=t)
        for k in sorted(keys, key=lambda _: random.random()):  # random insert order
            tree.insert(k)
        st.session_state.bt_tree = tree
        st.session_state.bt_keys = keys

    if insert_one and st.session_state.bt_tree:
        st.session_state.bt_tree.insert(int(manual_key))
        st.session_state.bt_keys.append(int(manual_key))

    if bulk_rand and st.session_state.bt_tree:
        new_keys = random.sample(range(200, 500), 10)
        for k in new_keys:
            st.session_state.bt_tree.insert(k)
        st.session_state.bt_keys.extend(new_keys)

    with col_vis:
        tree = st.session_state.bt_tree
        if not tree:
            st.info("Click ▶ Build Tree to start.")
        else:
            levels = tree.to_levels()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Height", tree.height)
            m2.metric("Keys", tree.size)
            m3.metric("Nodes", tree.node_count)
            m4.metric("Max Keys/Node", 2 * t - 1)

            # Tree structure visualization
            st.markdown("#### Tree Structure")
            if levels:
                for depth, level in enumerate(levels):
                    nodes_html = ""
                    total_nodes = len(level)
                    for node_keys in level:
                        keys_str = " | ".join(str(k) for k in node_keys)
                        # Color nodes near max capacity
                        fill = len(node_keys) / (2 * t - 1)
                        if fill >= 0.85:
                            cls = "dc-r"
                            hint = "⚠️ Split on next insert"
                        elif fill >= 0.6:
                            cls = "dc-o"
                            hint = ""
                        else:
                            cls = "dc-b"
                            hint = ""
                        nodes_html += f"""<div class="{cls} dc" style="display:inline-block;margin:.2rem;padding:.3rem .6rem;min-width:60px;text-align:center">
                          <div class="mono" style="font-size:.75rem">[{keys_str}]</div>
                        </div>"""

                    st.markdown(f"""<div style="margin:.25rem 0">
                      <span style="font-size:.68rem;color:#8b949e">L{depth}: {total_nodes} node{'s' if total_nodes>1 else ''}</span>
                      <div style="overflow-x:auto;white-space:nowrap">{nodes_html}</div>
                    </div>""", unsafe_allow_html=True)

            # Key distribution
            if st.session_state.bt_keys:
                st.markdown("#### Inserted Keys")
                sorted_keys = sorted(set(st.session_state.bt_keys))
                keys_html = " ".join(
                    f'<span class="bd bd-b mono">{k}</span>' for k in sorted_keys
                )
                st.markdown(keys_html, unsafe_allow_html=True)

            # Search
            if do_search:
                node, idx = tree.search(int(search_key))
                if node:
                    st.success(f"✅ Found key {search_key} in the tree")
                else:
                    st.error(f"❌ Key {search_key} not in tree")


# ── TWO-PHASE COMMIT ──────────────────────────────────────────────────────────
elif tab == "2-Phase Commit":
    st.markdown("## 🤝 Two-Phase Commit (2PC)")
    st.markdown("""<div class="dc" style="font-size:.8rem;color:#8b949e;line-height:1.6;margin-bottom:1rem">
      <strong style="color:#5ac8fa">Real 2PC state machines</strong> for coordinator and participants.
      Phase 1: PREPARE (all must say YES). Phase 2: COMMIT or ABORT.
      The famous <strong style="color:#f85149">blocking problem</strong>: if coordinator crashes after Phase 1,
      participants are stuck with locks held, waiting forever. This is why Spanner uses Paxos within 2PC.
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_vis = st.columns([1, 2.5])
    with col_ctrl:
        participants = st.multiselect(
            "Participants",
            ["UserDB", "PaymentDB", "InventoryDB", "OrderDB", "AuditDB"],
            default=["UserDB", "PaymentDB", "InventoryDB"],
            key="tpc_parts",
        )
        abort_parts = st.multiselect(
            "Vote ABORT (simulate lock timeout)",
            participants,
            key="tpc_abort",
        )
        fail_coord = st.checkbox("💥 Crash coordinator after Phase 1", key="tpc_coord_crash",
                                  help="Simulates the blocking problem — participants stuck with locks")
        fail_parts_ids = st.multiselect("💥 Kill participants (before protocol)", [], key="tpc_fail_p")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_tpc = st.button("▶ Run Transaction", use_container_width=True, key="run_tpc")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_vis:
        if run_tpc and participants:
            tpc = TwoPhaseCommit(participants)
            abort_idx = [participants.index(p) for p in abort_parts if p in participants]
            tpc.run_full(
                abort_participants=abort_idx,
                fail_coordinator=fail_coord,
            )
            st.session_state.tpc = tpc

        tpc = st.session_state.get("tpc")
        if tpc:
            # Timeline
            st.markdown("#### Protocol Timeline")
            for ev in tpc.events:
                actor = ev["actor"]
                event = ev["event"]
                detail = ev.get("detail", "")
                t_str = f"{ev['t']*1000:.1f}ms"

                if "COMMIT" in event:      cls, ic = "dc-g", "✅"
                elif "ABORT" in event:     cls, ic = "dc-r", "❌"
                elif "CRASH" in event or "FAIL" in event:  cls, ic = "dc-r", "💥"
                elif "PREPARE" in event or "PHASE" in event: cls, ic = "dc-b", "📡"
                elif "VOTE" in event:      cls, ic = "dc-o", "🗳️"
                elif "COMPLETE" in event:  cls, ic = "dc-g", "🏁"
                else:                      cls, ic = "dc", "•"

                st.markdown(f"""<div class="{cls} dc" style="padding:.45rem .75rem;display:flex;gap:.75rem;align-items:start">
                  <span style="font-size:.7rem;color:#8b949e;white-space:nowrap">{t_str}</span>
                  <span>{ic}</span>
                  <div>
                    <span style="font-size:.78rem;font-weight:600;color:#c9d1d9">{actor}</span>
                    <span style="font-size:.75rem;color:#8b949e;margin-left:.5rem">{event}</span>
                    <div style="font-size:.72rem;color:#6e7681">{detail}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Final state
            st.markdown("#### Final Participant States")
            pc = st.columns(len(tpc.participants))
            state_cls = {
                TwoPhaseState.COMMITTED: "dc-g",
                TwoPhaseState.ABORTED:   "dc-r",
                TwoPhaseState.PREPARED:  "dc-y",
                TwoPhaseState.FAILED:    "dc-r",
                TwoPhaseState.INIT:      "dc",
            }
            for i, p in enumerate(tpc.participants):
                cls = state_cls.get(p.state, "dc")
                icon = {"COMMITTED":"✅","ABORTED":"❌","PREPARED":"⏳","FAILED":"💥","INIT":"○"}.get(p.state, "?")
                with pc[i]:
                    st.markdown(f"""<div class="{cls} dc" style="text-align:center;padding:.6rem">
                      <div>{icon}</div>
                      <div style="font-size:.72rem;font-weight:600">{p.name}</div>
                      <div style="font-size:.68rem;color:#8b949e">{p.state}</div>
                    </div>""", unsafe_allow_html=True)

            if fail_coord:
                st.error("💥 BLOCKING PROBLEM: Coordinator crashed after collecting votes. All participants hold locks and wait forever. This is why Google Spanner uses Paxos (not 2PC alone).")
        else:
            st.info("Select participants and click ▶ Run Transaction.")


# ── BLOOM FILTER ──────────────────────────────────────────────────────────────
elif tab == "Bloom Filter":
    st.markdown("## 🌸 Bloom Filter")
    st.markdown("""<div class="dc" style="font-size:.8rem;color:#8b949e;line-height:1.6;margin-bottom:1rem">
      <strong style="color:#f85149">Real implementation:</strong> k independent MD5 hash functions, m-bit array.
      Optimal m = -n·ln(p)/ln(2)². Optimal k = (m/n)·ln(2). Space-efficient membership test with tunable false positive rate.
      Used in: Cassandra (avoid disk reads for missing keys), Chrome (safe browsing), Redis (key existence).
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_vis = st.columns([1, 2.5])
    with col_ctrl:
        expected_n = st.slider("Expected Elements", 100, 100000, 10000, 100, key="bf_n")
        fp_rate    = st.select_slider("Target FP Rate", [0.001, 0.005, 0.01, 0.05, 0.10], value=0.01, key="bf_fp")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_bf = st.button("▶ Benchmark", use_container_width=True, key="run_bf")
        st.markdown('</div>', unsafe_allow_html=True)

        check_item = st.text_input("Check Membership", value="user_1234", key="bf_check")
        st.button("🔍 Check", key="bf_do_check")

    with col_vis:
        if run_bf:
            with st.spinner("Benchmarking Bloom filter..."):
                bf = BloomFilter(expected_elements=expected_n, fp_rate=fp_rate)
                result = bf.benchmark(n=min(expected_n, 10000))
                st.session_state.bf     = bf
                st.session_state.bf_res = result

        if "bf_res" in st.session_state:
            res = st.session_state.bf_res
            bf  = st.session_state.bf

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits Used (m)", f"{bf.m:,}")
            c2.metric("Hash Functions (k)", bf.k)
            c3.metric("FP Rate (actual)", f"{res['fp_rate_actual']:.3f}%")
            c4.metric("FP Rate (theory)", f"{res['fp_rate_theoretical']:.3f}%")

            m1, m2, m3 = st.columns(3)
            m1.metric("True Positives", f"{res['true_positives']:,}")
            m2.metric("False Positives", f"{res['false_positives']:,}")
            m3.metric("Fill Ratio", f"{res['fill_ratio']:.1%}")

            # Sweep: FP rate vs n_inserted
            sweep_ns = [1000, 5000, 10000, 25000, 50000, 75000, 100000]
            sweep_ns = [n for n in sweep_ns if n <= expected_n * 2]
            sweep_fp = []
            for n in sweep_ns:
                b = BloomFilter(expected_elements=expected_n, fp_rate=fp_rate)
                for i in range(n):
                    b.add(f"element_{i}")
                sweep_fp.append(b.actual_fp_rate * 100)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sweep_ns, y=sweep_fp, mode="lines+markers",
                line=dict(color="#f85149", width=2),
                marker=dict(size=6, color="#f85149"),
            ))
            fig.add_vline(x=expected_n, line_dash="dash", line_color="#8b949e",
                           annotation_text="Designed capacity")
            fig.add_hline(y=fp_rate * 100, line_dash="dash", line_color="#3fb950",
                           annotation_text=f"Target FP = {fp_rate*100}%")
            fig.update_layout(**DARK, title="FP Rate as Filter Fills Beyond Design Capacity",
                               height=280,
                               xaxis=dict(**NO_GRID, title="Elements Inserted"),
                               yaxis=dict(**GRID, title="False Positive %"))
            st.plotly_chart(fig, use_container_width=True)

            # Space savings comparison
            naive_bytes = expected_n * 8  # set of 8-byte hashes
            bloom_bytes = math.ceil(bf.m / 8)
            st.markdown(f"""<div class="dc-g dc">
              <div style="font-size:.82rem;font-weight:600;color:#3fb950">Space Savings</div>
              <div style="font-size:.8rem;color:#c9d1d9;margin-top:.3rem">
                HashSet for {expected_n:,} elements: <strong>{naive_bytes:,} bytes ({naive_bytes/1024:.1f} KB)</strong><br>
                This Bloom Filter: <strong>{bloom_bytes:,} bytes ({bloom_bytes/1024:.1f} KB)</strong><br>
                Savings: <strong>{(1 - bloom_bytes/naive_bytes)*100:.1f}%</strong> — at {fp_rate*100}% FP rate
              </div>
            </div>""", unsafe_allow_html=True)

            # Check membership
            if "bf_do_check" in st.session_state:
                item = check_item
                if item in bf:
                    st.info(f"🔵 '{item}' → **POSSIBLY IN SET** (could be FP — {res['fp_rate_actual']:.3f}% FP rate)")
                else:
                    st.success(f"✅ '{item}' → **DEFINITELY NOT IN SET** (Bloom filter guarantees no false negatives)")
        else:
            st.info("Click ▶ Benchmark to run.")


# ── QUADTREE ──────────────────────────────────────────────────────────────────
elif tab == "QuadTree Geo":
    st.markdown("## 🗺️ QuadTree Spatial Index")
    st.markdown("""<div class="dc" style="font-size:.8rem;color:#8b949e;line-height:1.6;margin-bottom:1rem">
      <strong style="color:#3fb950">Real QuadTree</strong>: recursive spatial subdivision.
      Each leaf holds ≤4 points; subdivides when full. Range query in O(log N + k) where k = results.
      This is how Uber/Lyft find nearby drivers, how Google Maps indexes POIs, how game engines do collision detection.
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_vis = st.columns([1, 2.5])
    with col_ctrl:
        n_points = st.slider("Driver/Point Count", 10, 2000, 500, 50, key="qt_n")
        cluster  = st.checkbox("Clustered Distribution (city centers)", value=True, key="qt_cluster")
        qx1 = st.slider("Query Region X start", 0.0, 0.8, 0.3, 0.05, key="qt_x1")
        qy1 = st.slider("Query Region Y start", 0.0, 0.8, 0.3, 0.05, key="qt_y1")
        qw  = st.slider("Query Width", 0.05, 0.5, 0.2, 0.05, key="qt_w")
        qh  = st.slider("Query Height", 0.05, 0.5, 0.2, 0.05, key="qt_h")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_qt = st.button("▶ Build + Query", use_container_width=True, key="run_qt")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_vis:
        if run_qt:
            with st.spinner("Building QuadTree and running spatial query..."):
                qt = QuadTree(Rect(0, 0, 1, 1), max_depth=8)
                points = []
                if cluster:
                    # Clustered around 3 city centers
                    centers = [(0.2, 0.2), (0.7, 0.7), (0.5, 0.3)]
                    for i in range(n_points):
                        cx, cy = random.choice(centers)
                        x = max(0, min(1, cx + random.gauss(0, 0.12)))
                        y = max(0, min(1, cy + random.gauss(0, 0.12)))
                        p = Point(x, y, f"driver_{i}")
                        qt.insert(p)
                        points.append(p)
                else:
                    for i in range(n_points):
                        p = Point(random.random(), random.random(), f"driver_{i}")
                        qt.insert(p)
                        points.append(p)

                # Time the query
                query_rect = Rect(qx1, qy1, qw, qh)
                t_start = time.perf_counter()
                found = qt.query(query_rect)
                t_query = (time.perf_counter() - t_start) * 1000

                # Time brute force
                t_bf_start = time.perf_counter()
                brute_found = [p for p in points if query_rect.contains(p)]
                t_bf = (time.perf_counter() - t_bf_start) * 1000

                st.session_state.qt_result = {
                    "all_points": points,
                    "found": found,
                    "query_rect": query_rect,
                    "qt_ms": t_query,
                    "bf_ms": t_bf,
                    "qt_nodes": qt.count_nodes(),
                    "qt": qt,
                }

        if "qt_result" in st.session_state:
            res = st.session_state.qt_result
            all_pts = res["all_points"]
            found_set = {(p.x, p.y) for p in res["found"]}
            qr = res["query_rect"]

            # Plot
            all_x = [p.x for p in all_pts]
            all_y = [p.y for p in all_pts]
            colors_pt = ["#f85149" if (p.x, p.y) in found_set else "#21262d" for p in all_pts]

            fig = go.Figure()
            # All points
            fig.add_trace(go.Scatter(
                x=all_x, y=all_y, mode="markers",
                marker=dict(size=4, color=colors_pt),
                name="Drivers",
            ))
            # Query rectangle
            fig.add_shape(type="rect",
                x0=qr.x, y0=qr.y, x1=qr.x+qr.w, y1=qr.y+qr.h,
                line=dict(color="#3fb950", width=2),
                fillcolor="rgba(63,185,80,0.08)",
            )
            # Found points highlight
            found_x = [p.x for p in res["found"]]
            found_y = [p.y for p in res["found"]]
            fig.add_trace(go.Scatter(
                x=found_x, y=found_y, mode="markers",
                marker=dict(size=7, color="#3fb950", symbol="circle"),
                name=f"Found ({len(found_x)})",
            ))
            fig.update_layout(**DARK, title=f"QuadTree Query — {len(res['found'])} drivers found in region",
                               height=420,
                               xaxis=dict(range=[0,1], **NO_GRID, scaleanchor="y", title="Longitude"),
                               yaxis=dict(range=[0,1], **GRID, title="Latitude"),
                               legend=dict(orientation="h", y=-0.12))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Points in Tree", len(all_pts))
            c2.metric("Drivers Found", len(res["found"]))
            c3.metric("QuadTree Query", f"{res['qt_ms']:.3f}ms")
            c4.metric("Brute Force", f"{res['bf_ms']:.3f}ms")
            c5.metric("QuadTree Nodes", res["qt_nodes"])

            speedup = res["bf_ms"] / max(res["qt_ms"], 0.0001)
            if speedup > 1.2:
                st.success(f"⚡ QuadTree is **{speedup:.1f}×** faster than brute-force scan for this query.")
            else:
                st.info("For small N, brute force may match QuadTree. The gap grows dramatically at N=100k+.")


# Catch-all
else:
    st.session_state.tab = "Home"
    st.rerun()