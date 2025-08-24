import json
import os
import time
import bcrypt
import io
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------------------------------------
# App Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="StudyTracker — GATE DA 2026",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

# -------------------------------------------------------------
# Storage paths
# -------------------------------------------------------------
DATA_DIR = ".studytracker_data"
USER_DIR = os.path.join(DATA_DIR, "users")
DRAFT_DIR = os.path.join(DATA_DIR, "drafts")
AUTH_DIR = os.path.join(DATA_DIR, "auth")
for p in (DATA_DIR, USER_DIR, DRAFT_DIR, AUTH_DIR):
    os.makedirs(p, exist_ok=True)

def _safe(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_")

def user_file(email: str) -> str:
    return os.path.join(USER_DIR, f"{_safe(email)}.json")

def draft_file(email: str) -> str:
    return os.path.join(DRAFT_DIR, f"{_safe(email)}_draft.json")

ACTIVE_USERS_FILE     = os.path.join(AUTH_DIR, "active_users.json")   # concurrent active users set
GLOBAL_REGISTRY_FILE  = os.path.join(DATA_DIR, "users_registry.json") # all users
DM_DIR                = os.path.join(DATA_DIR, "dm_threads")          # per-pair DM messages
TOKENS_FILE           = os.path.join(AUTH_DIR, "tokens.json")         # per-tab login tokens
UNREAD_FILE           = os.path.join(DATA_DIR, "unread_index.json")   # unread per-thread index
os.makedirs(DM_DIR, exist_ok=True)

# CONFIG
MAX_ACTIVE_USERS = 10  # concurrent active sessions
TOKEN_BYTES = 24       # token size for URL param

# -------------------------------------------------------------
# Syllabus + Default Plan
# -------------------------------------------------------------
SYLLABUS: Dict[str, List[str]] = {
    "Probability & Statistics": [
        "Counting (perm/comb)", "Axioms, sample space, events",
        "Independence, mutual exclusivity", "Marginal/conditional/joint prob",
        "Bayes theorem", "Cond. expectation & variance",
        "Mean/Median/Mode/SD", "Correlation & Covariance",
        "RVs: discrete/pmf, continuous/pdf, CDF",
        "Distributions: Uniform, Bernoulli, Binomial, Poisson, Exponential, Normal, Standard Normal, t, Chi-squared",
        "CLT", "Confidence intervals", "z/t/chi-squared tests"
    ],
    "Linear Algebra": [
        "Vector spaces & subspaces", "Linear dependence/independence",
        "Matrices (projection/orthogonal/idempotent/partition)",
        "Quadratic forms", "Systems & Gaussian elimination",
        "Eigenvalues/Eigenvectors", "Determinant, Rank, Nullity, Projections",
        "LU, SVD"
    ],
    "Calculus & Optimization": [
        "Single variable functions", "Limit/Continuity/Differentiability",
        "Taylor series", "Maxima/Minima", "Single-variable optimization"
    ],
    "Programming & DSA": [
        "Python basics", "Stacks/Queues/Linked Lists/Trees/Hash Tables",
        "Search: Linear/Binary", "Sorting: Selection/Bubble/Insertion",
        "Divide-Conquer: Merge/Quick", "Graph intro & Traversals/Shortest path"
    ],
    "DBMS & Warehousing": [
        "ER model", "Relational algebra & tuple calculus", "SQL",
        "Integrity constraints", "Normal forms", "File org & Indexing",
        "Data types & transformation: normalization, discretization, sampling, compression",
        "Warehouse: schema, hierarchies, measures"
    ],
    "ML — Supervised": [
        "Regression (simple/multiple/ridge)", "Logistic regression",
        "k-NN", "Naive Bayes", "LDA", "SVM", "Decision Trees",
        "Bias-variance trade-off", "Cross-Validation: LOO, k-folds",
        "MLP/Feed-forward NN"
    ],
    "ML — Unsupervised & PCA": [
        "Clustering: k-means/k-medoids", "Hierarchical (top-down/bottom-up)",
        "Single/Complete/Avg-linkage", "Dimensionality Reduction",
        "PCA"
    ],
    "AI (Search/Logic/Uncertainty)": [
        "Search: Informed/Uninformed/Adversarial",
        "Logic: Propositional & Predicate",
        "Uncertainty: Conditional independence, Variable elimination (exact), Sampling (approx.)"
    ],
    "Aptitude/Revision/PYQs": [
        "Aptitude", "Mixed PYQs (MSQ/NAT)", "Mini Projects/Case", "Formula/Notes"
    ]
}

DEFAULT_PLAN: Dict[int, List[str]] = {
    1: ["Probability & Statistics", "Programming & DSA", "ML — Supervised", "PYQs: Stats + DSA"],
    2: ["Linear Algebra", "DBMS & Warehousing", "AI (Search/Logic/Uncertainty)", "PYQs: LA + DBMS"],
    3: ["Calculus & Optimization", "Programming & DSA", "ML — Supervised", "PYQs: Calc + Algos"],
    4: ["Probability & Statistics", "Programming & DSA", "AI (Search/Logic/Uncertainty)", "PYQs: Prob + Prog"],
    5: ["Linear Algebra", "DBMS & Warehousing", "ML — Supervised", "PYQs: LA + DBMS"],
    6: ["Calculus & Optimization", "Programming & DSA", "ML — Unsupervised & PCA", "PYQs: Opt + Graphs"],
    7: ["Test: Stats", "Test: LA", "Weak Area Patch", "Formula/Notes"],
    8: ["Aptitude/Revision/PYQs", "Mixed PYQs (MSQ/NAT)", "Mini Projects/Case", "Light Revision"]
}

# -------------------------------------------------------------
# Data model
# -------------------------------------------------------------
@dataclass
class LogRow:
    Date: str
    Subject: str
    Completed: bool
    Hours: float
    Notes: str
    Priority: str
    Mode: str
    Pomodoros: int
    XP: int

@dataclass
class Settings:
    theme: str = "dark"
    dailyGoal: float = 6.0
    pomoWork: int = 25
    pomoBreak: int = 5
    startDate: str = date.today().isoformat()
    xp: int = 0
    badges: List[str] = field(default_factory=list)
    username: str = ""  # Display name

@dataclass
class UserData:
    settings: Settings
    plan: Dict[int, List[str]]
    logs: List[LogRow]
    syllabusProgress: Dict[str, bool]
    taskOrder: Dict[str, List[str]]
    dayChats: Dict[str, str] = field(default_factory=dict)  # per-day note/message

# -------------------------------------------------------------
# Auth & global helpers
# -------------------------------------------------------------
def sha_bcrypt(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_bcrypt(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False

def default_user_data() -> UserData:
    return UserData(
        settings=Settings(),
        plan=json.loads(json.dumps(DEFAULT_PLAN)),
        logs=[],
        syllabusProgress={},
        taskOrder={},
        dayChats={}
    )

def load_user(email: str) -> Tuple[Optional[UserData], Optional[str]]:
    p = user_file(email)
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        passhash = raw.get("passHash")
        d = raw.get("data", {}) or {}
        s_raw = d.get("settings") or {}
        settings_defaults = Settings()
        safe_settings = {k: s_raw.get(k, getattr(settings_defaults, k)) for k in settings_defaults.__dataclass_fields__.keys()}
        settings = Settings(**safe_settings)
        plan = d.get("plan") or {}
        plan = {int(k): v for k, v in plan.items()} if plan else json.loads(json.dumps(DEFAULT_PLAN))
        logs = []
        for r in d.get("logs", []) or []:
            if isinstance(r, dict):
                try:
                    logs.append(LogRow(**r))
                except Exception:
                    pass
        syllabus = d.get("syllabusProgress") or {}
        order = d.get("taskOrder") or {}
        day_chats = d.get("dayChats") or {}
        return UserData(settings, plan, logs, syllabus, order, day_chats), passhash
    except Exception:
        return None, None

def save_user(email: str, data: UserData, pass_hash: str):
    payload = {
        "passHash": pass_hash,
        "data": {
            "settings": asdict(data.settings),
            "plan": data.plan,
            "logs": [asdict(r) for r in data.logs],
            "syllabusProgress": data.syllabusProgress,
            "taskOrder": data.taskOrder,
            "dayChats": data.dayChats,
        }
    }
    with open(user_file(email), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

# Active users (concurrency control)
def _load_active_users() -> List[str]:
    if not os.path.exists(ACTIVE_USERS_FILE):
        return []
    try:
        with open(ACTIVE_USERS_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
        return arr if isinstance(arr, list) else []
    except Exception:
        return []

def _save_active_users(arr: List[str]):
    with open(ACTIVE_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2)

def add_active_user(email: str) -> bool:
    arr = _load_active_users()
    if email in arr:
        return True
    if len(arr) >= MAX_ACTIVE_USERS:
        return False
    arr.append(email)
    _save_active_users(arr)
    return True

def remove_active_user(email: str):
    arr = _load_active_users()
    if email in arr:
        arr.remove(email)
        _save_active_users(arr)

# Drafts
def load_draft(email: str) -> List[Dict[str, Any]]:
    fp = draft_file(email)
    if not os.path.exists(fp):
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_draft(email: str, rows: List[Dict[str, Any]]):
    with open(draft_file(email), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

# User registry
def load_registry() -> Dict[str, Any]:
    if not os.path.exists(GLOBAL_REGISTRY_FILE):
        return {"users": []}
    try:
        with open(GLOBAL_REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": []}

def save_registry(reg: Dict[str, Any]):
    with open(GLOBAL_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)

def ensure_user_in_registry(email: str, username: str = ""):
    reg = load_registry()
    users = reg.get("users", [])
    for u in users:
        if u.get("email") == email:
            if username and u.get("username") != username:
                u["username"] = username
            save_registry(reg)
            return
    users.append({"email": email, "username": username})
    reg["users"] = users
    save_registry(reg)

# Direct messages (per-pair, pruned to today)
def _dm_key(email_a: str, email_b: str) -> str:
    a, b = sorted([email_a, email_b])
    return f"{_safe(a)}__{_safe(b)}.json"

def _dm_file(email_a: str, email_b: str) -> str:
    return os.path.join(DM_DIR, _dm_key(email_a, email_b))

def load_dm(email_a: str, email_b: str) -> List[Dict[str, Any]]:
    fp = _dm_file(email_a, email_b)
    if not os.path.exists(fp):
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            msgs = json.load(f)
    except Exception:
        msgs = []
    today_str = date.today().isoformat()
    new_msgs = []
    for m in msgs:
        ts = m.get("ts", "")
        day = ts[:10] if len(ts) >= 10 else ""
        if day == today_str:
            new_msgs.append(m)
    if new_msgs != msgs:
        save_dm(email_a, email_b, new_msgs)
    return new_msgs

def save_dm(email_a: str, email_b: str, msgs: List[Dict[str, Any]]):
    fp = _dm_file(email_a, email_b)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(msgs, f, indent=2)

# Unread index for DMs: { me: { peer: "last_seen_ts" } }
def _load_unread():
    if not os.path.exists(UNREAD_FILE):
        return {}
    try:
        with open(UNREAD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_unread(d: dict):
    with open(UNREAD_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def mark_thread_seen(me: str, peer: str):
    if not me or not peer:
        return
    idx = _load_unread()
    user_map = idx.get(me, {})
    user_map[peer] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    idx[me] = user_map
    _save_unread(idx)

def has_unread(me: str, peer: str) -> bool:
    if not me or not peer:
        return False
    last = None
    idx = _load_unread()
    if me in idx and peer in idx[me]:
        last = idx[me][peer]
    msgs = load_dm(me, peer)
    if not msgs:
        return False
    if not last:
        # Never seen: any message from peer is unread
        return any(m.get("from") == peer for m in msgs)
    try:
        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
    except Exception:
        last_dt = datetime.min
    for m in msgs:
        if m.get("from") == peer:
            try:
                ts_dt = datetime.strptime(m.get("ts", ""), "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if ts_dt > last_dt:
                return True
    return False

# Per-tab login tokens (remember across refresh)
def _load_tokens():
    if not os.path.exists(TOKENS_FILE):
        return {}
    try:
        with open(TOKENS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_tokens(d: dict):
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def _gen_token(n=TOKEN_BYTES):
    import secrets
    return secrets.token_urlsafe(n)

def set_login_token(email: str, pass_hash: str):
    token = _gen_token()
    db = _load_tokens()
    db[token] = {"email": email, "passHash": pass_hash, "ts": datetime.now().isoformat()}
    _save_tokens(db)
    qp = st.query_params
    qp["token"] = token
    st.query_params = qp
    return token

def clear_login_token():
    qp = st.query_params
    if "token" in qp:
        del qp["token"]
        st.query_params = qp

def resolve_token_login():
    qp = st.query_params
    token = qp.get("token")
    if not token:
        return None, None
    db = _load_tokens()
    row = db.get(token)
    if not row:
        return None, None
    return row.get("email"), row.get("passHash")

# -------------------------------------------------------------
# Time & plan helpers
# -------------------------------------------------------------
def today_iso() -> str:
    return date.today().isoformat()

def cycle_day(settings: Settings) -> int:
    try:
        s = datetime.fromisoformat(settings.startDate).date()
    except Exception:
        s = date.today()
    diff = (date.today() - s).days
    return (diff % 8) + 1

def subjects_for_today(data: UserData) -> List[str]:
    d = cycle_day(data.settings)
    base = data.plan.get(d, [])
    order = data.taskOrder.get(f"day{d}", [])
    if order and set(order) == set(base):
        return order
    return base

def xp_from(hours: float, completed: bool, priority: str) -> int:
    base = int(round((hours or 0) * 10))
    bonus = 10 if completed else 0
    pr_bonus = {"Low": 0, "Medium": 5, "High": 10}.get(priority, 0)
    return base + bonus + pr_bonus

def streak_days(data: UserData) -> int:
    try:
        goal = float(data.settings.dailyGoal)
    except Exception:
        goal = 6.0
    by_date: Dict[str, float] = {}
    for r in data.logs:
        by_date[r.Date] = by_date.get(r.Date, 0.0) + float(r.Hours or 0.0)
    if st.session_state.get("active_email"):
        for r in st.session_state.get("draft_rows", []):
            by_date[r.get("Date", "")] = by_date.get(r.get("Date", ""), 0.0) + float(r.get("Hours") or 0.0)
    n = 0
    cur = date.today()
    for _ in range(365):
        iso = cur.isoformat()
        if by_date.get(iso, 0.0) >= goal:
            n += 1
            cur = cur - timedelta(days=1)
        else:
            break
    return n

# -------------------------------------------------------------
# Session bootstrap (with token rehydrate)
# -------------------------------------------------------------
def ensure_session_state():
    if "active_email" not in st.session_state:
        st.session_state.active_email = None
    if "pass_hash" not in st.session_state:
        st.session_state.pass_hash = None
    if "data" not in st.session_state:
        st.session_state.data = None
    if "auto_save" not in st.session_state:
        st.session_state.auto_save = True
    if "draft_rows" not in st.session_state:
        st.session_state.draft_rows = []
    if "timers" not in st.session_state:
        st.session_state.timers = {}
    if "deleted_buffer" not in st.session_state:
        st.session_state.deleted_buffer = []
    if "order_state" not in st.session_state:
        st.session_state.order_state = {}
    if "dm_peer" not in st.session_state:
        st.session_state.dm_peer = ""

    # Rehydrate from token if session is empty
    if (not st.session_state.active_email) or (not st.session_state.data):
        email, ph = resolve_token_login()
        if email and ph:
            data, onfile_ph = load_user(email)
            if data and onfile_ph == ph:
                st.session_state.active_email = email
                st.session_state.pass_hash = ph
                st.session_state.draft_rows = load_draft(email)
                st.session_state.data = data

def persist_all():
    if st.session_state.get("active_email") and st.session_state.get("pass_hash") and st.session_state.get("data"):
        save_user(st.session_state.active_email, st.session_state.data, st.session_state.pass_hash)
        save_draft(st.session_state.active_email, st.session_state.draft_rows)

# -------------------------------------------------------------
# Auth UI (per-browser session; tokens remember across refresh)
# -------------------------------------------------------------
def view_login():
    st.header("Login to StudyTracker (Local)")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login", use_container_width=True):
            if not email or not password:
                st.error("Please enter email and password")
                return
            if not add_active_user(email):
                active_list = _load_active_users()
                st.error(f"Active user limit reached ({len(active_list)}/{MAX_ACTIVE_USERS}). Try again later.")
                return

            data, ph = load_user(email)
            if data is None or ph is None:
                remove_active_user(email)
                st.error("Account not found. Please sign up.")
                return
            if not check_bcrypt(password, ph):
                remove_active_user(email)
                st.error("Incorrect password")
                return

            st.session_state.active_email = email
            st.session_state.pass_hash = ph
            st.session_state.data = data
            st.session_state.draft_rows = load_draft(email)
            ensure_user_in_registry(email, data.settings.username or "")
            set_login_token(email, ph)  # remember across refresh for this tab
            st.toast("Logged in")
            safe_rerun()
    with c2:
        if st.button("Sign Up", use_container_width=True, type="secondary"):
            if not email or not password:
                st.error("Please enter email and password")
                return

            if not add_active_user(email):
                active_list = _load_active_users()
                st.error(f"Active user limit reached ({len(active_list)}/{MAX_ACTIVE_USERS}). Try again later.")
                return

            p = user_file(email)
            if os.path.exists(p):
                remove_active_user(email)
                st.error("Account already exists. Please log in.")
                return
            phash = sha_bcrypt(password)
            data = default_user_data()
            save_user(email, data, phash)
            st.session_state.active_email = email
            st.session_state.pass_hash = phash
            st.session_state.data = data
            st.session_state.draft_rows = []
            ensure_user_in_registry(email, "")
            set_login_token(email, phash)
            st.toast("Signed up")
            safe_rerun()

# -------------------------------------------------------------
# KPI + Draft management
# -------------------------------------------------------------
def kpi_cards(data: UserData):
    try:
        logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
            "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
        ])
    except Exception:
        logs_df = pd.DataFrame(columns=["Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"])
    dr = st.session_state.get("draft_rows", [])
    draft_df = pd.DataFrame(dr) if dr else pd.DataFrame(columns=logs_df.columns)

    today = today_iso()
    today_hours = float(draft_df[draft_df.get("Date") == today]["Hours"].sum() if not draft_df.empty else 0.0)
    today_completed = int(draft_df[(draft_df.get("Date") == today) & (draft_df.get("Completed") == True)].shape[0] if not draft_df.empty else 0)

    avg_all = 0.0
    if not logs_df.empty:
        try:
            avg_all = float(logs_df.groupby("Date")["Hours"].sum().mean())
        except Exception:
            avg_all = 0.0

    avg_7 = 0.0
    if not logs_df.empty:
        last7 = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
        ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum()
        merged = pd.DataFrame({"Date": last7}).merge(ddf, on="Date", how="left").fillna({"Hours": 0})
        avg_7 = float(merged["Hours"].mean())

    display_name = (data.settings.username or st.session_state.get("active_email") or "").strip()
    goal_val = float(data.settings.dailyGoal or 6.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Goal", f"{goal_val:.2f}h")
    c2.metric(f"Today ({display_name})", f"{today_hours:.2f}h")
    c3.metric("Completed Today", str(today_completed))
    c4.metric("Streak", str(streak_days(data)))
    c5.metric("Avg Hours (All-time)", f"{avg_all:.2f}h")

    st.caption(f"Avg Hours (Last 7d): {avg_7:.2f}h")
    st.progress(min(1.0, today_hours / max(1e-6, goal_val)))

def compute_user_avgs(email: str) -> Tuple[float, float]:
    data, _ = load_user(email)
    if not data:
        return 0.0, 0.0
    try:
        logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=["Date","Hours"])
    except Exception:
        return 0.0, 0.0
    if logs_df.empty:
        return 0.0, 0.0
    try:
        avg_all = float(logs_df.groupby("Date")["Hours"].sum().mean())
    except Exception:
        avg_all = 0.0
    last7 = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum()
    merged = pd.DataFrame({"Date": last7}).merge(ddf, on="Date", how="left").fillna({"Hours": 0})
    avg_7 = float(merged["Hours"].mean()) if not merged.empty else 0.0
    return avg_all, avg_7

def upsert_draft(subject: str, hours: float, done: bool, mode: str, priority: str, notes: str, pomos: int):
    completed = bool(done and float(hours) >= 3.0)
    row = {
        "Date": today_iso(),
        "Subject": subject,
        "Completed": completed,
        "Hours": float(hours),
        "Notes": notes or "",
        "Priority": priority,
        "Mode": mode,
        "Pomodoros": int(pomos),
        "XP": int(xp_from(hours, completed, priority))
    }
    rows = st.session_state.get("draft_rows", [])
    found = False
    for r in rows:
        if r.get("Subject") == subject and r.get("Date") == row["Date"]:
            r.update(row)
            found = True
            break
    if not found:
        rows.append(row)
        st.session_state.draft_rows = rows
    if st.session_state.get("auto_save", True):
        persist_all()

def clear_draft_today():
    today = today_iso()
    rows = st.session_state.get("draft_rows", [])
    rows = [r for r in rows if r.get("Date") != today]
    st.session_state.draft_rows = rows
    if st.session_state.get("auto_save", True):
        persist_all()

def save_log(data: UserData):
    today = today_iso()
    rows = st.session_state.get("draft_rows", [])
    todays = [r for r in rows if r.get("Date") == today and (r.get("Hours", 0) > 0 or r.get("Notes"))]
    if not todays:
        st.toast("No data to save")
        return
    for r in todays:
        try:
            data.logs.append(LogRow(**r))
            data.settings.xp += int(r.get("XP", 0))
        except Exception:
            pass
    st.session_state.draft_rows = [r for r in rows if r.get("Date") != today]
    total_hours = sum((float(l.Hours or 0.0) for l in data.logs), 0.0)
    badges = set(data.settings.badges or [])
    for threshold, name in [(10, "Rookie"), (50, "Committed"), (100, "Centurion"), (200, "Marathoner")]:
        if total_hours >= threshold:
            badges.add(name)
    data.settings.badges = sorted(badges)
    persist_all()
    st.toast("Log saved")

# -------------------------------------------------------------
# Today View
# -------------------------------------------------------------
def timer_ui(subject: str, data: UserData):
    tmap = st.session_state.get("timers", {})
    if subject not in tmap:
        tmap[subject] = {"running": False, "start": 0.0, "elapsed_ms": 0, "pomos": 0, "use_pomo": False}
        st.session_state.timers = tmap
    t = tmap[subject]

    def hms(ms: int) -> str:
        s = ms // 1000
        h = s // 3600
        s -= h*3600
        m = s // 60
        s -= m*60
        return f"{h:02d}:{m:02d}:{s:02d}"

    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    c1.write(f"⏱️ {hms(int(t['elapsed_ms']))}")
    start = c2.button("Start", key=f"start_{subject}")
    pause = c3.button("Pause", key=f"pause_{subject}")
    stop = c4.button("Stop", key=f"stop_{subject}")
    t["use_pomo"] = c5.toggle("Pomodoro", value=t.get("use_pomo", False), key=f"pomo_{subject}")

    now = time.time()
    if t["running"]:
        t["elapsed_ms"] = int((now - t["start"]) * 1000)

    if start and not t["running"]:
        t["running"] = True
        t["start"] = now - (t["elapsed_ms"] / 1000.0)
        st.toast(f"Timer started: {subject}")

    if pause and t["running"]:
        t["running"] = False
        t["elapsed_ms"] = int((now - t["start"]) * 1000)
        st.toast(f"Timer paused: {subject}")

    if t["use_pomo"]:
        work = (data.settings.pomoWork or 25) * 60 * 1000
        brk = (data.settings.pomoBreak or 5) * 60 * 1000
        cyc = work + brk
        completed = t["elapsed_ms"] // cyc
        if completed > t["pomos"]:
            t["pomos"] = int(completed)
            st.toast(f"Pomodoro complete: {subject}")
    st.caption(f"🍅 {t['pomos']}")

    if stop and (t["running"] or t["elapsed_ms"] > 0):
        if t["running"]:
            t["elapsed_ms"] = int((now - t["start"]) * 1000)
        t["running"] = False
        st.toast(f"Timer stopped: {subject}")

    hrs = min(10.0, round(t["elapsed_ms"] / 3600000.0, 2))
    return hrs, t["pomos"]

def reorder_ui(data: UserData):
    d = cycle_day(data.settings)
    st.subheader(f"Today's Subjects — Day {d} of 8")
    subjects = data.plan.get(d, [])
    key = f"day{d}"
    current = data.taskOrder.get(key, subjects[:])

    st.caption("Reorder with arrow buttons and Save.")
    arr = st.session_state["order_state"].setdefault(key, current[:])

    def move(i, di):
        j = i + di
        if 0 <= j < len(arr):
            arr[i], arr[j] = arr[j], arr[i]

    for i, s in enumerate(arr):
        c1, c2, c3 = st.columns([8,1,1])
        c1.write(f"- {s}")
        if c2.button("↑", key=f"up_{key}_{i}"):
            move(i, -1)
            safe_rerun()
        if c3.button("↓", key=f"dn_{key}_{i}"):
            move(i, 1)
            safe_rerun()

    if st.button("Save Order", key=f"save_order_{key}", type="primary"):
        data.taskOrder[key] = arr[:]
        st.toast("Order saved")
        if st.session_state.get("auto_save", True):
            persist_all()

def today_view(data: Optional[UserData]):
    if not data or not getattr(data, "settings", None):
        st.warning("No user data loaded. Please log in again.")
        return

    st.header("Today's Plan")
    st.caption("Hours up to 10h/subject. Completed only if hours ≥ 3h.")
    kpi_cards(data)

    a1, a2, a3, a4, a5 = st.columns(5)
    if a1.button("💾 Save Log", type="primary", use_container_width=True):
        save_log(data)
        safe_rerun()
    if a2.button("➡ Next Day", use_container_width=True):
        try:
            s = datetime.fromisoformat(data.settings.startDate).date()
        except Exception:
            s = date.today()
        data.settings.startDate = (s - timedelta(days=1)).isoformat()
        if st.session_state.get("auto_save", True):
            persist_all()
        safe_rerun()
    if a3.button("🔄 Refresh", use_container_width=True):
        safe_rerun()
    if a4.button("🧹 Clear Today Draft", use_container_width=True):
        clear_draft_today()
        safe_rerun()
    st.session_state.auto_save = a5.toggle("Auto-save on Change", value=st.session_state.get("auto_save", True))

    st.divider()

    st.subheader("Daily Note")
    tdy = today_iso()
    msg = data.dayChats.get(tdy, "")
    new_msg = st.text_area("Today's note", value=msg, height=80, placeholder="How did study go? Wins, blockers, next steps...")
    cmsg1, cmsg2 = st.columns(2)
    if cmsg1.button("Save Today Note"):
        data.dayChats[tdy] = new_msg.strip()
        st.toast("Today's note saved")
        if st.session_state.get("auto_save", True):
            persist_all()
    if cmsg2.button("Delete Today Note", type="secondary"):
        if tdy in data.dayChats:
            del data.dayChats[tdy]
            st.toast("Today's note deleted")
            if st.session_state.get("auto_save", True):
                persist_all()

    st.divider()
    reorder_ui(data)
    st.divider()

    for subject in subjects_for_today(data):
        with st.container(border=True):
            st.subheader(subject)
            st.caption("Enter hours and details. Completed requires ≥ 3h.")
            sugg, pomos = timer_ui(subject, data)

            c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,1.2])
            hours = c1.number_input("Hours", min_value=0.0, max_value=10.0, step=0.25, value=float(sugg), key=f"hrs_{subject}")
            mode = c2.selectbox("Mode", ["Deep Work","Review","Practice","PYQs","Test"], key=f"mode_{subject}")
            priority = c3.selectbox("Priority", ["Low","Medium","High"], index=1, key=f"prio_{subject}")
            done_checkbox = c4.checkbox("Completed (≥3h)", key=f"done_{subject}")

            if float(hours) < 3.0 and done_checkbox:
                st.warning("Completion requires ≥ 3.0 hours. It will be saved as not completed.")

            notes = st.text_area("Notes", value="", height=80, key=f"notes_{subject}", placeholder="Concepts, mistakes, formulas, tasks...")

            upsert_draft(subject, hours, done_checkbox, mode, priority, notes, pomos)

# -------------------------------------------------------------
# Syllabus View
# -------------------------------------------------------------
def syllabus_view(data: Optional[UserData]):
    if not data or not getattr(data, "settings", None):
        st.warning("No user data loaded. Please log in again.")
        return

    st.header("DA Syllabus Map")
    for cat, items in SYLLABUS.items():
        with st.expander(cat, expanded=False):
            comp = sum(1 for t in items if data.syllabusProgress.get(f"{cat}:{t}", False))
            st.caption(f"{comp}/{len(items)} completed")
            for t in items:
                k = f"{cat}:{t}"
                val = st.checkbox(t, value=data.syllabusProgress.get(k, False), key=f"sy_{k}")
                data.syllabusProgress[k] = bool(val)
    if st.session_state.get("auto_save", True):
        persist_all()

# -------------------------------------------------------------
# Logs View
# -------------------------------------------------------------
def logs_view(data: Optional[UserData]):
    if not data or not getattr(data, "settings", None):
        st.warning("No user data loaded. Please log in again.")
        return

    st.header("Logs")
    try:
        logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
            "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
        ])
    except Exception:
        logs_df = pd.DataFrame(columns=["Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"])

    if logs_df.empty:
        st.info("No logs yet.")
    else:
        st.dataframe(logs_df, use_container_width=True, height=420)

    st.subheader("Manage Rows")
    del_idx = st.number_input("Row index to delete", min_value=0, step=1, value=0 if not logs_df.empty else 0, disabled=logs_df.empty)
    c1, c2, c3 = st.columns(3)
    if c1.button("Delete Row", type="secondary", disabled=logs_df.empty):
        if 0 <= del_idx < len(data.logs):
            removed = data.logs.pop(int(del_idx))
            st.session_state.deleted_buffer.append(removed)
            st.toast(f"Deleted row {del_idx}")
            if st.session_state.get("auto_save", True):
                persist_all()
            safe_rerun()
    if c2.button("Undo Last Delete", disabled=(len(st.session_state.deleted_buffer)==0)):
        rec = st.session_state.deleted_buffer.pop()
        data.logs.append(rec)
        st.toast("Undo successful")
        if st.session_state.get("auto_save", True):
            persist_all()
        safe_rerun()
    if c3.button("🗑 Clear All Logs", type="secondary", disabled=logs_df.empty):
        data.logs = []
        st.session_state.deleted_buffer = []
        st.toast("Logs cleared")
        if st.session_state.get("auto_save", True):
            persist_all()
        safe_rerun()

    st.subheader("Import/Export")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        if st.button("⬇ Export CSV", use_container_width=True):
            csv = logs_df.to_csv(index=False)
            st.download_button("Download study_log.csv", data=csv, file_name="study_log.csv", mime="text/csv", use_container_width=True)
    with e2:
        if st.button("💾 Save JSON (Full Backup)", use_container_width=True):
            payload = {
                "email": st.session_state.get("active_email"),
                "data": {
                    "settings": asdict(st.session_state.data.settings),
                    "plan": st.session_state.data.plan,
                    "logs": [asdict(r) for r in st.session_state.data.logs],
                    "syllabusProgress": st.session_state.data.syllabusProgress,
                    "taskOrder": st.session_state.data.taskOrder,
                    "dayChats": st.session_state.data.dayChats,
                }
            }
            st.download_button("Download data.json", data=json.dumps(payload, indent=2),
                               file_name="data.json", mime="application/json", use_container_width=True)
    with e3:
        if st.button("💾 Save Excel (Logs Only)", use_container_width=True):
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                (logs_df if not logs_df.empty else pd.DataFrame(columns=[
                    "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
                ])).to_excel(writer, index=False, sheet_name="Logs")
            st.download_button("Download study_log.xlsx", data=out.getvalue(),
                               file_name="study_log.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
    with e4:
        if st.button("💾 Save Excel (Full Export)", use_container_width=True):
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                (logs_df if not logs_df.empty else pd.DataFrame(columns=[
                    "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
                ])).to_excel(writer, index=False, sheet_name="Logs")
                profile_df = pd.DataFrame([{
                    "Email": st.session_state.get("active_email") or "",
                    "Username": st.session_state.data.settings.username or "",
                    "DailyGoal": st.session_state.data.settings.dailyGoal,
                    "PomoWork": st.session_state.data.settings.pomoWork,
                    "PomoBreak": st.session_state.data.settings.pomoBreak,
                    "StartDate": st.session_state.data.settings.startDate,
                    "XP": st.session_state.data.settings.xp,
                    "Badges": ", ".join(st.session_state.data.settings.badges or []),
                }])
                profile_df.to_excel(writer, index=False, sheet_name="Profile")
                if st.session_state.data.dayChats:
                    day_notes_df = pd.DataFrame([{"Date": k, "Note": v} for k, v in sorted(st.session_state.data.dayChats.items())])
                else:
                    day_notes_df = pd.DataFrame(columns=["Date","Note"])
                day_notes_df.to_excel(writer, index=False, sheet_name="DayNotes")
            st.download_button("Download study_export.xlsx", data=out.getvalue(),
                               file_name="study_export.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
    with e5:
        uploaded = st.file_uploader("⬆ Import data.json", type=["json"], label_visibility="collapsed")
        if uploaded is not None and st.button("Import", use_container_width=True):
            try:
                parsed = json.loads(uploaded.read().decode("utf-8"))
                if "data" not in parsed:
                    st.error("Invalid data.json format")
                else:
                    d = parsed["data"]
                    settings = Settings(**(d.get("settings") or {}))
                    plan = {int(k): v for k, v in (d.get("plan") or {}).items()} if d.get("plan") else json.loads(json.dumps(DEFAULT_PLAN))
                    logs = [LogRow(**r) for r in (d.get("logs") or []) if isinstance(r, dict)]
                    syllabus = d.get("syllabusProgress") or {}
                    order = d.get("taskOrder") or {}
                    day_chats = d.get("dayChats") or {}
                    st.session_state.data = UserData(settings, plan, logs, syllabus, order, day_chats)
                    st.toast("Imported data.json")
                    persist_all()
                    safe_rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

# -------------------------------------------------------------
# Dashboard View
# -------------------------------------------------------------
def dashboard_view(data: Optional[UserData]):
    if not data or not getattr(data, "settings", None):
        st.warning("No user data loaded. Please log in again.")
        return

    st.header("Dashboard")
    try:
        logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
            "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
        ])
    except Exception:
        logs_df = pd.DataFrame(columns=["Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"])

    cA, cB = st.columns(2)
    with cA:
        st.subheader("Daily Hours")
        if not logs_df.empty:
            ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum().sort_values("Date")
            chart = alt.Chart(ddf).mark_line(point=True).encode(
                x="Date:T",
                y=alt.Y("Hours:Q", scale=alt.Scale(domain=[0, max(3.0, float(ddf['Hours'].max()))])),
                tooltip=["Date","Hours"]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data")
    with cB:
        st.subheader("Hours by Subject")
        if not logs_df.empty:
            sdf = logs_df.groupby("Subject", as_index=False)["Hours"].sum().sort_values("Hours", ascending=False)
            chart = alt.Chart(sdf).mark_bar().encode(
                x="Hours:Q",
                y=alt.Y("Subject:N", sort="-x"),
                tooltip=["Subject","Hours"]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data")

    cC, cD = st.columns(2)
    with cC:
        st.subheader("Completion Share")
        if not logs_df.empty:
            total = len(logs_df)
            comp = int((logs_df["Completed"]==True).sum())
            pie_df = pd.DataFrame({"Status":["Completed","Pending"],"Count":[comp, max(0,total-comp)]})
            pie = alt.Chart(pie_df).mark_arc().encode(
                theta="Count:Q", color="Status:N", tooltip=["Status","Count"]
            ).properties(height=260)
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("No data")

    with cD:
        st.subheader("Last 7 Days")
        if not logs_df.empty:
            last7 = [(date.today()-timedelta(days=i)).isoformat() for i in range(6,-1,-1)]
            ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum()
            merged = pd.DataFrame({"Date": last7}).merge(ddf, on="Date", how="left").fillna({"Hours":0})
            chart = alt.Chart(merged).mark_bar().encode(
                x=alt.X("Date:T", sort=None),
                y=alt.Y("Hours:Q", scale=alt.Scale(domain=[0, max(3.0, float(merged['Hours'].max()))])),
                tooltip=["Date","Hours"]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data")

# -------------------------------------------------------------
# Conversation View (private DMs, unread badges)
# -------------------------------------------------------------
def conversation_view():
    st.header("Conversation (Private)")

    current_email = st.session_state.get("active_email") or ""
    reg = load_registry()
    users = reg.get("users", [])

    st.subheader("Users")
    if not users:
        st.info("No registered users yet.")
    else:
        for u in users:
            peer_email = u.get("email") or ""
            peer_name = (u.get("username") or peer_email).strip()
            avg_all, avg_7 = compute_user_avgs(peer_email)

            new_badge = " NEW" if current_email and peer_email and peer_email != current_email and has_unread(current_email, peer_email) else ""
            cols = st.columns([5,2,2,1.5])
            cols[0].markdown(f"{peer_name} ({peer_email}){new_badge}")
            cols[1].markdown(f"Avg(all): {avg_all:.2f}h")
            cols[2].markdown(f"Avg(7d): {avg_7:.2f}h")
            if peer_email and peer_email != current_email:
                if cols[3].button("Chat", key=f"chat_{peer_email}"):
                    st.session_state.dm_peer = peer_email
                    mark_thread_seen(current_email, peer_email)
                    safe_rerun()
            else:
                cols[3].markdown("—")

    peer = st.session_state.get("dm_peer") or ""
    st.divider()
    st.subheader("Direct Message")
    if not peer:
        st.info("Select a user to start chatting.")
        return

    me = current_email
    msgs = load_dm(me, peer)
    # Viewing marks the thread as seen for me
    mark_thread_seen(me, peer)

    if msgs:
        for m in sorted(msgs, key=lambda x: x.get("ts", ""), reverse=True):
            who = m.get("username") or m.get("from") or m.get("email") or "Unknown"
            ts = m.get("ts", "")
            txt = m.get("text", "")
            st.markdown(f"• [{ts}] {who}: {txt}")
    else:
        st.info("No messages yet today with this user. Say hi!")

    my_username = ""
    try:
        my_username = st.session_state.data.settings.username or ""
    except Exception:
        my_username = ""
    st.write("")
    msg_text = st.text_area("Message", value="", height=80, placeholder=f"Message to {peer}...")
    c1, c2, c3 = st.columns(3)
    if c1.button("Send"):
        txt = msg_text.strip()
        if not txt:
            st.warning("Message is empty.")
        else:
            msgs = load_dm(me, peer)
            msgs.append({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from": me,
                "to": peer,
                "username": my_username,
                "text": txt
            })
            save_dm(me, peer, msgs)
            # Sending implies seen for me
            mark_thread_seen(me, peer)
            st.success("Message sent.")
            safe_rerun()
    if c2.button("Clear today's thread", type="secondary"):
        save_dm(me, peer, [])
        st.success("Cleared today's messages in this thread.")
        safe_rerun()
    if c3.button("Back to users"):
        st.session_state.dm_peer = ""
        safe_rerun()

# -------------------------------------------------------------
# Settings View
# -------------------------------------------------------------
def plan_editor_view(data: UserData):
    st.subheader("8-Day Plan Editor")
    for d in range(1, 9):
        with st.expander(f"Day {d}", expanded=(d == cycle_day(data.settings))):
            txt = st.text_area(
                f"Subjects (one per line) — Day {d}",
                value="\n".join(data.plan.get(d, [])),
                height=120,
                key=f"plan_day_{d}"
            )
            if st.button(f"Save Day {d}", key=f"s_day_{d}"):
                arr = [x.strip() for x in txt.split("\n") if x.strip()]
                data.plan[d] = arr
                st.toast(f"Saved Day {d}")
                if st.session_state.get("auto_save", True):
                    persist_all()

def settings_view(data: Optional[UserData]):
    if not data or not getattr(data, "settings", None):
        st.warning("No user data loaded. Please log in again.")
        return

    st.header("Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("General")
        uname = st.text_input("Username (display name)", value=data.settings.username or "")
        g1, g2 = st.columns(2)
        daily_goal = g1.number_input("Daily Goal (hrs)", min_value=0.0, step=0.5, value=float(data.settings.dailyGoal))
        theme = g2.selectbox("Theme", ["dark","light"], index=(0 if data.settings.theme=="dark" else 1))
        p1, p2 = st.columns(2)
        pomo_work = p1.number_input("Pomodoro Work (min)", min_value=1, step=1, value=int(data.settings.pomoWork))
        pomo_break = p2.number_input("Pomodoro Break (min)", min_value=1, step=1, value=int(data.settings.pomoBreak))

        s1, s2, s3 = st.columns(3)
        if s1.button("Save", type="primary"):
            data.settings.username = uname.strip()
            data.settings.dailyGoal = float(daily_goal)
            data.settings.theme = theme
            data.settings.pomoWork = int(pomo_work)
            data.settings.pomoBreak = int(pomo_break)
            ensure_user_in_registry(st.session_state.get("active_email") or "", data.settings.username or "")
            st.toast("Settings saved")
            if st.session_state.get("auto_save", True):
                persist_all()

        if s2.button("Export Config+Plan"):
            payload = {"settings": asdict(data.settings), "plan": data.plan}
            st.download_button("Download config_plan.json", data=json.dumps(payload, indent=2),
                               file_name="config_plan.json", mime="application/json")

        if s3.button("Reset 8-Day Plan", type="secondary"):
            data.plan = json.loads(json.dumps(DEFAULT_PLAN))
            st.toast("Plan reset")
            if st.session_state.get("auto_save", True):
                persist_all()

    with col2:
        plan_editor_view(data)

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
def header_bar(data: Optional[UserData]):
    if not data or not isinstance(data, UserData) or not getattr(data, "settings", None):
        display_name = (st.session_state.get("active_email") or "").strip()
    else:
        try:
            display_name = (data.settings.username or st.session_state.get("active_email") or "").strip()
        except Exception:
            display_name = (st.session_state.get("active_email") or "").strip()

    left, right = st.columns([3,2])
    with left:
        st.markdown("### StudyTracker · Data Science & AI")
        st.caption("GATE DA 2026")
    with right:
        st.write(f"👤 {display_name}")
        st.write(f"📅 {today_iso()}")
        hb1, hb2, hb3 = st.columns(3)
        if hb1.button("Logout", use_container_width=True):
            persist_all()
            if st.session_state.get("active_email"):
                remove_active_user(st.session_state.get("active_email"))
            clear_login_token()  # remove token so refresh won't auto-login
            st.session_state.active_email = None
            st.session_state.pass_hash = None
            st.session_state.data = None
            st.session_state.draft_rows = []
            st.session_state.timers = {}
            st.session_state.deleted_buffer = []
            st.session_state.dm_peer = ""
            st.toast("Logged out")
            safe_rerun()
        if data and getattr(data, "settings", None):
            if hb2.button("🌗 Theme", use_container_width=True):
                data.settings.theme = "light" if data.settings.theme == "dark" else "dark"
                st.toast("Theme switched")
                if st.session_state.get("auto_save", True):
                    persist_all()
        if hb3.button("🎯 Focus", use_container_width=True):
            st.sidebar.write("Focus mode toggled. Collapse sidebar if desired.")
            st.toast("Focus toggled")
    st.divider()

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    ensure_session_state()

    if not st.session_state.get("active_email") or not st.session_state.get("data") or not getattr(st.session_state.get("data"), "settings", None):
        view_login()
        return

    data: UserData = st.session_state.data
    header_bar(data)

    t1, t2, t3, t4, t5, t6 = st.tabs(["Today", "Syllabus", "Log", "Dashboard", "Conversation", "Settings"])
    with t1:
        today_view(data)
    with t2:
        syllabus_view(data)
    with t3:
        logs_view(data)
    with t4:
        dashboard_view(data)
    with t5:
        conversation_view()
    with t6:
        settings_view(data)

    if st.session_state.get("auto_save", True):
        persist_all()

if __name__ == "__main__":
    main()
