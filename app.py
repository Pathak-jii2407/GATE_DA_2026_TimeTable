# # # app.py

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

AUTH_SESSION_FILE = os.path.join(AUTH_DIR, "session.json")
ACTIVE_LOCK_FILE   = os.path.join(AUTH_DIR, "active_user.json")  # single-session lock
GLOBAL_REGISTRY_FILE = os.path.join(DATA_DIR, "users_registry.json")  # all users
GLOBAL_MSGS_FILE     = os.path.join(DATA_DIR, "conversation.json")    # global conversation

# Set this to True to allow only one total session (even same user)
STRICT_SINGLE_SESSION = False  # if True, even same email cannot start a second session

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

def save_session(email: Optional[str], pass_hash: Optional[str]):
    with open(AUTH_SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump({"email": email, "passHash": pass_hash}, f)

def load_session() -> Tuple[Optional[str], Optional[str]]:
    if not os.path.exists(AUTH_SESSION_FILE):
        return None, None
    try:
        with open(AUTH_SESSION_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        return s.get("email"), s.get("passHash")
    except Exception:
        return None, None

def get_active_user_lock() -> Optional[str]:
    if not os.path.exists(ACTIVE_LOCK_FILE):
        return None
    try:
        with open(ACTIVE_LOCK_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d.get("email")
    except Exception:
        return None

def set_active_user_lock(email: Optional[str]):
    with open(ACTIVE_LOCK_FILE, "w", encoding="utf-8") as f:
        json.dump({"email": email}, f)

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

def load_conversation() -> List[Dict[str, Any]]:
    # Also prune messages to keep only "today" to limit storage
    if not os.path.exists(GLOBAL_MSGS_FILE):
        return []
    try:
        with open(GLOBAL_MSGS_FILE, "r", encoding="utf-8") as f:
            msgs = json.load(f)
    except Exception:
        msgs = []
    # Prune: keep only today's messages
    today_str = date.today().isoformat()
    new_msgs = []
    for m in msgs:
        # Expect ts "YYYY-MM-DD HH:MM:SS"
        ts = m.get("ts", "")
        day = ts[:10] if len(ts) >= 10 else ""
        if day == today_str:
            new_msgs.append(m)
    if new_msgs != msgs:
        save_conversation(new_msgs)
    return new_msgs

def save_conversation(msgs: List[Dict[str, Any]]):
    with open(GLOBAL_MSGS_FILE, "w", encoding="utf-8") as f:
        json.dump(msgs, f, indent=2)

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
# Session bootstrap
# -------------------------------------------------------------
def ensure_session_state():
    if "active_email" not in st.session_state:
        email, ph = load_session()
        st.session_state.active_email = email
        st.session_state.pass_hash = ph
        st.session_state.data = None
        if email and ph:
            data, onfile_ph = load_user(email)
            if data and onfile_ph == ph:
                st.session_state.data = data
            else:
                save_session(None, None)
                st.session_state.active_email = None
                st.session_state.pass_hash = None
                st.session_state.data = None
    if "auto_save" not in st.session_state:
        st.session_state.auto_save = True
    if "draft_rows" not in st.session_state:
        if st.session_state.active_email:
            st.session_state.draft_rows = load_draft(st.session_state.active_email)
        else:
            st.session_state.draft_rows = []
    if "timers" not in st.session_state:
        st.session_state.timers = {}
    if "deleted_buffer" not in st.session_state:
        st.session_state.deleted_buffer = []
    if "order_state" not in st.session_state:
        st.session_state.order_state = {}

def persist_all():
    if st.session_state.get("active_email") and st.session_state.get("pass_hash") and st.session_state.get("data"):
        save_user(st.session_state.active_email, st.session_state.data, st.session_state.pass_hash)
        save_draft(st.session_state.active_email, st.session_state.draft_rows)

# -------------------------------------------------------------
# Auth UI
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

            # single-session restriction
            locked = get_active_user_lock()
            if locked:
                if STRICT_SINGLE_SESSION or locked != email:
                    st.error(f"Another user ({locked}) is currently active. Please try again later.")
                    return

            data, ph = load_user(email)
            if data is None or ph is None:
                st.error("Account not found. Please sign up.")
                return
            if not check_bcrypt(password, ph):
                st.error("Incorrect password")
                return

            st.session_state.active_email = email
            st.session_state.pass_hash = ph
            st.session_state.data = data
            save_session(email, ph)
            st.session_state.draft_rows = load_draft(email)
            ensure_user_in_registry(email, data.settings.username or "")
            set_active_user_lock(email)
            st.toast("Logged in")
            st.rerun()
    with c2:
        if st.button("Sign Up", use_container_width=True, type="secondary"):
            if not email or not password:
                st.error("Please enter email and password")
                return

            # single-session restriction (even during signup)
            locked = get_active_user_lock()
            if locked and STRICT_SINGLE_SESSION:
                st.error(f"User ({locked}) is currently active. Please try again later.")
                return

            p = user_file(email)
            if os.path.exists(p):
                st.error("Account already exists. Please log in.")
                return
            phash = sha_bcrypt(password)
            data = default_user_data()
            save_user(email, data, phash)
            save_session(email, phash)
            st.session_state.active_email = email
            st.session_state.pass_hash = phash
            st.session_state.data = data
            st.session_state.draft_rows = []
            ensure_user_in_registry(email, "")
            set_active_user_lock(email)
            st.toast("Signed up")
            st.rerun()

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
            st.rerun()
        if c3.button("↓", key=f"dn_{key}_{i}"):
            move(i, 1)
            st.rerun()

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
        st.rerun()
    if a2.button("➡ Next Day", use_container_width=True):
        try:
            s = datetime.fromisoformat(data.settings.startDate).date()
        except Exception:
            s = date.today()
        data.settings.startDate = (s - timedelta(days=1)).isoformat()
        if st.session_state.get("auto_save", True):
            persist_all()
        st.rerun()
    if a3.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    if a4.button("🧹 Clear Today Draft", use_container_width=True):
        clear_draft_today()
        st.rerun()
    st.session_state.auto_save = a5.toggle("Auto-save on Change", value=st.session_state.get("auto_save", True))

    st.divider()

    # Daily message
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
# Logs View (with per-row delete + undo)
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
            st.rerun()
    if c2.button("Undo Last Delete", disabled=(len(st.session_state.deleted_buffer)==0)):
        rec = st.session_state.deleted_buffer.pop()
        data.logs.append(rec)
        st.toast("Undo successful")
        if st.session_state.get("auto_save", True):
            persist_all()
        st.rerun()
    if c3.button("🗑 Clear All Logs", type="secondary", disabled=logs_df.empty):
        data.logs = []
        st.session_state.deleted_buffer = []
        st.toast("Logs cleared")
        if st.session_state.get("auto_save", True):
            persist_all()
        st.rerun()

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
                # Logs sheet
                (logs_df if not logs_df.empty else pd.DataFrame(columns=[
                    "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
                ])).to_excel(writer, index=False, sheet_name="Logs")
                # Profile sheet
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
                # Day notes
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
                    st.rerun()
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
# Conversation View (global users + per-day messages that auto-delete old days)
# -------------------------------------------------------------
def conversation_view():
    st.header("Conversation")

    # List all users
    reg = load_registry()
    users = reg.get("users", [])
    if not users:
        st.info("No registered users yet.")
    else:
        st.subheader("All Users")
        for u in users:
            disp = (u.get("username") or u.get("email") or "").strip()
            st.write(f"- {disp}")

    # Messages (auto-pruned on load to only today's)
    st.subheader("Messages (today)")
    msgs = load_conversation()
    if msgs:
        for m in sorted(msgs, key=lambda x: x.get("ts", ""), reverse=True):
            who = m.get("username") or m.get("email") or "Unknown"
            ts = m.get("ts", "")
            txt = m.get("text", "")
            st.markdown(f"• [{ts}] {who}: {txt}")
    else:
        st.info("No messages yet today. Post one below.")

    st.divider()
    st.subheader("Post a message")
    current_email = st.session_state.get("active_email") or ""
    current_username = ""
    try:
        current_username = st.session_state.data.settings.username or ""
    except Exception:
        pass
    msg_text = st.text_area("Message", value="", height=80, placeholder="Share an update or ask a question...")
    c1, c2 = st.columns(2)
    if c1.button("Send"):
        txt = msg_text.strip()
        if not txt:
            st.warning("Message is empty.")
        else:
            msgs = load_conversation()  # loads and prunes old automatically
            msgs.append({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "email": current_email,
                "username": current_username,
                "text": txt
            })
            save_conversation(msgs)
            st.success("Message sent.")
            st.experimental_rerun()
    if c2.button("Clear today's messages", type="secondary"):
        # simply save empty list (for today)
        save_conversation([])
        st.success("Today's messages cleared.")
        st.experimental_rerun()

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
            save_session(None, None)
            set_active_user_lock(None)  # release single-session lock
            st.session_state.active_email = None
            st.session_state.pass_hash = None
            st.session_state.data = None
            st.session_state.draft_rows = []
            st.session_state.timers = {}
            st.session_state.deleted_buffer = []
            st.toast("Logged out")
            st.rerun()
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






# import json
# import os
# import time
# import bcrypt
# import io
# from datetime import datetime, timedelta, date
# from typing import Dict, List, Optional, Any, Tuple
# from dataclasses import dataclass, asdict, field

# import streamlit as st
# import pandas as pd
# import numpy as np
# import altair as alt

# # -------------------------------------------------------------
# # App Config
# # -------------------------------------------------------------
# st.set_page_config(
#     page_title="StudyTracker — GATE DA 2026",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # -------------------------------------------------------------
# # Storage paths
# # -------------------------------------------------------------
# DATA_DIR = ".studytracker_data"
# USER_DIR = os.path.join(DATA_DIR, "users")
# DRAFT_DIR = os.path.join(DATA_DIR, "drafts")
# AUTH_DIR = os.path.join(DATA_DIR, "auth")
# for p in (DATA_DIR, USER_DIR, DRAFT_DIR, AUTH_DIR):
#     os.makedirs(p, exist_ok=True)

# def _safe(email: str) -> str:
#     return email.replace("@", "_at_").replace(".", "_")

# def user_file(email: str) -> str:
#     return os.path.join(USER_DIR, f"{_safe(email)}.json")

# def draft_file(email: str) -> str:
#     return os.path.join(DRAFT_DIR, f"{_safe(email)}_draft.json")

# AUTH_SESSION_FILE = os.path.join(AUTH_DIR, "session.json")

# # -------------------------------------------------------------
# # Syllabus + Default Plan
# # -------------------------------------------------------------
# SYLLABUS: Dict[str, List[str]] = {
#     "Probability & Statistics": [
#         "Counting (perm/comb)", "Axioms, sample space, events",
#         "Independence, mutual exclusivity", "Marginal/conditional/joint prob",
#         "Bayes theorem", "Cond. expectation & variance",
#         "Mean/Median/Mode/SD", "Correlation & Covariance",
#         "RVs: discrete/pmf, continuous/pdf, CDF",
#         "Distributions: Uniform, Bernoulli, Binomial, Poisson, Exponential, Normal, Standard Normal, t, Chi-squared",
#         "CLT", "Confidence intervals", "z/t/chi-squared tests"
#     ],
#     "Linear Algebra": [
#         "Vector spaces & subspaces", "Linear dependence/independence",
#         "Matrices (projection/orthogonal/idempotent/partition)",
#         "Quadratic forms", "Systems & Gaussian elimination",
#         "Eigenvalues/Eigenvectors", "Determinant, Rank, Nullity, Projections",
#         "LU, SVD"
#     ],
#     "Calculus & Optimization": [
#         "Single variable functions", "Limit/Continuity/Differentiability",
#         "Taylor series", "Maxima/Minima", "Single-variable optimization"
#     ],
#     "Programming & DSA": [
#         "Python basics", "Stacks/Queues/Linked Lists/Trees/Hash Tables",
#         "Search: Linear/Binary", "Sorting: Selection/Bubble/Insertion",
#         "Divide-Conquer: Merge/Quick", "Graph intro & Traversals/Shortest path"
#     ],
#     "DBMS & Warehousing": [
#         "ER model", "Relational algebra & tuple calculus", "SQL",
#         "Integrity constraints", "Normal forms", "File org & Indexing",
#         "Data types & transformation: normalization, discretization, sampling, compression",
#         "Warehouse: schema, hierarchies, measures"
#     ],
#     "ML — Supervised": [
#         "Regression (simple/multiple/ridge)", "Logistic regression",
#         "k-NN", "Naive Bayes", "LDA",
#         "SVM", "Decision Trees",
#         "Bias-variance trade-off", "Cross-Validation: LOO, k-folds",
#         "MLP/Feed-forward NN"
#     ],
#     "ML — Unsupervised & PCA": [
#         "Clustering: k-means/k-medoids", "Hierarchical (top-down/bottom-up)",
#         "Single/Complete/Avg-linkage", "Dimensionality Reduction",
#         "PCA"
#     ],
#     "AI (Search/Logic/Uncertainty)": [
#         "Search: Informed/Uninformed/Adversarial",
#         "Logic: Propositional & Predicate",
#         "Uncertainty: Conditional independence, Variable elimination (exact), Sampling (approx.)"
#     ],
#     "Aptitude/Revision/PYQs": [
#         "Aptitude", "Mixed PYQs (MSQ/NAT)", "Mini Projects/Case", "Formula/Notes"
#     ]
# }

# DEFAULT_PLAN: Dict[int, List[str]] = {
#     1: ["Probability & Statistics", "Programming & DSA", "ML — Supervised", "PYQs: Stats + DSA"],
#     2: ["Linear Algebra", "DBMS & Warehousing", "AI (Search/Logic/Uncertainty)", "PYQs: LA + DBMS"],
#     3: ["Calculus & Optimization", "Programming & DSA", "ML — Supervised", "PYQs: Calc + Algos"],
#     4: ["Probability & Statistics", "Programming & DSA", "AI (Search/Logic/Uncertainty)", "PYQs: Prob + Prog"],
#     5: ["Linear Algebra", "DBMS & Warehousing", "ML — Supervised", "PYQs: LA + DBMS"],
#     6: ["Calculus & Optimization", "Programming & DSA", "ML — Unsupervised & PCA", "PYQs: Opt + Graphs"],
#     7: ["Test: Stats", "Test: LA", "Weak Area Patch", "Formula/Notes"],
#     8: ["Aptitude/Revision/PYQs", "Mixed PYQs (MSQ/NAT)", "Mini Projects/Case", "Light Revision"]
# }

# # -------------------------------------------------------------
# # Data model
# # -------------------------------------------------------------
# @dataclass
# class LogRow:
#     Date: str
#     Subject: str
#     Completed: bool
#     Hours: float
#     Notes: str
#     Priority: str
#     Mode: str
#     Pomodoros: int
#     XP: int

# @dataclass
# class Settings:
#     theme: str = "dark"
#     dailyGoal: float = 6.0
#     pomoWork: int = 25
#     pomoBreak: int = 5
#     startDate: str = date.today().isoformat()
#     xp: int = 0
#     badges: List[str] = field(default_factory=list)
#     username: str = ""  # Display name

# @dataclass
# class UserData:
#     settings: Settings
#     plan: Dict[int, List[str]]
#     logs: List[LogRow]
#     syllabusProgress: Dict[str, bool]
#     taskOrder: Dict[str, List[str]]
#     dayChats: Dict[str, str] = field(default_factory=dict)  # per-day note/message

# # -------------------------------------------------------------
# # Utils
# # -------------------------------------------------------------
# def sha_bcrypt(password: str) -> str:
#     return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# def check_bcrypt(password: str, hashed: str) -> bool:
#     try:
#         return bcrypt.checkpw(password.encode(), hashed.encode())
#     except Exception:
#         return False

# def default_user_data() -> UserData:
#     return UserData(
#         settings=Settings(),
#         plan=json.loads(json.dumps(DEFAULT_PLAN)),
#         logs=[],
#         syllabusProgress={},
#         taskOrder={},
#         dayChats={}
#     )

# def load_user(email: str) -> Tuple[Optional[UserData], Optional[str]]:
#     p = user_file(email)
#     if not os.path.exists(p):
#         return None, None
#     try:
#         with open(p, "r", encoding="utf-8") as f:
#             raw = json.load(f)
#         passhash = raw.get("passHash")
#         d = raw.get("data", {}) or {}
#         # Safe defaults for settings
#         s_raw = d.get("settings") or {}
#         settings_defaults = Settings()
#         safe_settings = {k: s_raw.get(k, getattr(settings_defaults, k)) for k in settings_defaults.__dataclass_fields__.keys()}
#         settings = Settings(**safe_settings)
#         plan = d.get("plan") or {}
#         plan = {int(k): v for k, v in plan.items()} if plan else json.loads(json.dumps(DEFAULT_PLAN))
#         logs_list = d.get("logs") or []
#         logs = []
#         for r in logs_list:
#             if isinstance(r, dict):
#                 try:
#                     logs.append(LogRow(**r))
#                 except Exception:
#                     # skip malformed rows gracefully
#                     pass
#         syllabus = d.get("syllabusProgress") or {}
#         order = d.get("taskOrder") or {}
#         day_chats = d.get("dayChats") or {}
#         return UserData(settings, plan, logs, syllabus, order, day_chats), passhash
#     except Exception:
#         return None, None

# def save_user(email: str, data: UserData, pass_hash: str):
#     payload = {
#         "passHash": pass_hash,
#         "data": {
#             "settings": asdict(data.settings),
#             "plan": data.plan,
#             "logs": [asdict(r) for r in data.logs],
#             "syllabusProgress": data.syllabusProgress,
#             "taskOrder": data.taskOrder,
#             "dayChats": data.dayChats,
#         }
#     }
#     with open(user_file(email), "w", encoding="utf-8") as f:
#         json.dump(payload, f, indent=2)

# def save_session(email: Optional[str], pass_hash: Optional[str]):
#     with open(AUTH_SESSION_FILE, "w", encoding="utf-8") as f:
#         json.dump({"email": email, "passHash": pass_hash}, f)

# def load_session() -> Tuple[Optional[str], Optional[str]]:
#     if not os.path.exists(AUTH_SESSION_FILE):
#         return None, None
#     try:
#         with open(AUTH_SESSION_FILE, "r", encoding="utf-8") as f:
#             s = json.load(f)
#         return s.get("email"), s.get("passHash")
#     except Exception:
#         return None, None

# def load_draft(email: str) -> List[Dict[str, Any]]:
#     fp = draft_file(email)
#     if not os.path.exists(fp):
#         return []
#     try:
#         with open(fp, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return []

# def save_draft(email: str, rows: List[Dict[str, Any]]):
#     with open(draft_file(email), "w", encoding="utf-8") as f:
#         json.dump(rows, f, indent=2)

# def today_iso() -> str:
#     return date.today().isoformat()

# def cycle_day(settings: Settings) -> int:
#     try:
#         s = datetime.fromisoformat(settings.startDate).date()
#     except Exception:
#         s = date.today()
#     diff = (date.today() - s).days
#     return (diff % 8) + 1

# def subjects_for_today(data: UserData) -> List[str]:
#     d = cycle_day(data.settings)
#     base = data.plan.get(d, [])
#     order = data.taskOrder.get(f"day{d}", [])
#     if order and set(order) == set(base):
#         return order
#     return base

# def xp_from(hours: float, completed: bool, priority: str) -> int:
#     base = int(round((hours or 0) * 10))
#     bonus = 10 if completed else 0
#     pr_bonus = {"Low": 0, "Medium": 5, "High": 10}.get(priority, 0)
#     return base + bonus + pr_bonus

# def streak_days(data: UserData) -> int:
#     try:
#         goal = float(data.settings.dailyGoal)
#     except Exception:
#         goal = 6.0
#     by_date: Dict[str, float] = {}
#     for r in data.logs:
#         by_date[r.Date] = by_date.get(r.Date, 0.0) + float(r.Hours or 0.0)
#     # include draft preview for streak
#     if st.session_state.get("active_email"):
#         for r in st.session_state.get("draft_rows", []):
#             by_date[r.get("Date", "")] = by_date.get(r.get("Date", ""), 0.0) + float(r.get("Hours") or 0.0)
#     n = 0
#     cur = date.today()
#     for _ in range(365):
#         iso = cur.isoformat()
#         if by_date.get(iso, 0.0) >= goal:
#             n += 1
#             cur = cur - timedelta(days=1)
#         else:
#             break
#     return n

# # -------------------------------------------------------------
# # Session bootstrap
# # -------------------------------------------------------------
# def ensure_session_state():
#     if "active_email" not in st.session_state:
#         email, ph = load_session()
#         st.session_state.active_email = email
#         st.session_state.pass_hash = ph
#         st.session_state.data = None
#         if email and ph:
#             data, onfile_ph = load_user(email)
#             if data and onfile_ph == ph:
#                 st.session_state.data = data
#             else:
#                 # Invalid on-disk user; clear session pointer
#                 save_session(None, None)
#                 st.session_state.active_email = None
#                 st.session_state.pass_hash = None
#                 st.session_state.data = None
#     if "auto_save" not in st.session_state:
#         st.session_state.auto_save = True
#     if "draft_rows" not in st.session_state:
#         if st.session_state.active_email:
#             st.session_state.draft_rows = load_draft(st.session_state.active_email)
#         else:
#             st.session_state.draft_rows = []
#     if "timers" not in st.session_state:
#         st.session_state.timers = {}
#     if "deleted_buffer" not in st.session_state:
#         st.session_state.deleted_buffer = []
#     if "order_state" not in st.session_state:
#         st.session_state.order_state = {}

# def persist_all():
#     if st.session_state.get("active_email") and st.session_state.get("pass_hash") and st.session_state.get("data"):
#         save_user(st.session_state.active_email, st.session_state.data, st.session_state.pass_hash)
#         save_draft(st.session_state.active_email, st.session_state.draft_rows)

# # -------------------------------------------------------------
# # Auth UI
# # -------------------------------------------------------------
# def view_login():
#     st.header("Login to StudyTracker (Local)")
#     email = st.text_input("Email", key="login_email")
#     password = st.text_input("Password", type="password", key="login_pass")
#     c1, c2 = st.columns(2)
#     with c1:
#         if st.button("Login", use_container_width=True):
#             if not email or not password:
#                 st.error("Please enter email and password")
#                 return
#             data, ph = load_user(email)
#             if data is None or ph is None:
#                 st.error("Account not found. Please sign up.")
#                 return
#             if not check_bcrypt(password, ph):
#                 st.error("Incorrect password")
#                 return
#             st.session_state.active_email = email
#             st.session_state.pass_hash = ph
#             st.session_state.data = data
#             save_session(email, ph)
#             st.session_state.draft_rows = load_draft(email)
#             st.toast("Logged in")
#             st.rerun()
#     with c2:
#         if st.button("Sign Up", use_container_width=True, type="secondary"):
#             if not email or not password:
#                 st.error("Please enter email and password")
#                 return
#             p = user_file(email)
#             if os.path.exists(p):
#                 st.error("Account already exists. Please log in.")
#                 return
#             phash = sha_bcrypt(password)
#             data = default_user_data()
#             save_user(email, data, phash)
#             save_session(email, phash)
#             st.session_state.active_email = email
#             st.session_state.pass_hash = phash
#             st.session_state.data = data
#             st.session_state.draft_rows = []
#             st.toast("Signed up")
#             st.rerun()

# # -------------------------------------------------------------
# # KPI + Draft management
# # -------------------------------------------------------------
# def kpi_cards(data: UserData):
#     try:
#         logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
#             "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
#         ])
#     except Exception:
#         logs_df = pd.DataFrame(columns=["Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"])
#     dr = st.session_state.get("draft_rows", [])
#     draft_df = pd.DataFrame(dr) if dr else pd.DataFrame(columns=logs_df.columns)

#     today = today_iso()
#     today_hours = float(draft_df[draft_df.get("Date") == today]["Hours"].sum() if not draft_df.empty else 0.0)
#     today_completed = int(draft_df[(draft_df.get("Date") == today) & (draft_df.get("Completed") == True)].shape[0] if not draft_df.empty else 0)

#     avg_all = 0.0
#     if not logs_df.empty:
#         try:
#             avg_all = float(logs_df.groupby("Date")["Hours"].sum().mean())
#         except Exception:
#             avg_all = 0.0

#     avg_7 = 0.0
#     if not logs_df.empty:
#         last7 = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
#         ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum()
#         merged = pd.DataFrame({"Date": last7}).merge(ddf, on="Date", how="left").fillna({"Hours": 0})
#         avg_7 = float(merged["Hours"].mean())

#     display_name = ""
#     try:
#         display_name = (data.settings.username or st.session_state.get("active_email") or "").strip()
#     except Exception:
#         display_name = (st.session_state.get("active_email") or "").strip()

#     c1, c2, c3, c4, c5 = st.columns(5)
#     goal_val = 6.0
#     try:
#         goal_val = float(data.settings.dailyGoal)
#     except Exception:
#         pass
#     c1.metric("Goal", f"{goal_val:.2f}h")
#     c2.metric(f"Today ({display_name})", f"{today_hours:.2f}h")
#     c3.metric("Completed Today", str(today_completed))
#     c4.metric("Streak", str(streak_days(data)))
#     c5.metric("Avg Hours (All-time)", f"{avg_all:.2f}h")

#     st.caption(f"Avg Hours (Last 7d): {avg_7:.2f}h")
#     st.progress(min(1.0, today_hours / max(1e-6, goal_val)))

# def upsert_draft(subject: str, hours: float, done: bool, mode: str, priority: str, notes: str, pomos: int):
#     completed = bool(done and float(hours) >= 3.0)
#     row = {
#         "Date": today_iso(),
#         "Subject": subject,
#         "Completed": completed,
#         "Hours": float(hours),
#         "Notes": notes or "",
#         "Priority": priority,
#         "Mode": mode,
#         "Pomodoros": int(pomos),
#         "XP": int(xp_from(hours, completed, priority))
#     }
#     rows = st.session_state.get("draft_rows", [])
#     found = False
#     for r in rows:
#         if r.get("Subject") == subject and r.get("Date") == row["Date"]:
#             r.update(row)
#             found = True
#             break
#     if not found:
#         rows.append(row)
#         st.session_state.draft_rows = rows
#     if st.session_state.get("auto_save", True):
#         persist_all()

# def clear_draft_today():
#     today = today_iso()
#     rows = st.session_state.get("draft_rows", [])
#     rows = [r for r in rows if r.get("Date") != today]
#     st.session_state.draft_rows = rows
#     if st.session_state.get("auto_save", True):
#         persist_all()

# def save_log(data: UserData):
#     today = today_iso()
#     rows = st.session_state.get("draft_rows", [])
#     todays = [r for r in rows if r.get("Date") == today and (r.get("Hours", 0) > 0 or r.get("Notes"))]
#     if not todays:
#         st.toast("No data to save")
#         return
#     for r in todays:
#         try:
#             data.logs.append(LogRow(**r))
#             data.settings.xp += int(r.get("XP", 0))
#         except Exception:
#             pass
#     st.session_state.draft_rows = [r for r in rows if r.get("Date") != today]
#     total_hours = sum((float(l.Hours or 0.0) for l in data.logs), 0.0)
#     badges = set(data.settings.badges or [])
#     for threshold, name in [(10, "Rookie"), (50, "Committed"), (100, "Centurion"), (200, "Marathoner")]:
#         if total_hours >= threshold:
#             badges.add(name)
#     data.settings.badges = sorted(badges)
#     persist_all()
#     st.toast("Log saved")

# # -------------------------------------------------------------
# # Today View
# # -------------------------------------------------------------
# def timer_ui(subject: str, data: UserData):
#     tmap = st.session_state.get("timers", {})
#     if subject not in tmap:
#         tmap[subject] = {"running": False, "start": 0.0, "elapsed_ms": 0, "pomos": 0, "use_pomo": False}
#         st.session_state.timers = tmap
#     t = tmap[subject]

#     def hms(ms: int) -> str:
#         s = ms // 1000
#         h = s // 3600
#         s -= h*3600
#         m = s // 60
#         s -= m*60
#         return f"{h:02d}:{m:02d}:{s:02d}"

#     c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
#     c1.write(f"⏱️ {hms(int(t['elapsed_ms']))}")
#     start = c2.button("Start", key=f"start_{subject}")
#     pause = c3.button("Pause", key=f"pause_{subject}")
#     stop = c4.button("Stop", key=f"stop_{subject}")
#     t["use_pomo"] = c5.toggle("Pomodoro", value=t.get("use_pomo", False), key=f"pomo_{subject}")

#     now = time.time()
#     if t["running"]:
#         t["elapsed_ms"] = int((now - t["start"]) * 1000)

#     if start and not t["running"]:
#         t["running"] = True
#         t["start"] = now - (t["elapsed_ms"] / 1000.0)
#         st.toast(f"Timer started: {subject}")

#     if pause and t["running"]:
#         t["running"] = False
#         t["elapsed_ms"] = int((now - t["start"]) * 1000)
#         st.toast(f"Timer paused: {subject}")

#     if t["use_pomo"]:
#         work = (data.settings.pomoWork or 25) * 60 * 1000
#         brk = (data.settings.pomoBreak or 5) * 60 * 1000
#         cyc = work + brk
#         completed = t["elapsed_ms"] // cyc
#         if completed > t["pomos"]:
#             t["pomos"] = int(completed)
#             st.toast(f"Pomodoro complete: {subject}")
#     st.caption(f"🍅 {t['pomos']}")

#     if stop and (t["running"] or t["elapsed_ms"] > 0):
#         if t["running"]:
#             t["elapsed_ms"] = int((now - t["start"]) * 1000)
#         t["running"] = False
#         st.toast(f"Timer stopped: {subject}")

#     hrs = min(10.0, round(t["elapsed_ms"] / 3600000.0, 2))
#     return hrs, t["pomos"]

# def reorder_ui(data: UserData):
#     d = cycle_day(data.settings)
#     st.subheader(f"Today's Subjects — Day {d} of 8")
#     subjects = data.plan.get(d, [])
#     key = f"day{d}"
#     current = data.taskOrder.get(key, subjects[:])

#     st.caption("Reorder with arrow buttons and Save.")

#     arr = st.session_state["order_state"].setdefault(key, current[:])

#     def move(i, di):
#         j = i + di
#         if 0 <= j < len(arr):
#             arr[i], arr[j] = arr[j], arr[i]

#     for i, s in enumerate(arr):
#         c1, c2, c3 = st.columns([8,1,1])
#         c1.write(f"- {s}")
#         if c2.button("↑", key=f"up_{key}_{i}"):
#             move(i, -1)
#             st.rerun()
#         if c3.button("↓", key=f"dn_{key}_{i}"):
#             move(i, 1)
#             st.rerun()

#     if st.button("Save Order", key=f"save_order_{key}", type="primary"):
#         data.taskOrder[key] = arr[:]
#         st.toast("Order saved")
#         if st.session_state.get("auto_save", True):
#             persist_all()

# def today_view(data: Optional[UserData]):
#     if not data or not getattr(data, "settings", None):
#         st.warning("No user data loaded. Please log in again.")
#         return

#     st.header("Today's Plan")
#     st.caption("Hours up to 10h/subject. Completed only if hours ≥ 3h.")
#     kpi_cards(data)

#     a1, a2, a3, a4, a5 = st.columns(5)
#     if a1.button("💾 Save Log", type="primary", use_container_width=True):
#         save_log(data)
#         st.rerun()
#     if a2.button("➡ Next Day", use_container_width=True):
#         try:
#             s = datetime.fromisoformat(data.settings.startDate).date()
#         except Exception:
#             s = date.today()
#         data.settings.startDate = (s - timedelta(days=1)).isoformat()
#         if st.session_state.get("auto_save", True):
#             persist_all()
#         st.rerun()
#     if a3.button("🔄 Refresh", use_container_width=True):
#         st.rerun()
#     if a4.button("🧹 Clear Today Draft", use_container_width=True):
#         clear_draft_today()
#         st.rerun()
#     st.session_state.auto_save = a5.toggle("Auto-save on Change", value=st.session_state.get("auto_save", True))

#     st.divider()

#     # Daily message
#     st.subheader("Daily Note")
#     tdy = today_iso()
#     msg = data.dayChats.get(tdy, "")
#     new_msg = st.text_area("Today's note", value=msg, height=80, placeholder="How did study go? Wins, blockers, next steps...")
#     cmsg1, cmsg2 = st.columns(2)
#     if cmsg1.button("Save Today Note"):
#         data.dayChats[tdy] = new_msg.strip()
#         st.toast("Today's note saved")
#         if st.session_state.get("auto_save", True):
#             persist_all()
#     if cmsg2.button("Delete Today Note", type="secondary"):
#         if tdy in data.dayChats:
#             del data.dayChats[tdy]
#             st.toast("Today's note deleted")
#             if st.session_state.get("auto_save", True):
#                 persist_all()

#     st.divider()
#     reorder_ui(data)
#     st.divider()

#     for subject in subjects_for_today(data):
#         with st.container(border=True):
#             st.subheader(subject)
#             st.caption("Enter hours and details. Completed requires ≥ 3h.")
#             sugg, pomos = timer_ui(subject, data)

#             c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,1.2])
#             hours = c1.number_input("Hours", min_value=0.0, max_value=10.0, step=0.25, value=float(sugg), key=f"hrs_{subject}")
#             mode = c2.selectbox("Mode", ["Deep Work","Review","Practice","PYQs","Test"], key=f"mode_{subject}")
#             priority = c3.selectbox("Priority", ["Low","Medium","High"], index=1, key=f"prio_{subject}")
#             done_checkbox = c4.checkbox("Completed (≥3h)", key=f"done_{subject}")

#             if float(hours) < 3.0 and done_checkbox:
#                 st.warning("Completion requires ≥ 3.0 hours. It will be saved as not completed.")

#             notes = st.text_area("Notes", value="", height=80, key=f"notes_{subject}", placeholder="Concepts, mistakes, formulas, tasks...")

#             upsert_draft(subject, hours, done_checkbox, mode, priority, notes, pomos)

# # -------------------------------------------------------------
# # Syllabus View
# # -------------------------------------------------------------
# def syllabus_view(data: Optional[UserData]):
#     if not data or not getattr(data, "settings", None):
#         st.warning("No user data loaded. Please log in again.")
#         return

#     st.header("DA Syllabus Map")
#     for cat, items in SYLLABUS.items():
#         with st.expander(cat, expanded=False):
#             comp = sum(1 for t in items if data.syllabusProgress.get(f"{cat}:{t}", False))
#             st.caption(f"{comp}/{len(items)} completed")
#             for t in items:
#                 k = f"{cat}:{t}"
#                 val = st.checkbox(t, value=data.syllabusProgress.get(k, False), key=f"sy_{k}")
#                 data.syllabusProgress[k] = bool(val)
#     if st.session_state.get("auto_save", True):
#         persist_all()

# # -------------------------------------------------------------
# # Logs View (with per-row delete + undo)
# # -------------------------------------------------------------
# def logs_view(data: Optional[UserData]):
#     if not data or not getattr(data, "settings", None):
#         st.warning("No user data loaded. Please log in again.")
#         return

#     st.header("Logs")

#     try:
#         logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
#             "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
#         ])
#     except Exception:
#         logs_df = pd.DataFrame(columns=["Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"])

#     if logs_df.empty:
#         st.info("No logs yet.")
#     else:
#         st.dataframe(logs_df, use_container_width=True, height=420)

#     st.subheader("Manage Rows")
#     del_idx = st.number_input("Row index to delete", min_value=0, step=1, value=0 if not logs_df.empty else 0, disabled=logs_df.empty)
#     c1, c2, c3 = st.columns(3)
#     if c1.button("Delete Row", type="secondary", disabled=logs_df.empty):
#         if 0 <= del_idx < len(data.logs):
#             removed = data.logs.pop(int(del_idx))
#             st.session_state.deleted_buffer.append(removed)
#             st.toast(f"Deleted row {del_idx}")
#             if st.session_state.get("auto_save", True):
#                 persist_all()
#             st.rerun()
#     if c2.button("Undo Last Delete", disabled=(len(st.session_state.deleted_buffer)==0)):
#         rec = st.session_state.deleted_buffer.pop()
#         data.logs.append(rec)
#         st.toast("Undo successful")
#         if st.session_state.get("auto_save", True):
#             persist_all()
#         st.rerun()
#     if c3.button("🗑 Clear All Logs", type="secondary", disabled=logs_df.empty):
#         data.logs = []
#         st.session_state.deleted_buffer = []
#         st.toast("Logs cleared")
#         if st.session_state.get("auto_save", True):
#             persist_all()
#         st.rerun()

#     st.subheader("Import/Export")
#     e1, e2, e3, e4, e5 = st.columns(5)

#     with e1:
#         if st.button("⬇ Export CSV", use_container_width=True):
#             csv = logs_df.to_csv(index=False)
#             st.download_button("Download study_log.csv", data=csv, file_name="study_log.csv", mime="text/csv", use_container_width=True)

#     with e2:
#         if st.button("💾 Save JSON (Full Backup)", use_container_width=True):
#             payload = {
#                 "email": st.session_state.get("active_email"),
#                 "data": {
#                     "settings": asdict(st.session_state.data.settings),
#                     "plan": st.session_state.data.plan,
#                     "logs": [asdict(r) for r in st.session_state.data.logs],
#                     "syllabusProgress": st.session_state.data.syllabusProgress,
#                     "taskOrder": st.session_state.data.taskOrder,
#                     "dayChats": st.session_state.data.dayChats,
#                 }
#             }
#             st.download_button("Download data.json", data=json.dumps(payload, indent=2),
#                                file_name="data.json", mime="application/json", use_container_width=True)

#     with e3:
#         if st.button("💾 Save Excel (Logs Only)", use_container_width=True):
#             out = io.BytesIO()
#             with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
#                 (logs_df if not logs_df.empty else pd.DataFrame(columns=[
#                     "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
#                 ])).to_excel(writer, index=False, sheet_name="Logs")
#             st.download_button("Download study_log.xlsx", data=out.getvalue(),
#                                file_name="study_log.xlsx",
#                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                                use_container_width=True)

#     with e4:
#         if st.button("💾 Save Excel (Full Export)", use_container_width=True):
#             out = io.BytesIO()
#             with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
#                 # Logs sheet
#                 (logs_df if not logs_df.empty else pd.DataFrame(columns=[
#                     "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
#                 ])).to_excel(writer, index=False, sheet_name="Logs")
#                 # Profile sheet
#                 profile_df = pd.DataFrame([{
#                     "Email": st.session_state.get("active_email") or "",
#                     "Username": st.session_state.data.settings.username or "",
#                     "DailyGoal": st.session_state.data.settings.dailyGoal,
#                     "PomoWork": st.session_state.data.settings.pomoWork,
#                     "PomoBreak": st.session_state.data.settings.pomoBreak,
#                     "StartDate": st.session_state.data.settings.startDate,
#                     "XP": st.session_state.data.settings.xp,
#                     "Badges": ", ".join(st.session_state.data.settings.badges or []),
#                 }])
#                 profile_df.to_excel(writer, index=False, sheet_name="Profile")
#                 # Day notes
#                 if st.session_state.data.dayChats:
#                     day_notes_df = pd.DataFrame([{"Date": k, "Note": v} for k, v in sorted(st.session_state.data.dayChats.items())])
#                 else:
#                     day_notes_df = pd.DataFrame(columns=["Date","Note"])
#                 day_notes_df.to_excel(writer, index=False, sheet_name="DayNotes")
#             st.download_button("Download study_export.xlsx", data=out.getvalue(),
#                                file_name="study_export.xlsx",
#                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                                use_container_width=True)

#     with e5:
#         uploaded = st.file_uploader("⬆ Import data.json", type=["json"], label_visibility="collapsed")
#         if uploaded is not None and st.button("Import", use_container_width=True):
#             try:
#                 parsed = json.loads(uploaded.read().decode("utf-8"))
#                 if "data" not in parsed:
#                     st.error("Invalid data.json format")
#                 else:
#                     d = parsed["data"]
#                     settings = Settings(**(d.get("settings") or {}))
#                     plan = {int(k): v for k, v in (d.get("plan") or {}).items()} if d.get("plan") else json.loads(json.dumps(DEFAULT_PLAN))
#                     logs = [LogRow(**r) for r in (d.get("logs") or []) if isinstance(r, dict)]
#                     syllabus = d.get("syllabusProgress") or {}
#                     order = d.get("taskOrder") or {}
#                     day_chats = d.get("dayChats") or {}
#                     st.session_state.data = UserData(settings, plan, logs, syllabus, order, day_chats)
#                     st.toast("Imported data.json")
#                     persist_all()
#                     st.rerun()
#             except Exception as e:
#                 st.error(f"Import failed: {e}")

# # -------------------------------------------------------------
# # Dashboard View
# # -------------------------------------------------------------
# def dashboard_view(data: Optional[UserData]):
#     if not data or not getattr(data, "settings", None):
#         st.warning("No user data loaded. Please log in again.")
#         return

#     st.header("Dashboard")
#     try:
#         logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
#             "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
#         ])
#     except Exception:
#         logs_df = pd.DataFrame(columns=["Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"])

#     cA, cB = st.columns(2)
#     with cA:
#         st.subheader("Daily Hours")
#         if not logs_df.empty:
#             ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum().sort_values("Date")
#             chart = alt.Chart(ddf).mark_line(point=True).encode(
#                 x="Date:T",
#                 y=alt.Y("Hours:Q", scale=alt.Scale(domain=[0, max(3.0, float(ddf['Hours'].max()))])),
#                 tooltip=["Date","Hours"]
#             ).properties(height=260)
#             st.altair_chart(chart, use_container_width=True)
#         else:
#             st.info("No data")

#     with cB:
#         st.subheader("Hours by Subject")
#         if not logs_df.empty:
#             sdf = logs_df.groupby("Subject", as_index=False)["Hours"].sum().sort_values("Hours", ascending=False)
#             chart = alt.Chart(sdf).mark_bar().encode(
#                 x="Hours:Q",
#                 y=alt.Y("Subject:N", sort="-x"),
#                 tooltip=["Subject","Hours"]
#             ).properties(height=260)
#             st.altair_chart(chart, use_container_width=True)
#         else:
#             st.info("No data")

#     cC, cD = st.columns(2)
#     with cC:
#         st.subheader("Completion Share")
#         if not logs_df.empty:
#             total = len(logs_df)
#             comp = int((logs_df["Completed"]==True).sum())
#             pie_df = pd.DataFrame({"Status":["Completed","Pending"],"Count":[comp, max(0,total-comp)]})
#             pie = alt.Chart(pie_df).mark_arc().encode(
#                 theta="Count:Q", color="Status:N", tooltip=["Status","Count"]
#             ).properties(height=260)
#             st.altair_chart(pie, use_container_width=True)
#         else:
#             st.info("No data")

#     with cD:
#         st.subheader("Last 7 Days")
#         if not logs_df.empty:
#             last7 = [(date.today()-timedelta(days=i)).isoformat() for i in range(6,-1,-1)]
#             ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum()
#             merged = pd.DataFrame({"Date": last7}).merge(ddf, on="Date", how="left").fillna({"Hours":0})
#             chart = alt.Chart(merged).mark_bar().encode(
#                 x=alt.X("Date:T", sort=None),
#                 y=alt.Y("Hours:Q", scale=alt.Scale(domain=[0, max(3.0, float(merged['Hours'].max()))])),
#                 tooltip=["Date","Hours"]
#             ).properties(height=260)
#             st.altair_chart(chart, use_container_width=True)
#         else:
#             st.info("No data")

# # -------------------------------------------------------------
# # Settings View
# # -------------------------------------------------------------
# def plan_editor_view(data: UserData):
#     st.subheader("8-Day Plan Editor")
#     for d in range(1, 9):
#         with st.expander(f"Day {d}", expanded=(d == cycle_day(data.settings))):
#             txt = st.text_area(
#                 f"Subjects (one per line) — Day {d}",
#                 value="\n".join(data.plan.get(d, [])),
#                 height=120,
#                 key=f"plan_day_{d}"
#             )
#             if st.button(f"Save Day {d}", key=f"s_day_{d}"):
#                 arr = [x.strip() for x in txt.split("\n") if x.strip()]
#                 data.plan[d] = arr
#                 st.toast(f"Saved Day {d}")
#                 if st.session_state.get("auto_save", True):
#                     persist_all()

# def settings_view(data: Optional[UserData]):
#     if not data or not getattr(data, "settings", None):
#         st.warning("No user data loaded. Please log in again.")
#         return

#     st.header("Settings")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("General")
#         uname = st.text_input("Username (display name)", value=data.settings.username or "")
#         g1, g2 = st.columns(2)
#         daily_goal = g1.number_input("Daily Goal (hrs)", min_value=0.0, step=0.5, value=float(data.settings.dailyGoal))
#         theme = g2.selectbox("Theme", ["dark","light"], index=(0 if data.settings.theme=="dark" else 1))
#         p1, p2 = st.columns(2)
#         pomo_work = p1.number_input("Pomodoro Work (min)", min_value=1, step=1, value=int(data.settings.pomoWork))
#         pomo_break = p2.number_input("Pomodoro Break (min)", min_value=1, step=1, value=int(data.settings.pomoBreak))

#         s1, s2, s3 = st.columns(3)
#         if s1.button("Save", type="primary"):
#             data.settings.username = uname.strip()
#             data.settings.dailyGoal = float(daily_goal)
#             data.settings.theme = theme
#             data.settings.pomoWork = int(pomo_work)
#             data.settings.pomoBreak = int(pomo_break)
#             st.toast("Settings saved")
#             if st.session_state.get("auto_save", True):
#                 persist_all()

#         if s2.button("Export Config+Plan"):
#             payload = {"settings": asdict(data.settings), "plan": data.plan}
#             st.download_button("Download config_plan.json", data=json.dumps(payload, indent=2),
#                                file_name="config_plan.json", mime="application/json")

#         if s3.button("Reset 8-Day Plan", type="secondary"):
#             data.plan = json.loads(json.dumps(DEFAULT_PLAN))
#             st.toast("Plan reset")
#             if st.session_state.get("auto_save", True):
#                 persist_all()

#     with col2:
#         plan_editor_view(data)

# # -------------------------------------------------------------
# # Header
# # -------------------------------------------------------------
# def header_bar(data: Optional[UserData]):
#     # Defensive: build a safe display name even if data/settings are None
#     if not data or not isinstance(data, UserData) or not getattr(data, "settings", None):
#         display_name = (st.session_state.get("active_email") or "").strip()
#     else:
#         try:
#             display_name = (data.settings.username or st.session_state.get("active_email") or "").strip()
#         except Exception:
#             display_name = (st.session_state.get("active_email") or "").strip()

#     left, right = st.columns([3,2])
#     with left:
#         st.markdown("### StudyTracker · Data Science & AI")
#         st.caption("GATE DA 2026")
#     with right:
#         st.write(f"👤 {display_name}")
#         st.write(f"📅 {today_iso()}")
#         hb1, hb2, hb3 = st.columns(3)
#         if hb1.button("Logout", use_container_width=True):
#             persist_all()
#             save_session(None, None)
#             st.session_state.active_email = None
#             st.session_state.pass_hash = None
#             st.session_state.data = None
#             st.session_state.draft_rows = []
#             st.session_state.timers = {}
#             st.session_state.deleted_buffer = []
#             st.toast("Logged out")
#             st.rerun()
#         if data and getattr(data, "settings", None):
#             if hb2.button("🌗 Theme", use_container_width=True):
#                 data.settings.theme = "light" if data.settings.theme == "dark" else "dark"
#                 st.toast("Theme switched")
#                 if st.session_state.get("auto_save", True):
#                     persist_all()
#         if hb3.button("🎯 Focus", use_container_width=True):
#             st.sidebar.write("Focus mode toggled. Collapse sidebar if desired.")
#             st.toast("Focus toggled")
#     st.divider()

# # -------------------------------------------------------------
# # Main
# # -------------------------------------------------------------
# def main():
#     ensure_session_state()

#     # If data is missing or malformed, show login rather than proceeding
#     if not st.session_state.get("active_email") or not st.session_state.get("data") or not getattr(st.session_state.get("data"), "settings", None):
#         view_login()
#         return

#     data: UserData = st.session_state.data
#     header_bar(data)

#     t1, t2, t3, t4, t5 = st.tabs(["Today", "Syllabus", "Log", "Dashboard", "Settings"])
#     with t1:
#         today_view(data)
#     with t2:
#         syllabus_view(data)
#     with t3:
#         logs_view(data)
#     with t4:
#         dashboard_view(data)
#     with t5:
#         settings_view(data)

#     if st.session_state.get("auto_save", True):
#         persist_all()

# if __name__ == "__main__":
#     main()





# # import json
# # import os
# # import time
# # import bcrypt
# # from datetime import datetime, timedelta, date
# # from typing import Dict, List, Optional, Any, Tuple
# # from dataclasses import dataclass, asdict, field

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import altair as alt

# # # -------------------------------------------------------------
# # # App Config
# # # -------------------------------------------------------------
# # st.set_page_config(
# #     page_title="StudyTracker — GATE DA 2026",
# #     page_icon="📚",
# #     layout="wide",
# #     initial_sidebar_state="expanded",
# # )

# # # -------------------------------------------------------------
# # # Storage paths
# # # -------------------------------------------------------------
# # DATA_DIR = ".studytracker_data"
# # USER_DIR = os.path.join(DATA_DIR, "users")
# # DRAFT_DIR = os.path.join(DATA_DIR, "drafts")
# # AUTH_DIR = os.path.join(DATA_DIR, "auth")
# # for p in (DATA_DIR, USER_DIR, DRAFT_DIR, AUTH_DIR):
# #     os.makedirs(p, exist_ok=True)

# # def user_file(email: str) -> str:
# #     safe = email.replace("@", "_at_").replace(".", "_")
# #     return os.path.join(USER_DIR, f"{safe}.json")

# # def draft_file(email: str) -> str:
# #     safe = email.replace("@", "_at_").replace(".", "_")
# #     return os.path.join(DRAFT_DIR, f"{safe}_draft.json")

# # AUTH_SESSION_FILE = os.path.join(AUTH_DIR, "session.json")  # persist active_email + passhash

# # # -------------------------------------------------------------
# # # Syllabus + Default Plan
# # # -------------------------------------------------------------
# # SYLLABUS: Dict[str, List[str]] = {
# #     "Probability & Statistics": [
# #         "Counting (perm/comb)", "Axioms, sample space, events",
# #         "Independence, mutual exclusivity", "Marginal/conditional/joint prob",
# #         "Bayes theorem", "Cond. expectation & variance",
# #         "Mean/Median/Mode/SD", "Correlation & Covariance",
# #         "RVs: discrete/pmf, continuous/pdf, CDF",
# #         "Distributions: Uniform, Bernoulli, Binomial, Poisson, Exponential, Normal, Standard Normal, t, Chi-squared",
# #         "CLT", "Confidence intervals", "z/t/chi-squared tests"
# #     ],
# #     "Linear Algebra": [
# #         "Vector spaces & subspaces", "Linear dependence/independence",
# #         "Matrices (projection/orthogonal/idempotent/partition)",
# #         "Quadratic forms", "Systems & Gaussian elimination",
# #         "Eigenvalues/Eigenvectors", "Determinant, Rank, Nullity, Projections",
# #         "LU, SVD"
# #     ],
# #     "Calculus & Optimization": [
# #         "Single variable functions", "Limit/Continuity/Differentiability",
# #         "Taylor series", "Maxima/Minima", "Single-variable optimization"
# #     ],
# #     "Programming & DSA": [
# #         "Python basics", "Stacks/Queues/Linked Lists/Trees/Hash Tables",
# #         "Search: Linear/Binary", "Sorting: Selection/Bubble/Insertion",
# #         "Divide-Conquer: Merge/Quick", "Graph intro & Traversals/Shortest path"
# #     ],
# #     "DBMS & Warehousing": [
# #         "ER model", "Relational algebra & tuple calculus", "SQL",
# #         "Integrity constraints", "Normal forms", "File org & Indexing",
# #         "Data types & transformation: normalization, discretization, sampling, compression",
# #         "Warehouse: schema, hierarchies, measures"
# #     ],
# #     "ML — Supervised": [
# #         "Regression (simple/multiple/ridge)", "Logistic regression",
# #         "k-NN", "Naive Bayes", "LDA",
# #         "SVM", "Decision Trees",
# #         "Bias-variance trade-off", "Cross-Validation: LOO, k-folds",
# #         "MLP/Feed-forward NN"
# #     ],
# #     "ML — Unsupervised & PCA": [
# #         "Clustering: k-means/k-medoids", "Hierarchical (top-down/bottom-up)",
# #         "Single/Complete/Avg-linkage", "Dimensionality Reduction",
# #         "PCA"
# #     ],
# #     "AI (Search/Logic/Uncertainty)": [
# #         "Search: Informed/Uninformed/Adversarial",
# #         "Logic: Propositional & Predicate",
# #         "Uncertainty: Conditional independence, Variable elimination (exact), Sampling (approx.)"
# #     ],
# #     "Aptitude/Revision/PYQs": [
# #         "Aptitude", "Mixed PYQs (MSQ/NAT)", "Mini Projects/Case", "Formula/Notes"
# #     ]
# # }

# # DEFAULT_PLAN: Dict[int, List[str]] = {
# #     1: ["Probability & Statistics", "Programming & DSA", "ML — Supervised", "PYQs: Stats + DSA"],
# #     2: ["Linear Algebra", "DBMS & Warehousing", "AI (Search/Logic/Uncertainty)", "PYQs: LA + DBMS"],
# #     3: ["Calculus & Optimization", "Programming & DSA", "ML — Supervised", "PYQs: Calc + Algos"],
# #     4: ["Probability & Statistics", "Programming & DSA", "AI (Search/Logic/Uncertainty)", "PYQs: Prob + Prog"],
# #     5: ["Linear Algebra", "DBMS & Warehousing", "ML — Supervised", "PYQs: LA + DBMS"],
# #     6: ["Calculus & Optimization", "Programming & DSA", "ML — Unsupervised & PCA", "PYQs: Opt + Graphs"],
# #     7: ["Test: Stats", "Test: LA", "Weak Area Patch", "Formula/Notes"],
# #     8: ["Aptitude/Revision/PYQs", "Mixed PYQs (MSQ/NAT)", "Mini Projects/Case", "Light Revision"]
# # }

# # # -------------------------------------------------------------
# # # Data model
# # # -------------------------------------------------------------
# # @dataclass
# # class LogRow:
# #     Date: str
# #     Subject: str
# #     Completed: bool
# #     Hours: float
# #     Notes: str
# #     Priority: str
# #     Mode: str
# #     Pomodoros: int
# #     XP: int

# # @dataclass
# # class Settings:
# #     theme: str = "dark"
# #     dailyGoal: float = 6.0
# #     pomoWork: int = 25
# #     pomoBreak: int = 5
# #     startDate: str = date.today().isoformat()
# #     xp: int = 0
# #     badges: List[str] = field(default_factory=list)

# # @dataclass
# # class UserData:
# #     settings: Settings
# #     plan: Dict[int, List[str]]
# #     logs: List[LogRow]
# #     syllabusProgress: Dict[str, bool]
# #     taskOrder: Dict[str, List[str]]

# # # -------------------------------------------------------------
# # # Utils
# # # -------------------------------------------------------------
# # def sha_bcrypt(password: str) -> str:
# #     return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# # def check_bcrypt(password: str, hashed: str) -> bool:
# #     try:
# #         return bcrypt.checkpw(password.encode(), hashed.encode())
# #     except Exception:
# #         return False

# # def default_user_data() -> UserData:
# #     return UserData(
# #         settings=Settings(),
# #         plan=json.loads(json.dumps(DEFAULT_PLAN)),
# #         logs=[],
# #         syllabusProgress={},
# #         taskOrder={}
# #     )

# # def load_user(email: str) -> Tuple[Optional[UserData], Optional[str]]:
# #     p = user_file(email)
# #     if not os.path.exists(p):
# #         return None, None
# #     try:
# #         with open(p, "r", encoding="utf-8") as f:
# #             raw = json.load(f)
# #         passhash = raw.get("passHash")
# #         d = raw.get("data", {})
# #         settings = Settings(**d.get("settings", {}))
# #         plan = {int(k): v for k, v in d.get("plan", {}).items()}
# #         logs = [LogRow(**r) for r in d.get("logs", [])]
# #         syllabus = d.get("syllabusProgress", {})
# #         order = d.get("taskOrder", {})
# #         return UserData(settings, plan, logs, syllabus, order), passhash
# #     except Exception:
# #         return None, None

# # def save_user(email: str, data: UserData, pass_hash: str):
# #     payload = {
# #         "passHash": pass_hash,
# #         "data": {
# #             "settings": asdict(data.settings),
# #             "plan": data.plan,
# #             "logs": [asdict(r) for r in data.logs],
# #             "syllabusProgress": data.syllabusProgress,
# #             "taskOrder": data.taskOrder
# #         }
# #     }
# #     with open(user_file(email), "w", encoding="utf-8") as f:
# #         json.dump(payload, f, indent=2)

# # def save_session(email: Optional[str], pass_hash: Optional[str]):
# #     with open(AUTH_SESSION_FILE, "w", encoding="utf-8") as f:
# #         json.dump({"email": email, "passHash": pass_hash}, f)

# # def load_session() -> Tuple[Optional[str], Optional[str]]:
# #     if not os.path.exists(AUTH_SESSION_FILE):
# #         return None, None
# #     try:
# #         with open(AUTH_SESSION_FILE, "r", encoding="utf-8") as f:
# #             s = json.load(f)
# #         return s.get("email"), s.get("passHash")
# #     except Exception:
# #         return None, None

# # def load_draft(email: str) -> List[Dict[str, Any]]:
# #     fp = draft_file(email)
# #     if not os.path.exists(fp):
# #         return []
# #     try:
# #         with open(fp, "r", encoding="utf-8") as f:
# #             return json.load(f)
# #     except Exception:
# #         return []

# # def save_draft(email: str, rows: List[Dict[str, Any]]):
# #     with open(draft_file(email), "w", encoding="utf-8") as f:
# #         json.dump(rows, f, indent=2)

# # def today_iso() -> str:
# #     return date.today().isoformat()

# # def cycle_day(settings: Settings) -> int:
# #     try:
# #         s = datetime.fromisoformat(settings.startDate).date()
# #     except Exception:
# #         s = date.today()
# #     diff = (date.today() - s).days
# #     return (diff % 8) + 1

# # def subjects_for_today(data: UserData) -> List[str]:
# #     d = cycle_day(data.settings)
# #     base = data.plan.get(d, [])
# #     order = data.taskOrder.get(f"day{d}", [])
# #     if order and set(order) == set(base):
# #         return order
# #     return base

# # def xp_from(hours: float, completed: bool, priority: str) -> int:
# #     base = int(round((hours or 0) * 10))
# #     bonus = 10 if completed else 0
# #     pr_bonus = {"Low": 0, "Medium": 5, "High": 10}.get(priority, 0)
# #     return base + bonus + pr_bonus

# # def streak_days(data: UserData) -> int:
# #     goal = data.settings.dailyGoal
# #     by_date: Dict[str, float] = {}
# #     for r in data.logs:
# #         by_date[r.Date] = by_date.get(r.Date, 0.0) + float(r.Hours or 0.0)
# #     # Include today's draft in streak computation preview
# #     if st.session_state.active_email:
# #         for r in st.session_state.draft_rows:
# #             by_date[r["Date"]] = by_date.get(r["Date"], 0.0) + float(r.get("Hours") or 0.0)

# #     n = 0
# #     cur = date.today()
# #     for _ in range(365):
# #         iso = cur.isoformat()
# #         if by_date.get(iso, 0.0) >= goal:
# #             n += 1
# #             cur = cur - timedelta(days=1)
# #         else:
# #             break
# #     return n

# # # -------------------------------------------------------------
# # # Session bootstrap
# # # -------------------------------------------------------------
# # def ensure_session_state():
# #     if "active_email" not in st.session_state:
# #         # Try restore from disk
# #         email, ph = load_session()
# #         st.session_state.active_email = email
# #         st.session_state.pass_hash = ph
# #         if email and ph:
# #             data, onfile_ph = load_user(email)
# #             if data and onfile_ph == ph:
# #                 st.session_state.data = data
# #             else:
# #                 st.session_state.data = None
# #         else:
# #             st.session_state.data = None
# #     if "auto_save" not in st.session_state:
# #         st.session_state.auto_save = True
# #     if "draft_rows" not in st.session_state:
# #         if st.session_state.active_email:
# #             st.session_state.draft_rows = load_draft(st.session_state.active_email)
# #         else:
# #             st.session_state.draft_rows = []
# #     if "timers" not in st.session_state:
# #         st.session_state.timers = {}  # subject timers

# # def persist_all():
# #     # save user + draft atomically on each important change
# #     if st.session_state.active_email and st.session_state.pass_hash and st.session_state.data:
# #         save_user(st.session_state.active_email, st.session_state.data, st.session_state.pass_hash)
# #         save_draft(st.session_state.active_email, st.session_state.draft_rows)

# # # -------------------------------------------------------------
# # # Auth UI
# # # -------------------------------------------------------------
# # def view_login():
# #     st.header("Login to StudyTracker (Local)")
# #     email = st.text_input("Email", key="login_email")
# #     password = st.text_input("Password", type="password", key="login_pass")
# #     c1, c2 = st.columns(2)
# #     with c1:
# #         if st.button("Login", use_container_width=True):
# #             if not email or not password:
# #                 st.error("Please enter email and password")
# #                 return
# #             data, ph = load_user(email)
# #             if data is None or ph is None:
# #                 st.error("Account not found. Please sign up.")
# #                 return
# #             if not check_bcrypt(password, ph):
# #                 st.error("Incorrect password")
# #                 return
# #             st.session_state.active_email = email
# #             st.session_state.pass_hash = ph
# #             st.session_state.data = data
# #             save_session(email, ph)
# #             st.session_state.draft_rows = load_draft(email)
# #             st.toast("Logged in")
# #             st.rerun()
# #     with c2:
# #         if st.button("Sign Up", use_container_width=True, type="secondary"):
# #             if not email or not password:
# #                 st.error("Please enter email and password")
# #                 return
# #             p = user_file(email)
# #             if os.path.exists(p):
# #                 st.error("Account already exists. Please log in.")
# #                 return
# #             phash = sha_bcrypt(password)
# #             data = default_user_data()
# #             save_user(email, data, phash)
# #             save_session(email, phash)
# #             st.session_state.active_email = email
# #             st.session_state.pass_hash = phash
# #             st.session_state.data = data
# #             st.session_state.draft_rows = []
# #             st.toast("Signed up")
# #             st.rerun()

# # # -------------------------------------------------------------
# # # KPI + Draft management
# # # -------------------------------------------------------------
# # def kpi_cards(data: UserData):
# #     # Use saved logs + draft (today) to show live KPIs
# #     logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
# #         "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
# #     ])
# #     draft_df = pd.DataFrame(st.session_state.draft_rows) if st.session_state.draft_rows else pd.DataFrame(columns=logs_df.columns)

# #     # Today totals (from draft)
# #     today = today_iso()
# #     today_hours = float(draft_df[draft_df["Date"] == today]["Hours"].sum() if not draft_df.empty else 0.0)
# #     today_completed = int(draft_df[(draft_df["Date"] == today) & (draft_df["Completed"] == True)].shape[0] if not draft_df.empty else 0)

# #     # Avg hours/day from combined saved logs (draft excluded to avoid partial duplicates)
# #     avg_hours = 0.0
# #     if not logs_df.empty:
# #         avg_hours = logs_df.groupby("Date")["Hours"].sum().mean()

# #     c1, c2, c3, c4, c5 = st.columns(5)
# #     c1.metric("Goal", f"{data.settings.dailyGoal:.2f}h")
# #     c2.metric("Today", f"{today_hours:.2f}h")
# #     c3.metric("Completed", str(today_completed))
# #     c4.metric("Streak", str(streak_days(data)))
# #     c5.metric("Avg Hours/Day", f"{avg_hours:.2f}h")

# #     st.progress(min(1.0, today_hours / max(1e-6, data.settings.dailyGoal)))

# # def upsert_draft(subject: str, hours: float, done: bool, mode: str, priority: str, notes: str, pomos: int):
# #     # Enforce rule: if hours < 3h, Completed must be False
# #     completed = bool(done and hours >= 3.0)
# #     row = {
# #         "Date": today_iso(),
# #         "Subject": subject,
# #         "Completed": completed,
# #         "Hours": float(hours),
# #         "Notes": notes or "",
# #         "Priority": priority,
# #         "Mode": mode,
# #         "Pomodoros": int(pomos),
# #         "XP": int(xp_from(hours, completed, priority))
# #     }
# #     found = False
# #     for r in st.session_state.draft_rows:
# #         if r["Subject"] == subject and r["Date"] == row["Date"]:
# #             r.update(row)
# #             found = True
# #             break
# #     if not found:
# #         st.session_state.draft_rows.append(row)
# #     if st.session_state.auto_save:
# #         persist_all()

# # def clear_draft_today():
# #     today = today_iso()
# #     st.session_state.draft_rows = [r for r in st.session_state.draft_rows if r["Date"] != today]
# #     if st.session_state.auto_save:
# #         persist_all()

# # def save_log(data: UserData):
# #     today = today_iso()
# #     todays = [r for r in st.session_state.draft_rows if r["Date"] == today and (r["Hours"] > 0 or r["Notes"])]
# #     if not todays:
# #         st.toast("No data to save")
# #         return
# #     for r in todays:
# #         data.logs.append(LogRow(**r))
# #         data.settings.xp += int(r.get("XP", 0))
# #     # Keep draft for future edits but clear today's if desired:
# #     st.session_state.draft_rows = [r for r in st.session_state.draft_rows if r["Date"] != today]
# #     # Badges
# #     total_hours = sum((float(l.Hours or 0.0) for l in data.logs), 0.0)
# #     badges = set(data.settings.badges or [])
# #     for threshold, name in [(10, "Rookie"), (50, "Committed"), (100, "Centurion"), (200, "Marathoner")]:
# #         if total_hours >= threshold:
# #             badges.add(name)
# #     data.settings.badges = sorted(badges)
# #     persist_all()
# #     st.toast("Log saved")

# # # -------------------------------------------------------------
# # # Today View
# # # -------------------------------------------------------------
# # def timer_ui(subject: str, data: UserData):
# #     tmap = st.session_state.timers
# #     if subject not in tmap:
# #         tmap[subject] = {"running": False, "start": 0.0, "elapsed_ms": 0, "pomos": 0, "use_pomo": False}
# #     t = tmap[subject]

# #     def hms(ms: int) -> str:
# #         s = ms // 1000
# #         h = s // 3600
# #         s -= h*3600
# #         m = s // 60
# #         s -= m*60
# #         return f"{h:02d}:{m:02d}:{s:02d}"

# #     c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
# #     c1.write(f"⏱️ {hms(int(t['elapsed_ms']))}")
# #     start = c2.button("Start", key=f"start_{subject}")
# #     pause = c3.button("Pause", key=f"pause_{subject}")
# #     stop = c4.button("Stop", key=f"stop_{subject}")
# #     t["use_pomo"] = c5.toggle("Pomodoro", value=t.get("use_pomo", False), key=f"pomo_{subject}")

# #     now = time.time()
# #     if t["running"]:
# #         t["elapsed_ms"] = int((now - t["start"]) * 1000)

# #     if start and not t["running"]:
# #         t["running"] = True
# #         t["start"] = now - (t["elapsed_ms"] / 1000.0)
# #         st.toast(f"Timer started: {subject}")

# #     if pause and t["running"]:
# #         t["running"] = False
# #         t["elapsed_ms"] = int((now - t["start"]) * 1000)
# #         st.toast(f"Timer paused: {subject}")

# #     if t["use_pomo"]:
# #         work = (data.settings.pomoWork or 25) * 60 * 1000
# #         brk = (data.settings.pomoBreak or 5) * 60 * 1000
# #         cyc = work + brk
# #         completed = t["elapsed_ms"] // cyc
# #         if completed > t["pomos"]:
# #             t["pomos"] = int(completed)
# #             st.toast(f"Pomodoro complete: {subject}")
# #     st.caption(f"🍅 {t['pomos']}")

# #     if stop and (t["running"] or t["elapsed_ms"] > 0):
# #         if t["running"]:
# #             t["elapsed_ms"] = int((now - t["start"]) * 1000)
# #         t["running"] = False
# #         st.toast(f"Timer stopped: {subject}")

# #     # Suggested hours capped 10h
# #     hrs = min(10.0, round(t["elapsed_ms"] / 3600000.0, 2))
# #     return hrs, t["pomos"]

# # def reorder_ui(data: UserData):
# #     d = cycle_day(data.settings)
# #     st.subheader(f"Today's Subjects — Day {d} of 8")
# #     subjects = data.plan.get(d, [])
# #     key = f"day{d}"
# #     current = data.taskOrder.get(key, subjects[:])

# #     st.caption("Reorder with arrow buttons and Save.")

# #     if "order_state" not in st.session_state:
# #         st.session_state.order_state = {}
# #     arr = st.session_state.order_state.setdefault(key, current[:])

# #     def move(i, di):
# #         j = i + di
# #         if 0 <= j < len(arr):
# #             arr[i], arr[j] = arr[j], arr[i]

# #     for i, s in enumerate(arr):
# #         c1, c2, c3 = st.columns([8,1,1])
# #         c1.write(f"- {s}")
# #         if c2.button("↑", key=f"up_{key}_{i}"):
# #             move(i, -1)
# #             st.rerun()
# #         if c3.button("↓", key=f"dn_{key}_{i}"):
# #             move(i, 1)
# #             st.rerun()

# #     if st.button("Save Order", key=f"save_order_{key}", type="primary"):
# #         data.taskOrder[key] = arr[:]
# #         st.toast("Order saved")
# #         if st.session_state.auto_save:
# #             persist_all()

# # def today_view(data: UserData):
# #     st.header("Today's Plan")
# #     st.caption("Hours up to 10h/subject. Completed only if hours ≥ 3h.")
# #     kpi_cards(data)

# #     a1, a2, a3, a4, a5 = st.columns(5)
# #     if a1.button("💾 Save Log", type="primary", use_container_width=True):
# #         save_log(data)
# #         st.rerun()
# #     if a2.button("➡ Next Day", use_container_width=True):
# #         s = datetime.fromisoformat(data.settings.startDate).date() if data.settings.startDate else date.today()
# #         data.settings.startDate = (s - timedelta(days=1)).isoformat()
# #         if st.session_state.auto_save:
# #             persist_all()
# #         st.rerun()
# #     if a3.button("🔄 Refresh", use_container_width=True):
# #         st.rerun()
# #     if a4.button("🧹 Clear Today Draft", use_container_width=True):
# #         clear_draft_today()
# #         st.rerun()
# #     st.session_state.auto_save = a5.toggle("Auto-save on Change", value=st.session_state.auto_save)

# #     st.divider()
# #     reorder_ui(data)
# #     st.divider()

# #     for subject in subjects_for_today(data):
# #         with st.container(border=True):
# #             st.subheader(subject)
# #             st.caption("Enter hours and details. Completed requires ≥ 3h.")
# #             # Timer and suggested hours
# #             sugg, pomos = timer_ui(subject, data)

# #             c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,1.2])
# #             hours = c1.number_input("Hours", min_value=0.0, max_value=10.0, step=0.25, value=float(sugg), key=f"hrs_{subject}")
# #             mode = c2.selectbox("Mode", ["Deep Work","Review","Practice","PYQs","Test"], key=f"mode_{subject}")
# #             priority = c3.selectbox("Priority", ["Low","Medium","High"], index=1, key=f"prio_{subject}")
# #             # The checkbox value is advisory; true completion is enforced (hours>=3)
# #             done_checkbox = c4.checkbox("Completed (≥3h)", key=f"done_{subject}")

# #             notes = st.text_area("Notes", value="", height=80, key=f"notes_{subject}", placeholder="Concepts, mistakes, formulas, tasks...")

# #             # Upsert draft and persist
# #             upsert_draft(subject, hours, done_checkbox, mode, priority, notes, pomos)

# # # -------------------------------------------------------------
# # # Syllabus View
# # # -------------------------------------------------------------
# # def syllabus_view(data: UserData):
# #     st.header("DA Syllabus Map")
# #     for cat, items in SYLLABUS.items():
# #         with st.expander(cat, expanded=False):
# #             comp = sum(1 for t in items if data.syllabusProgress.get(f"{cat}:{t}", False))
# #             st.caption(f"{comp}/{len(items)} completed")
# #             for t in items:
# #                 k = f"{cat}:{t}"
# #                 val = st.checkbox(t, value=data.syllabusProgress.get(k, False), key=f"sy_{k}")
# #                 data.syllabusProgress[k] = bool(val)
# #     if st.session_state.auto_save:
# #         persist_all()

# # # -------------------------------------------------------------
# # # Logs View (with per-row delete + undo)
# # # -------------------------------------------------------------
# # def logs_view(data: UserData):
# #     st.header("Logs")

# #     logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
# #         "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
# #     ])

# #     if logs_df.empty:
# #         st.info("No logs yet.")
# #     else:
# #         st.dataframe(logs_df, use_container_width=True, height=420)

# #     # Row delete controls
# #     st.subheader("Manage Rows")
# #     if "deleted_buffer" not in st.session_state:
# #         st.session_state.deleted_buffer = []

# #     c1, c2, c3, c4, c5 = st.columns(5)
# #     # Delete by selecting an exact row index
# #     del_idx = c1.number_input("Row index to delete", min_value=0, step=1, value=0 if not logs_df.empty else 0, disabled=logs_df.empty)
# #     if c2.button("Delete Row", type="secondary", disabled=logs_df.empty):
# #         if 0 <= del_idx < len(data.logs):
# #             removed = data.logs.pop(int(del_idx))
# #             st.session_state.deleted_buffer.append(removed)
# #             st.toast(f"Deleted row {del_idx}")
# #             if st.session_state.auto_save:
# #                 persist_all()
# #             st.rerun()

# #     if c3.button("Undo Last Delete", disabled=(len(st.session_state.deleted_buffer)==0)):
# #         rec = st.session_state.deleted_buffer.pop()
# #         data.logs.append(rec)
# #         st.toast("Undo successful")
# #         if st.session_state.auto_save:
# #             persist_all()
# #         st.rerun()

# #     # Export / Import
# #     st.subheader("Import/Export")
# #     e1, e2, e3, e4 = st.columns(4)

# #     with e1:
# #         if st.button("⬇ Export CSV", use_container_width=True):
# #             csv = logs_df.to_csv(index=False)
# #             st.download_button("Download study_log.csv", data=csv, file_name="study_log.csv", mime="text/csv", use_container_width=True)

# #     with e2:
# #         # Save JSON backup (full)
# #         if st.button("💾 Save JSON (Full Backup)", use_container_width=True):
# #             payload = {
# #                 "email": st.session_state.active_email,
# #                 "data": {
# #                     "settings": asdict(st.session_state.data.settings),
# #                     "plan": st.session_state.data.plan,
# #                     "logs": [asdict(r) for r in st.session_state.data.logs],
# #                     "syllabusProgress": st.session_state.data.syllabusProgress,
# #                     "taskOrder": st.session_state.data.taskOrder
# #                 }
# #             }
# #             st.download_button("Download data.json", data=json.dumps(payload, indent=2),
# #                                file_name="data.json", mime="application/json", use_container_width=True)

# #     with e3:
# #         # Save Excel (XLSX) — logs only
# #         if st.button("💾 Save Excel (Logs)", use_container_width=True):
# #             xls = pd.ExcelWriter("study_log.xlsx", engine="openpyxl")
# #             (logs_df if not logs_df.empty else pd.DataFrame(columns=[
# #                 "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
# #             ])).to_excel(xls, index=False, sheet_name="Logs")
# #             xls.close()
# #             with open("study_log.xlsx", "rb") as f:
# #                 st.download_button("Download study_log.xlsx", data=f.read(), file_name="study_log.xlsx",
# #                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
# #             os.remove("study_log.xlsx")

# #     with e4:
# #         uploaded = st.file_uploader("⬆ Import data.json", type=["json"], label_visibility="collapsed")
# #         if uploaded is not None and st.button("Import", use_container_width=True):
# #             try:
# #                 parsed = json.loads(uploaded.read().decode("utf-8"))
# #                 if "data" not in parsed:
# #                     st.error("Invalid data.json format")
# #                 else:
# #                     d = parsed["data"]
# #                     settings = Settings(**d.get("settings", {}))
# #                     plan = {int(k): v for k, v in d.get("plan", {}).items()}
# #                     logs = [LogRow(**r) for r in d.get("logs", [])]
# #                     syllabus = d.get("syllabusProgress", {})
# #                     order = d.get("taskOrder", {})
# #                     st.session_state.data = UserData(settings, plan, logs, syllabus, order)
# #                     st.toast("Imported data.json")
# #                     persist_all()
# #                     st.rerun()
# #             except Exception as e:
# #                 st.error(f"Import failed: {e}")

# # # -------------------------------------------------------------
# # # Dashboard View
# # # -------------------------------------------------------------
# # def dashboard_view(data: UserData):
# #     st.header("Dashboard")
# #     logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
# #         "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
# #     ])

# #     cA, cB = st.columns(2)
# #     with cA:
# #         st.subheader("Daily Hours")
# #         if not logs_df.empty:
# #             ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum().sort_values("Date")
# #             chart = alt.Chart(ddf).mark_line(point=True).encode(
# #                 x="Date:T", y=alt.Y("Hours:Q", scale=alt.Scale(domain=[0, max(3.0, float(ddf['Hours'].max()))])),
# #                 tooltip=["Date","Hours"]
# #             ).properties(height=260)
# #             st.altair_chart(chart, use_container_width=True)
# #         else:
# #             st.info("No data")

# #     with cB:
# #         st.subheader("Hours by Subject")
# #         if not logs_df.empty:
# #             sdf = logs_df.groupby("Subject", as_index=False)["Hours"].sum().sort_values("Hours", ascending=False)
# #             chart = alt.Chart(sdf).mark_bar().encode(
# #                 x="Hours:Q", y=alt.Y("Subject:N", sort="-x"),
# #                 tooltip=["Subject","Hours"]
# #             ).properties(height=260)
# #             st.altair_chart(chart, use_container_width=True)
# #         else:
# #             st.info("No data")

# #     cC, cD = st.columns(2)
# #     with cC:
# #         st.subheader("Completion Share")
# #         if not logs_df.empty:
# #             total = len(logs_df)
# #             comp = int((logs_df["Completed"]==True).sum())
# #             pie_df = pd.DataFrame({"Status":["Completed","Pending"],"Count":[comp, max(0,total-comp)]})
# #             pie = alt.Chart(pie_df).mark_arc().encode(
# #                 theta="Count:Q", color="Status:N", tooltip=["Status","Count"]
# #             ).properties(height=260)
# #             st.altair_chart(pie, use_container_width=True)
# #         else:
# #             st.info("No data")

# #     with cD:
# #         st.subheader("Last 7 Days")
# #         if not logs_df.empty:
# #             last7 = [(date.today()-timedelta(days=i)).isoformat() for i in range(6,-1,-1)]
# #             ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum()
# #             merged = pd.DataFrame({"Date": last7}).merge(ddf, on="Date", how="left").fillna({"Hours":0})
# #             chart = alt.Chart(merged).mark_bar().encode(
# #                 x=alt.X("Date:T", sort=None),
# #                 y=alt.Y("Hours:Q", scale=alt.Scale(domain=[0, max(3.0, float(merged['Hours'].max()))])),
# #                 tooltip=["Date","Hours"]
# #             ).properties(height=260)
# #             st.altair_chart(chart, use_container_width=True)
# #         else:
# #             st.info("No data")

# # # -------------------------------------------------------------
# # # Settings View
# # # -------------------------------------------------------------
# # def plan_editor_view(data: UserData):
# #     st.subheader("8-Day Plan Editor")
# #     for d in range(1, 9):
# #         with st.expander(f"Day {d}", expanded=(d == cycle_day(data.settings))):
# #             txt = st.text_area(
# #                 f"Subjects (one per line) — Day {d}",
# #                 value="\n".join(data.plan.get(d, [])),
# #                 height=120,
# #                 key=f"plan_day_{d}"
# #             )
# #             if st.button(f"Save Day {d}", key=f"s_day_{d}"):
# #                 arr = [x.strip() for x in txt.split("\n") if x.strip()]
# #                 data.plan[d] = arr
# #                 st.toast(f"Saved Day {d}")
# #                 if st.session_state.auto_save:
# #                     persist_all()

# # def settings_view(data: UserData):
# #     st.header("Settings")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.subheader("General")
# #         g1, g2 = st.columns(2)
# #         daily_goal = g1.number_input("Daily Goal (hrs)", min_value=0.0, step=0.5, value=float(data.settings.dailyGoal))
# #         theme = g2.selectbox("Theme", ["dark","light"], index=(0 if data.settings.theme=="dark" else 1))
# #         p1, p2 = st.columns(2)
# #         pomo_work = p1.number_input("Pomodoro Work (min)", min_value=1, step=1, value=int(data.settings.pomoWork))
# #         pomo_break = p2.number_input("Pomodoro Break (min)", min_value=1, step=1, value=int(data.settings.pomoBreak))

# #         s1, s2, s3 = st.columns(3)
# #         if s1.button("Save", type="primary"):
# #             data.settings.dailyGoal = float(daily_goal)
# #             data.settings.theme = theme
# #             data.settings.pomoWork = int(pomo_work)
# #             data.settings.pomoBreak = int(pomo_break)
# #             st.toast("Settings saved")
# #             if st.session_state.auto_save:
# #                 persist_all()

# #         if s2.button("Export Config+Plan"):
# #             payload = {"settings": asdict(data.settings), "plan": data.plan}
# #             st.download_button("Download config_plan.json", data=json.dumps(payload, indent=2),
# #                                file_name="config_plan.json", mime="application/json")

# #         if s3.button("Reset 8-Day Plan", type="secondary"):
# #             data.plan = json.loads(json.dumps(DEFAULT_PLAN))
# #             st.toast("Plan reset")
# #             if st.session_state.auto_save:
# #                 persist_all()

# #     with col2:
# #         plan_editor_view(data)

# # # -------------------------------------------------------------
# # # Header
# # # -------------------------------------------------------------
# # def header_bar(data: UserData):
# #     left, right = st.columns([3,2])
# #     with left:
# #         st.markdown("### StudyTracker · Data Science & AI")
# #         st.caption("GATE DA 2026")
# #     with right:
# #         st.write(f"👤 {st.session_state.active_email or ''}")
# #         st.write(f"📅 {today_iso()}")
# #         hb1, hb2, hb3 = st.columns(3)
# #         if hb1.button("Logout", use_container_width=True):
# #             persist_all()
# #             save_session(None, None)
# #             st.session_state.active_email = None
# #             st.session_state.pass_hash = None
# #             st.session_state.data = None
# #             st.session_state.draft_rows = []
# #             st.session_state.timers = {}
# #             st.toast("Logged out")
# #             st.rerun()
# #         if hb2.button("🌗 Theme", use_container_width=True):
# #             data.settings.theme = "light" if data.settings.theme == "dark" else "dark"
# #             st.toast("Theme switched")
# #             if st.session_state.auto_save:
# #                 persist_all()
# #         if hb3.button("🎯 Focus", use_container_width=True):
# #             st.sidebar.write("Focus mode toggled. Collapse sidebar if desired.")
# #             st.toast("Focus toggled")
# #     st.divider()

# # # -------------------------------------------------------------
# # # Main
# # # -------------------------------------------------------------
# # def main():
# #     ensure_session_state()

# #     if not st.session_state.active_email or not st.session_state.data:
# #         view_login()
# #         return

# #     data: UserData = st.session_state.data
# #     header_bar(data)

# #     t1, t2, t3, t4, t5 = st.tabs(["Today", "Syllabus", "Log", "Dashboard", "Settings"])
# #     with t1:
# #         today_view(data)
# #     with t2:
# #         syllabus_view(data)
# #     with t3:
# #         logs_view(data)
# #     with t4:
# #         dashboard_view(data)
# #     with t5:
# #         settings_view(data)

# #     # Final auto-persist
# #     if st.session_state.auto_save:
# #         persist_all()

# # if __name__ == "__main__":
# #     main()
