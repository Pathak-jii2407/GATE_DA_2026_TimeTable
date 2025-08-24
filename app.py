# app.py
import json
import os
import time
import bcrypt
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

def user_file(email: str) -> str:
    safe = email.replace("@", "_at_").replace(".", "_")
    return os.path.join(USER_DIR, f"{safe}.json")

def draft_file(email: str) -> str:
    safe = email.replace("@", "_at_").replace(".", "_")
    return os.path.join(DRAFT_DIR, f"{safe}_draft.json")

AUTH_SESSION_FILE = os.path.join(AUTH_DIR, "session.json")  # persist active_email + passhash

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
        "k-NN", "Naive Bayes", "LDA",
        "SVM", "Decision Trees",
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

@dataclass
class UserData:
    settings: Settings
    plan: Dict[int, List[str]]
    logs: List[LogRow]
    syllabusProgress: Dict[str, bool]
    taskOrder: Dict[str, List[str]]

# -------------------------------------------------------------
# Utils
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
        taskOrder={}
    )

def load_user(email: str) -> Tuple[Optional[UserData], Optional[str]]:
    p = user_file(email)
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        passhash = raw.get("passHash")
        d = raw.get("data", {})
        settings = Settings(**d.get("settings", {}))
        plan = {int(k): v for k, v in d.get("plan", {}).items()}
        logs = [LogRow(**r) for r in d.get("logs", [])]
        syllabus = d.get("syllabusProgress", {})
        order = d.get("taskOrder", {})
        return UserData(settings, plan, logs, syllabus, order), passhash
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
            "taskOrder": data.taskOrder
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
    goal = data.settings.dailyGoal
    by_date: Dict[str, float] = {}
    for r in data.logs:
        by_date[r.Date] = by_date.get(r.Date, 0.0) + float(r.Hours or 0.0)
    # Include today's draft in streak computation preview
    if st.session_state.active_email:
        for r in st.session_state.draft_rows:
            by_date[r["Date"]] = by_date.get(r["Date"], 0.0) + float(r.get("Hours") or 0.0)

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
        # Try restore from disk
        email, ph = load_session()
        st.session_state.active_email = email
        st.session_state.pass_hash = ph
        if email and ph:
            data, onfile_ph = load_user(email)
            if data and onfile_ph == ph:
                st.session_state.data = data
            else:
                st.session_state.data = None
        else:
            st.session_state.data = None
    if "auto_save" not in st.session_state:
        st.session_state.auto_save = True
    if "draft_rows" not in st.session_state:
        if st.session_state.active_email:
            st.session_state.draft_rows = load_draft(st.session_state.active_email)
        else:
            st.session_state.draft_rows = []
    if "timers" not in st.session_state:
        st.session_state.timers = {}  # subject timers

def persist_all():
    # save user + draft atomically on each important change
    if st.session_state.active_email and st.session_state.pass_hash and st.session_state.data:
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
            st.toast("Logged in")
            st.rerun()
    with c2:
        if st.button("Sign Up", use_container_width=True, type="secondary"):
            if not email or not password:
                st.error("Please enter email and password")
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
            st.toast("Signed up")
            st.rerun()

# -------------------------------------------------------------
# KPI + Draft management
# -------------------------------------------------------------
def kpi_cards(data: UserData):
    # Use saved logs + draft (today) to show live KPIs
    logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
        "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
    ])
    draft_df = pd.DataFrame(st.session_state.draft_rows) if st.session_state.draft_rows else pd.DataFrame(columns=logs_df.columns)

    # Today totals (from draft)
    today = today_iso()
    today_hours = float(draft_df[draft_df["Date"] == today]["Hours"].sum() if not draft_df.empty else 0.0)
    today_completed = int(draft_df[(draft_df["Date"] == today) & (draft_df["Completed"] == True)].shape[0] if not draft_df.empty else 0)

    # Avg hours/day from combined saved logs (draft excluded to avoid partial duplicates)
    avg_hours = 0.0
    if not logs_df.empty:
        avg_hours = logs_df.groupby("Date")["Hours"].sum().mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Goal", f"{data.settings.dailyGoal:.2f}h")
    c2.metric("Today", f"{today_hours:.2f}h")
    c3.metric("Completed", str(today_completed))
    c4.metric("Streak", str(streak_days(data)))
    c5.metric("Avg Hours/Day", f"{avg_hours:.2f}h")

    st.progress(min(1.0, today_hours / max(1e-6, data.settings.dailyGoal)))

def upsert_draft(subject: str, hours: float, done: bool, mode: str, priority: str, notes: str, pomos: int):
    # Enforce rule: if hours < 3h, Completed must be False
    completed = bool(done and hours >= 3.0)
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
    found = False
    for r in st.session_state.draft_rows:
        if r["Subject"] == subject and r["Date"] == row["Date"]:
            r.update(row)
            found = True
            break
    if not found:
        st.session_state.draft_rows.append(row)
    if st.session_state.auto_save:
        persist_all()

def clear_draft_today():
    today = today_iso()
    st.session_state.draft_rows = [r for r in st.session_state.draft_rows if r["Date"] != today]
    if st.session_state.auto_save:
        persist_all()

def save_log(data: UserData):
    today = today_iso()
    todays = [r for r in st.session_state.draft_rows if r["Date"] == today and (r["Hours"] > 0 or r["Notes"])]
    if not todays:
        st.toast("No data to save")
        return
    for r in todays:
        data.logs.append(LogRow(**r))
        data.settings.xp += int(r.get("XP", 0))
    # Keep draft for future edits but clear today's if desired:
    st.session_state.draft_rows = [r for r in st.session_state.draft_rows if r["Date"] != today]
    # Badges
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
    tmap = st.session_state.timers
    if subject not in tmap:
        tmap[subject] = {"running": False, "start": 0.0, "elapsed_ms": 0, "pomos": 0, "use_pomo": False}
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

    # Suggested hours capped 10h
    hrs = min(10.0, round(t["elapsed_ms"] / 3600000.0, 2))
    return hrs, t["pomos"]

def reorder_ui(data: UserData):
    d = cycle_day(data.settings)
    st.subheader(f"Today's Subjects — Day {d} of 8")
    subjects = data.plan.get(d, [])
    key = f"day{d}"
    current = data.taskOrder.get(key, subjects[:])

    st.caption("Reorder with arrow buttons and Save.")

    if "order_state" not in st.session_state:
        st.session_state.order_state = {}
    arr = st.session_state.order_state.setdefault(key, current[:])

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
        if st.session_state.auto_save:
            persist_all()

def today_view(data: UserData):
    st.header("Today's Plan")
    st.caption("Hours up to 10h/subject. Completed only if hours ≥ 3h.")
    kpi_cards(data)

    a1, a2, a3, a4, a5 = st.columns(5)
    if a1.button("💾 Save Log", type="primary", use_container_width=True):
        save_log(data)
        st.rerun()
    if a2.button("➡ Next Day", use_container_width=True):
        s = datetime.fromisoformat(data.settings.startDate).date() if data.settings.startDate else date.today()
        data.settings.startDate = (s - timedelta(days=1)).isoformat()
        if st.session_state.auto_save:
            persist_all()
        st.rerun()
    if a3.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    if a4.button("🧹 Clear Today Draft", use_container_width=True):
        clear_draft_today()
        st.rerun()
    st.session_state.auto_save = a5.toggle("Auto-save on Change", value=st.session_state.auto_save)

    st.divider()
    reorder_ui(data)
    st.divider()

    for subject in subjects_for_today(data):
        with st.container(border=True):
            st.subheader(subject)
            st.caption("Enter hours and details. Completed requires ≥ 3h.")
            # Timer and suggested hours
            sugg, pomos = timer_ui(subject, data)

            c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,1.2])
            hours = c1.number_input("Hours", min_value=0.0, max_value=10.0, step=0.25, value=float(sugg), key=f"hrs_{subject}")
            mode = c2.selectbox("Mode", ["Deep Work","Review","Practice","PYQs","Test"], key=f"mode_{subject}")
            priority = c3.selectbox("Priority", ["Low","Medium","High"], index=1, key=f"prio_{subject}")
            # The checkbox value is advisory; true completion is enforced (hours>=3)
            done_checkbox = c4.checkbox("Completed (≥3h)", key=f"done_{subject}")

            notes = st.text_area("Notes", value="", height=80, key=f"notes_{subject}", placeholder="Concepts, mistakes, formulas, tasks...")

            # Upsert draft and persist
            upsert_draft(subject, hours, done_checkbox, mode, priority, notes, pomos)

# -------------------------------------------------------------
# Syllabus View
# -------------------------------------------------------------
def syllabus_view(data: UserData):
    st.header("DA Syllabus Map")
    for cat, items in SYLLABUS.items():
        with st.expander(cat, expanded=False):
            comp = sum(1 for t in items if data.syllabusProgress.get(f"{cat}:{t}", False))
            st.caption(f"{comp}/{len(items)} completed")
            for t in items:
                k = f"{cat}:{t}"
                val = st.checkbox(t, value=data.syllabusProgress.get(k, False), key=f"sy_{k}")
                data.syllabusProgress[k] = bool(val)
    if st.session_state.auto_save:
        persist_all()

# -------------------------------------------------------------
# Logs View (with per-row delete + undo)
# -------------------------------------------------------------
def logs_view(data: UserData):
    st.header("Logs")

    logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
        "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
    ])

    if logs_df.empty:
        st.info("No logs yet.")
    else:
        st.dataframe(logs_df, use_container_width=True, height=420)

    # Row delete controls
    st.subheader("Manage Rows")
    if "deleted_buffer" not in st.session_state:
        st.session_state.deleted_buffer = []

    c1, c2, c3, c4, c5 = st.columns(5)
    # Delete by selecting an exact row index
    del_idx = c1.number_input("Row index to delete", min_value=0, step=1, value=0 if not logs_df.empty else 0, disabled=logs_df.empty)
    if c2.button("Delete Row", type="secondary", disabled=logs_df.empty):
        if 0 <= del_idx < len(data.logs):
            removed = data.logs.pop(int(del_idx))
            st.session_state.deleted_buffer.append(removed)
            st.toast(f"Deleted row {del_idx}")
            if st.session_state.auto_save:
                persist_all()
            st.rerun()

    if c3.button("Undo Last Delete", disabled=(len(st.session_state.deleted_buffer)==0)):
        rec = st.session_state.deleted_buffer.pop()
        data.logs.append(rec)
        st.toast("Undo successful")
        if st.session_state.auto_save:
            persist_all()
        st.rerun()

    # Export / Import
    st.subheader("Import/Export")
    e1, e2, e3, e4 = st.columns(4)

    with e1:
        if st.button("⬇ Export CSV", use_container_width=True):
            csv = logs_df.to_csv(index=False)
            st.download_button("Download study_log.csv", data=csv, file_name="study_log.csv", mime="text/csv", use_container_width=True)

    with e2:
        # Save JSON backup (full)
        if st.button("💾 Save JSON (Full Backup)", use_container_width=True):
            payload = {
                "email": st.session_state.active_email,
                "data": {
                    "settings": asdict(st.session_state.data.settings),
                    "plan": st.session_state.data.plan,
                    "logs": [asdict(r) for r in st.session_state.data.logs],
                    "syllabusProgress": st.session_state.data.syllabusProgress,
                    "taskOrder": st.session_state.data.taskOrder
                }
            }
            st.download_button("Download data.json", data=json.dumps(payload, indent=2),
                               file_name="data.json", mime="application/json", use_container_width=True)

    with e3:
        # Save Excel (XLSX) — logs only
        if st.button("💾 Save Excel (Logs)", use_container_width=True):
            xls = pd.ExcelWriter("study_log.xlsx", engine="openpyxl")
            (logs_df if not logs_df.empty else pd.DataFrame(columns=[
                "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
            ])).to_excel(xls, index=False, sheet_name="Logs")
            xls.close()
            with open("study_log.xlsx", "rb") as f:
                st.download_button("Download study_log.xlsx", data=f.read(), file_name="study_log.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            os.remove("study_log.xlsx")

    with e4:
        uploaded = st.file_uploader("⬆ Import data.json", type=["json"], label_visibility="collapsed")
        if uploaded is not None and st.button("Import", use_container_width=True):
            try:
                parsed = json.loads(uploaded.read().decode("utf-8"))
                if "data" not in parsed:
                    st.error("Invalid data.json format")
                else:
                    d = parsed["data"]
                    settings = Settings(**d.get("settings", {}))
                    plan = {int(k): v for k, v in d.get("plan", {}).items()}
                    logs = [LogRow(**r) for r in d.get("logs", [])]
                    syllabus = d.get("syllabusProgress", {})
                    order = d.get("taskOrder", {})
                    st.session_state.data = UserData(settings, plan, logs, syllabus, order)
                    st.toast("Imported data.json")
                    persist_all()
                    st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

# -------------------------------------------------------------
# Dashboard View
# -------------------------------------------------------------
def dashboard_view(data: UserData):
    st.header("Dashboard")
    logs_df = pd.DataFrame([asdict(r) for r in data.logs]) if data.logs else pd.DataFrame(columns=[
        "Date","Subject","Completed","Hours","Notes","Priority","Mode","Pomodoros","XP"
    ])

    cA, cB = st.columns(2)
    with cA:
        st.subheader("Daily Hours")
        if not logs_df.empty:
            ddf = logs_df.groupby("Date", as_index=False)["Hours"].sum().sort_values("Date")
            chart = alt.Chart(ddf).mark_line(point=True).encode(
                x="Date:T", y=alt.Y("Hours:Q", scale=alt.Scale(domain=[0, max(3.0, float(ddf['Hours'].max()))])),
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
                x="Hours:Q", y=alt.Y("Subject:N", sort="-x"),
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
                if st.session_state.auto_save:
                    persist_all()

def settings_view(data: UserData):
    st.header("Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("General")
        g1, g2 = st.columns(2)
        daily_goal = g1.number_input("Daily Goal (hrs)", min_value=0.0, step=0.5, value=float(data.settings.dailyGoal))
        theme = g2.selectbox("Theme", ["dark","light"], index=(0 if data.settings.theme=="dark" else 1))
        p1, p2 = st.columns(2)
        pomo_work = p1.number_input("Pomodoro Work (min)", min_value=1, step=1, value=int(data.settings.pomoWork))
        pomo_break = p2.number_input("Pomodoro Break (min)", min_value=1, step=1, value=int(data.settings.pomoBreak))

        s1, s2, s3 = st.columns(3)
        if s1.button("Save", type="primary"):
            data.settings.dailyGoal = float(daily_goal)
            data.settings.theme = theme
            data.settings.pomoWork = int(pomo_work)
            data.settings.pomoBreak = int(pomo_break)
            st.toast("Settings saved")
            if st.session_state.auto_save:
                persist_all()

        if s2.button("Export Config+Plan"):
            payload = {"settings": asdict(data.settings), "plan": data.plan}
            st.download_button("Download config_plan.json", data=json.dumps(payload, indent=2),
                               file_name="config_plan.json", mime="application/json")

        if s3.button("Reset 8-Day Plan", type="secondary"):
            data.plan = json.loads(json.dumps(DEFAULT_PLAN))
            st.toast("Plan reset")
            if st.session_state.auto_save:
                persist_all()

    with col2:
        plan_editor_view(data)

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
def header_bar(data: UserData):
    left, right = st.columns([3,2])
    with left:
        st.markdown("### StudyTracker · Data Science & AI")
        st.caption("GATE DA 2026")
    with right:
        st.write(f"👤 {st.session_state.active_email or ''}")
        st.write(f"📅 {today_iso()}")
        hb1, hb2, hb3 = st.columns(3)
        if hb1.button("Logout", use_container_width=True):
            persist_all()
            save_session(None, None)
            st.session_state.active_email = None
            st.session_state.pass_hash = None
            st.session_state.data = None
            st.session_state.draft_rows = []
            st.session_state.timers = {}
            st.toast("Logged out")
            st.rerun()
        if hb2.button("🌗 Theme", use_container_width=True):
            data.settings.theme = "light" if data.settings.theme == "dark" else "dark"
            st.toast("Theme switched")
            if st.session_state.auto_save:
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

    if not st.session_state.active_email or not st.session_state.data:
        view_login()
        return

    data: UserData = st.session_state.data
    header_bar(data)

    t1, t2, t3, t4, t5 = st.tabs(["Today", "Syllabus", "Log", "Dashboard", "Settings"])
    with t1:
        today_view(data)
    with t2:
        syllabus_view(data)
    with t3:
        logs_view(data)
    with t4:
        dashboard_view(data)
    with t5:
        settings_view(data)

    # Final auto-persist
    if st.session_state.auto_save:
        persist_all()

if __name__ == "__main__":
    main()
