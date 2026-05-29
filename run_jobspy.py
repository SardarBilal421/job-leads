from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import textwrap
from datetime import date as date_type, datetime

import pandas as pd

from jobspy import scrape_jobs

# ── PROFILE ───────────────────────────────────────────────────────────────────
# Edit this section to tune what counts as a "good" job for you.

PROFILE = {
    "role_keywords": [
        "full stack", "fullstack", "full-stack",
        "frontend", "front-end", "front end",
        "backend", "back-end", "back end",
        "ai developer", "ai engineer",
        "software engineer", "software developer",
        "web developer", "web engineer",
    ],
    "tech_title_keywords": [
        "react", "typescript", "next.js", "nextjs",
        "node", "node.js", "angular",
    ],
    "tech_desc_keywords": [
        "react", "typescript", "next.js", "nextjs",
        "node", "node.js", "angular", "javascript",
        "python", "django", "fastapi",
    ],
    "seniority_blocklist": [
        "senior", "sr.", "sr ", "lead", "principal", "staff",
        "manager", "director", "head of", "vp ", "vice president", "architect",
    ],
    "weights": {
        "role_title_match":   30,   # points per matching role keyword in title
        "role_title_cap":     30,   # max points from role keywords
        "tech_title_match":   10,   # points per tech keyword in title
        "tech_title_cap":     20,   # max points from tech in title
        "tech_desc_match":     3,   # points per tech keyword in description
        "tech_desc_cap":      15,   # max points from tech in description
        "is_remote_bonus":    10,   # job flagged as remote
        "recency_bonus":       5,   # date_posted == today
        "seniority_penalty": -40,   # flat penalty if any blocklist word in title
    },
}

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SEEN_JOBS_PATH = os.path.join(SCRIPT_DIR, "seen_jobs.json")
CSV_PATH       = os.path.join(SCRIPT_DIR, "jobs_results.csv")
MAX_SEEN_JOBS        = 50_000
NOTIFY_SCORE_THRESHOLD = 40   # toast fires only for jobs scoring at or above this

SEARCH_QUERY = (
    '("Full Stack" OR "Frontend" OR "Backend" OR "AI") '
    '(Developer OR Engineer) '
    '(React OR Node OR Angular OR "Next.js" OR TypeScript) '
    '-Senior -Lead -Principal -Staff -Manager'
)

# ── NOTIFICATIONS ────────────────────────────────────────────────────────────

def send_toast(title: str, message: str) -> None:
    """Fire a Windows 10/11 toast notification via PowerShell WinRT API."""
    # Escape XML special chars so the toast XML stays valid
    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    # Use PowerShell's own registered AUMID — always available on Windows 10/11
    app_id = "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe"
    xml = textwrap.dedent(f"""
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
        [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text>{_esc(title)}</text><text>{_esc(message)}</text></binding></visual></toast>')
        $toast = New-Object Windows.UI.Notifications.ToastNotification $xml
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("{app_id}").Show($toast)
    """).strip()

    try:
        subprocess.run(
            ["powershell", "-NonInteractive", "-Command", xml],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass  # never crash the main pipeline over a notification failure


# ── SCORING ───────────────────────────────────────────────────────────────────

def score_job(row: pd.Series, profile: dict) -> int:
    weights = profile["weights"]
    title       = str(row.get("title", "")       or "").lower()
    description = str(row.get("description", "") or "").lower()
    is_remote   = row.get("is_remote")
    date_posted = row.get("date_posted")

    score = 0

    # Role keywords in title
    role_pts = sum(
        weights["role_title_match"]
        for kw in profile["role_keywords"]
        if kw in title
    )
    score += min(role_pts, weights["role_title_cap"])

    # Tech keywords in title
    tech_title_pts = sum(
        weights["tech_title_match"]
        for kw in profile["tech_title_keywords"]
        if kw in title
    )
    score += min(tech_title_pts, weights["tech_title_cap"])

    # Tech keywords in description
    tech_desc_pts = sum(
        weights["tech_desc_match"]
        for kw in profile["tech_desc_keywords"]
        if kw in description
    )
    score += min(tech_desc_pts, weights["tech_desc_cap"])

    # Remote bonus
    if is_remote is True:
        score += weights["is_remote_bonus"]

    # Recency bonus
    try:
        if date_posted is not None:
            posted = (
                date_posted
                if hasattr(date_posted, "year")
                else date_type.fromisoformat(str(date_posted))
            )
            if posted >= date_type.today():
                score += weights["recency_bonus"]
    except (ValueError, TypeError):
        pass

    # Seniority penalty (flat — applied once even if multiple words match)
    if any(kw in title for kw in profile["seniority_blocklist"]):
        score += weights["seniority_penalty"]

    return max(0, min(score, 100))


# ── DEDUPLICATION ─────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["_nonnull"] = df.apply(lambda r: r.notna().sum(), axis=1)
    df = df.sort_values("_nonnull", ascending=False)

    # Pass 1: exact URL
    df = df.drop_duplicates(subset=["job_url"], keep="first")

    # Pass 2: normalized (title, company) — catches cross-site duplicates
    def _norm(s):
        if pd.isna(s):
            return ""
        s = str(s).lower().strip()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return re.sub(r"\s+", " ", s)

    df["_title_norm"]   = df["title"].apply(_norm)
    df["_company_norm"] = df["company"].apply(_norm)
    df = df.drop_duplicates(subset=["_title_norm", "_company_norm"], keep="first")

    return df.drop(columns=["_nonnull", "_title_norm", "_company_norm"]).reset_index(drop=True)


# ── SEEN-JOBS TRACKING ────────────────────────────────────────────────────────

def load_seen_jobs() -> set:
    if not os.path.exists(SEEN_JOBS_PATH):
        return set()
    try:
        with open(SEEN_JOBS_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f).get("seen_urls", []))
    except (json.JSONDecodeError, IOError):
        print("[WARN] seen_jobs.json unreadable — starting with empty seen set.")
        return set()


def save_seen_jobs(seen: set) -> None:
    urls = list(seen)
    if len(urls) > MAX_SEEN_JOBS:
        urls = urls[-MAX_SEEN_JOBS:]
    try:
        with open(SEEN_JOBS_PATH, "w", encoding="utf-8") as f:
            json.dump({"seen_urls": urls}, f, separators=(",", ":"))
    except IOError as e:
        print(f"[WARN] Could not save seen_jobs.json: {e}")


def filter_new_jobs(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    if df.empty:
        return df
    return df[~df["job_url"].isin(seen)].reset_index(drop=True)


# ── CSV OUTPUT ────────────────────────────────────────────────────────────────

def append_to_csv(df: pd.DataFrame) -> None:
    if df.empty:
        return
    path = CSV_PATH
    try:
        df.to_csv(
            path,
            mode="a",
            header=not os.path.exists(path),
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar="\\",
        )
    except PermissionError:
        # File is open in another program (e.g. Excel) — write to a timestamped fallback
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = os.path.join(SCRIPT_DIR, f"jobs_results_{ts}.csv")
        df.to_csv(
            fallback,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar="\\",
        )
        print(f"[WARN] {os.path.basename(path)} is locked — saved to {os.path.basename(fallback)} instead.")


# ── CONSOLE OUTPUT ────────────────────────────────────────────────────────────

def print_summary(
    total_scraped: int,
    after_dedup: int,
    new_count: int,
    new_jobs: pd.DataFrame,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  JobSpy Run   {now}")
    print(sep)
    print(f"  Scraped       : {total_scraped}")
    print(f"  After dedup   : {after_dedup}  ({total_scraped - after_dedup} duplicates removed)")
    print(f"  Already seen  : {after_dedup - new_count}")
    print(f"  NEW jobs      : {new_count}")
    print(sep)

    if new_jobs.empty:
        print("  No new jobs this run.\n")
        return

    top = new_jobs.sort_values("relevance_score", ascending=False).head(10)
    print(f"\n  Top {len(top)} by Relevance Score:\n")
    print(f"  {'Score':>5}  {'Title':<42}  {'Company':<28}  {'Location':<22}  URL")
    print("  " + "-" * 140)
    for _, row in top.iterrows():
        score   = int(row.get("relevance_score", 0))
        title   = str(row.get("title",    "") or "")[:42]
        company = str(row.get("company",  "") or "")[:28]
        loc     = str(row.get("location", "") or "")[:22]
        url     = str(row.get("job_url",  "") or "")
        print(f"  {score:>5}  {title:<42}  {company:<28}  {loc:<22}  {url}")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[INFO] Fetching jobs from LinkedIn + Indeed (last 2 hours)...")

    raw = scrape_jobs(
        site_name=["indeed", "linkedin"],
        search_term=SEARCH_QUERY,
        location="United Kingdom",
        results_wanted=50,
        hours_old=2,                    # 2h window; seen-filter eliminates overlap
        is_remote=True,
        country_indeed="UK",
        linkedin_fetch_description=True,
        enforce_annual_salary=True,
        verbose=0,
    )

    total_scraped = len(raw)

    if raw.empty:
        print_summary(0, 0, 0, pd.DataFrame())
        return

    deduped     = deduplicate(raw)
    seen        = load_seen_jobs()
    new_jobs    = filter_new_jobs(deduped, seen)
    new_count   = len(new_jobs)

    if not new_jobs.empty:
        new_jobs = new_jobs.copy()
        new_jobs["relevance_score"] = new_jobs.apply(
            lambda row: score_job(row, PROFILE), axis=1
        )
        append_to_csv(new_jobs)
        print(f"[INFO] Appended {new_count} new jobs to {CSV_PATH}")

        # Toast notification for high-relevance matches
        top_matches = new_jobs[new_jobs["relevance_score"] >= NOTIFY_SCORE_THRESHOLD]
        if not top_matches.empty:
            count = len(top_matches)
            best  = top_matches.sort_values("relevance_score", ascending=False).iloc[0]
            body  = f"{best['title']} @ {best['company']}"
            if count > 1:
                body += f" (+{count - 1} more)"
            send_toast(f"JobSpy — {count} relevant job{'s' if count > 1 else ''} found", body)

    print_summary(total_scraped, len(deduped), new_count, new_jobs)

    # Update seen from the full deduped set (not just new_jobs)
    seen.update(deduped["job_url"].dropna().tolist())
    save_seen_jobs(seen)


if __name__ == "__main__":
    main()
