#!/usr/bin/env python3
"""
Resume Tailor — AI-powered resume customization using local Ollama models.

Commands:
  python tailor.py setup                           Create/update default profile
  python tailor.py setup --profile alice           Create named profile
  python tailor.py tailor "JD text"                Tailor resume (inline JD)
  python tailor.py tailor --file jd.txt            Tailor from file
  python tailor.py tailor "JD" --company Google    Tag output with company name
  python tailor.py tailor "JD" --template classic  Choose template
  python tailor.py tailor "JD" --profile alice     Use specific profile
  python tailor.py list                            List all profiles + history
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import ollama
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa

# ── PATHS & DEFAULTS ──────────────────────────────────────────────────────────

BUILDER_DIR   = Path(__file__).parent.resolve()
PROFILES_DIR  = BUILDER_DIR / "profiles"
TEMPLATES_DIR = BUILDER_DIR / "templates"
OUTPUT_DIR    = BUILDER_DIR / "output"

DEFAULT_PROFILE  = "default"
PRIMARY_MODEL    = "qwen2.5-coder:32b"
FALLBACK_MODEL   = "qwen2.5-coder:7b"

# Ordered fallback chain — tried left to right when a model fails due to resources
MODEL_CHAIN = [
    "qwen2.5-coder:32b",
    "qwen2.5-coder:7b",
    "dolphin-llama3:latest",
    "llama3.2:latest",
]

# Models that have already failed this session — skip them immediately
_failed_models: set[str] = set()
DEFAULT_TEMPLATE = "modern"

TEMPLATES = ["modern", "classic", "minimal", "ats_safe", "tech_bold"]

# ── RESUME JSON SCHEMA (used as blank template during setup) ──────────────────

BLANK_RESUME: dict = {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "github": "",
    "portfolio": "",
    "summary": "",
    "experience": [
        {
            "company": "",
            "title": "",
            "location": "",
            "start_date": "",
            "end_date": "",
            "bullets": []
        }
    ],
    "education": [
        {
            "institution": "",
            "degree": "",
            "start_date": "",
            "end_date": "",
            "gpa": "",
            "achievements": []
        }
    ],
    "skills": {
        "languages":  [],
        "frameworks": [],
        "tools":      [],
        "other":      []
    },
    "projects": [
        {
            "name":        "",
            "description": "",
            "tech":        [],
            "url":         ""
        }
    ],
    "certifications": []
}

# ── PROFILE HELPERS ───────────────────────────────────────────────────────────

def profile_dir(profile: str) -> Path:
    return PROFILES_DIR / profile

def resume_path(profile: str) -> Path:
    return profile_dir(profile) / "resume.json"

def history_dir(profile: str) -> Path:
    return profile_dir(profile) / "history"

def load_resume(profile: str) -> dict:
    path = resume_path(profile)
    if not path.exists():
        print(f"\n[ERROR] No resume found for profile '{profile}'.")
        print(f"        Run: python tailor.py setup --profile {profile}\n")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))

def save_resume(profile: str, data: dict) -> None:
    p = resume_path(profile)
    p.parent.mkdir(parents=True, exist_ok=True)
    history_dir(profile).mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# ── OLLAMA ────────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = PRIMARY_MODEL) -> str:
    global _failed_models

    # Skip models already known to be unavailable this session
    if model in _failed_models:
        next_model = _next_in_chain(model)
        if next_model:
            return call_ollama(prompt, model=next_model)
        raise RuntimeError("No available Ollama model. Make sure 'ollama serve' is running.")

    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response["response"]
    except Exception as e:
        err = str(e).lower()
        is_fallback_err = any(kw in err for kw in [
            "not found", "pull", "does not exist",
            "memory", "out of memory", "insufficient",
            "500",
        ])
        next_model = _next_in_chain(model)
        if is_fallback_err and next_model:
            _failed_models.add(model)  # remember — don't retry this model again
            print(f"[WARN] {model} unavailable — using {next_model} for this session")
            return call_ollama(prompt, model=next_model)
        raise RuntimeError(
            f"Ollama error: {e}\n"
            "Make sure Ollama is running: open a terminal and run 'ollama serve'"
        ) from e

def _next_in_chain(model: str) -> str | None:
    """Return the next smaller model in MODEL_CHAIN, or None if already at the end."""
    try:
        idx = MODEL_CHAIN.index(model)
        return MODEL_CHAIN[idx + 1] if idx + 1 < len(MODEL_CHAIN) else None
    except ValueError:
        return FALLBACK_MODEL

def extract_json(text: str) -> dict:
    """Extract JSON object from LLM output (handles markdown code fences)."""
    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    # Find outermost { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start:end])

# ── SETUP COMMAND ─────────────────────────────────────────────────────────────

def cmd_setup(args: argparse.Namespace) -> None:
    profile = args.profile
    path    = resume_path(profile)

    print(f"\n── Resume Setup  [ profile: {profile} ] " + "─" * 30)

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        print(f"  Existing resume found for: {existing.get('name', '(unnamed)')}")
        print("  [u] Update existing   [r] Replace entirely")
        choice = input("  Choice > ").strip().lower()
        if choice != "r":
            print("  Keeping existing data as base.")
    else:
        print("  No existing resume found — starting fresh.\n")

    # Accept resume input
    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            print(f"[ERROR] File not found: {fpath}")
            sys.exit(1)
        raw_text = fpath.read_text(encoding="utf-8")
        print(f"  Loaded resume from: {fpath}")
    else:
        print("\n  Paste your resume as plain text.")
        print("  When done, press Enter on a blank line then type END:\n")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        raw_text = "\n".join(lines).strip()

    if not raw_text:
        print("[ERROR] No resume text provided.")
        sys.exit(1)

    model = args.model or PRIMARY_MODEL
    print(f"\n  Parsing with {model} …")

    parse_prompt = f"""You are a resume parser. Extract all information from the resume text below and return it as a single JSON object.

Use this EXACT structure (no extra fields, no missing fields):

{{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+1234567890",
  "location": "City, Country",
  "linkedin": "linkedin.com/in/username",
  "github": "github.com/username",
  "portfolio": "website.com",
  "summary": "Professional summary paragraph",
  "experience": [
    {{
      "company": "Company Name",
      "title": "Job Title",
      "location": "City, Country",
      "start_date": "Mon Year",
      "end_date": "Mon Year or Present",
      "bullets": ["Achievement or responsibility 1", "Achievement 2"]
    }}
  ],
  "education": [
    {{
      "institution": "University Name",
      "degree": "BSc Computer Science",
      "start_date": "2018",
      "end_date": "2022",
      "gpa": "",
      "achievements": []
    }}
  ],
  "skills": {{
    "languages":  ["Python", "JavaScript"],
    "frameworks": ["React", "Node.js"],
    "tools":      ["Docker", "Git"],
    "other":      ["REST APIs", "Agile"]
  }},
  "projects": [
    {{
      "name": "Project Name",
      "description": "What it does and its impact",
      "tech": ["React", "Python"],
      "url": "github.com/..."
    }}
  ],
  "certifications": []
}}

Leave unknown fields as empty strings or empty arrays.
Return ONLY the JSON object — no markdown, no explanation.

RESUME TEXT:
{raw_text}"""

    try:
        raw_response = call_ollama(parse_prompt, model=model)
        resume_data  = extract_json(raw_response)
        save_resume(profile, resume_data)

        skills_count = sum(
            len(v) for v in resume_data.get("skills", {}).values() if isinstance(v, list)
        )
        print(f"\n  [✓] Resume saved  →  {resume_path(profile)}")
        print(f"      Name       : {resume_data.get('name', 'N/A')}")
        print(f"      Experience : {len(resume_data.get('experience', []))} entries")
        print(f"      Projects   : {len(resume_data.get('projects', []))} entries")
        print(f"      Skills     : {skills_count} total\n")

    except Exception as exc:
        print(f"\n[ERROR] Parsing failed: {exc}")
        print("  Saving raw text — edit the JSON file manually:")
        fallback = json.loads(json.dumps(BLANK_RESUME))
        fallback["_raw_text"] = raw_text
        save_resume(profile, fallback)
        print(f"  {resume_path(profile)}\n")

# ── TAILOR COMMAND ────────────────────────────────────────────────────────────

def _generate_summary(skeleton: dict, jd: str, model: str) -> str:
    """Generate a rich professional summary in a dedicated focused call."""
    name        = skeleton.get("name", "The candidate")
    titles      = [e["title"] for e in skeleton.get("experience", []) if e.get("title")]
    companies   = [e["company"] for e in skeleton.get("experience", []) if e.get("company")]
    edu         = [e["degree"] for e in skeleton.get("education", []) if e.get("degree")]
    orig_skills = skeleton.get("existing_skills", {})
    all_skills  = [s for lst in orig_skills.values() for s in lst]

    prompt = f"""Write a sharp 3-sentence professional resume summary for this candidate.

CANDIDATE FACTS:
- Roles: {', '.join(titles[:4])}
- Education: {', '.join(edu)}
- Skills: {', '.join(all_skills[:12])}

JOB DESCRIPTION:
{jd}

STRICT RULES:
• Write exactly 3 sentences as a single flowing paragraph — not a list, not bullet points
• Total word count must be between 55 and 75 words — aim for 65 words
• ALWAYS write "7+ years of experience" or "over 7 years" — never write any lower number
• Sentence 1 (~25 words): "[Job title from JD] with 7+ years of experience in [3 technologies from JD — use programming languages or key frameworks, NOT 'Angular' as a language], delivering [type of product] that [scale/impact]."
• Sentence 2 (~25 words): Highlight a specific technical depth area and 2-3 more JD technologies applied in real projects.
• Sentence 3 (~15 words): MSc in AI + leadership/mentoring + unique value for this specific role.
• Do NOT start with "I", "Experienced", or "Passionate"
• Confident third-person tone throughout

Output ONLY the paragraph — no labels, no JSON, no markdown, no extra text:"""

    print("  Generating summary …")
    try:
        raw = call_ollama(prompt, model=model)
        summary = raw.strip().strip('"').strip("'")
        summary = re.sub(r"^summary[:\s]*", "", summary, flags=re.IGNORECASE).strip()
        # Hard-correct any experience figure the model wrote below 7
        summary = _enforce_years(summary)
        return summary
    except Exception:
        return ""  # fall through to main prompt if this fails


def _enforce_years(text: str) -> str:
    """Replace any 'X years' figure below 7 with '7+ years'."""
    def _fix(m):
        try:
            n = int(m.group(1))
            if n < 7:
                return "7+ years"
        except ValueError:
            pass
        return m.group(0)
    # Match patterns like "4 years", "5+ years", "over 4 years", "4-5 years"
    text = re.sub(r"\b(\d+)\+?\s*(?:to\s*\d+\s*)?years?\b", _fix, text, flags=re.IGNORECASE)
    text = re.sub(r"\bover\s+(\d+)\s+years?\b", lambda m: "over 7 years" if int(m.group(1)) < 7 else m.group(0), text, flags=re.IGNORECASE)
    return text


def tailor_with_ollama(resume: dict, jd: str, model: str) -> dict:
    # Build a skeleton — only the verified facts we must keep
    skeleton = _build_skeleton(resume)

    # Generate summary separately for richer, more focused output
    summary = _generate_summary(skeleton, jd, model)

    prompt = f"""You are a senior technical recruiter and professional resume writer. Your job is to write resume content that passes both ATS screening AND human recruiter review.

CANDIDATE WORK HISTORY (do not change companies, titles, or dates):
{json.dumps(skeleton, indent=2, ensure_ascii=False)}

JOB DESCRIPTION:
{jd}

════════ CRITICAL RULES — a real recruiter will reject this resume if you break any of these ════════

■ SUMMARY (copy exactly, do not change a single word):
"{summary if summary else 'Experienced Full Stack Engineer with 7+ years delivering scalable web and AI systems.'}"

■ ACTION VERB RULES — this is the most important rule:
  • Across the ENTIRE resume, use each verb MAXIMUM ONCE
  • "Led" may appear AT MOST ONCE in the whole document
  • "Built" may appear AT MOST ONCE
  • "Developed" may appear AT MOST ONCE
  • Choose from this diverse verb bank and spread them across all roles:
    Architected, Engineered, Delivered, Designed, Implemented, Spearheaded, Scaled, Optimised,
    Refactored, Automated, Integrated, Deployed, Migrated, Owned, Shipped, Established,
    Collaborated, Reduced, Increased, Streamlined, Introduced, Launched, Consolidated

■ EXPERIENCE BULLETS — 5 per role, 22–32 words each:
  • Formula: [Unique verb] + [what was built/done] + [specific tech from JD] + [business or user outcome]
  • GOOD bullet: "Architected a microservices API gateway using Node.js and AWS Lambda, processing 3M daily transactions with 99.9% uptime and cutting infrastructure costs by 30%."
  • BAD bullet: "Led the development of APIs improving performance." (too vague, no context, no tech)
  • METRICS must be believable — use these formats ONLY:
    ✓ "improved conversion by 18%"   ✓ "serving 50k daily users"   ✓ "saving £12k/month"
    ✓ "reduced deployment time from 2 hours to 8 minutes"   ✓ "across a 5-engineer team"
    ✗ NEVER write "reduced latency by 35ms" or "improved performance by 70ms" — ms savings are not credible
    ✗ NEVER write two bullets in the same role that contradict each other (e.g. one says frequency increased, another says it decreased)
  • At least 1 of the 5 bullets per role must mention a BUSINESS OUTCOME: revenue, cost saving, user retention, conversion rate, or customer satisfaction
  • At least 1 bullet per role must show LEADERSHIP or PEOPLE MANAGEMENT: team size managed, engineers mentored, direct reports, or stakeholder communication
  • If the JD mentions specific Gen AI frameworks (LangGraph, Crew AI, OpenAI, LangChain), include at least one bullet referencing them in the most recent role

■ SKILLS — maximum 22 skills total, no keyword stuffing:
  • Keep only the candidate's strongest and most JD-relevant skills
  • Add JD skills including any named Gen AI frameworks (LangGraph, Crew AI, OpenAI SDK, LangChain etc.)
  • IMPORTANT: "languages" must contain ONLY programming languages (TypeScript, Python, JavaScript, Go etc.) — Angular, React, Vue are FRAMEWORKS not languages, never put them in languages
  • Cap: languages ≤5, frameworks ≤7, tools ≤7, other ≤5

■ EDUCATION: copy exactly as provided, no changes

════════════════════════════════════════════════
Return ONLY valid JSON, no markdown, no explanation:
════════════════════════════════════════════════
{{
  "name":"...","email":"...","phone":"...","location":"...","linkedin":"...","github":"...","portfolio":"...",
  "summary":"...",
  "experience":[{{"company":"...","title":"...","location":"...","start_date":"...","end_date":"...","bullets":["...","...","...","...","..."]}}],
  "education":[{{"institution":"...","degree":"...","start_date":"...","end_date":"...","gpa":"","achievements":[]}}],
  "skills":{{"languages":[],"frameworks":[],"tools":[],"other":[]}},
  "projects":[],"certifications":[]
}}"""

    print(f"  Generating with {model} …")

    for attempt in range(3):
        try:
            raw      = call_ollama(prompt, model=model)
            tailored = extract_json(raw)
            tailored = _enforce_skeleton(tailored, resume)
            if summary:
                tailored["summary"] = summary
            tailored = _expand_short_bullets(tailored, jd, model)
            return tailored
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt < 2:
                print(f"  [WARN] JSON parse error on attempt {attempt + 1}, retrying…")
            else:
                raise RuntimeError(f"Model returned unparseable JSON after 3 attempts: {exc}") from exc
    return resume


def _expand_short_bullets(tailored: dict, jd: str, model: str, min_words: int = 22) -> dict:
    """Expand any bullet under min_words with a focused Ollama call."""
    jd_snippet = jd[:400]
    short_count = sum(
        1 for exp in tailored.get("experience", [])
        for b in exp.get("bullets", [])
        if len(b.split()) < min_words
    )
    if short_count == 0:
        return tailored

    print(f"  Expanding {short_count} short bullet(s) …")

    for exp in tailored.get("experience", []):
        expanded = []
        for bullet in exp.get("bullets", []):
            if len(bullet.split()) < min_words:
                better = _expand_one_bullet(
                    bullet, exp.get("title", ""), exp.get("company", ""), jd_snippet, model
                )
                expanded.append(better)
            else:
                expanded.append(bullet)
        exp["bullets"] = expanded

    tailored = _deduplicate_verbs(tailored)
    return tailored


def _deduplicate_verbs(tailored: dict) -> dict:
    """Replace repeated opening verbs — only when the replacement is grammatically safe."""
    # Only verbs that take a direct object (safe to swap in any bullet context)
    replacements = [
        "Delivered", "Engineered", "Scaled", "Shipped", "Streamlined",
        "Championed", "Overhauled", "Drove", "Consolidated", "Coordinated",
    ]
    # Words that follow a verb and signal it cannot be safely swapped
    # e.g. "Reduced latency" is safe; "Collaborated with" is not
    unsafe_followers = {"with", "by", "to", "for", "on", "in", "as", "across", "page", "load"}

    seen_verbs: dict[str, int] = {}
    replacement_idx = 0

    for exp in tailored.get("experience", []):
        new_bullets = []
        for bullet in exp.get("bullets", []):
            words = bullet.split()
            if not words:
                new_bullets.append(bullet)
                continue
            verb = words[0].rstrip(",")
            next_word = words[1].lower().rstrip(",") if len(words) > 1 else ""
            seen_verbs[verb] = seen_verbs.get(verb, 0) + 1

            if (seen_verbs[verb] > 1
                    and replacement_idx < len(replacements)
                    and next_word not in unsafe_followers):
                new_verb = replacements[replacement_idx]
                replacement_idx += 1
                bullet = new_verb + bullet[len(verb):]
                seen_verbs[new_verb] = seen_verbs.get(new_verb, 0) + 1

            new_bullets.append(bullet)
        exp["bullets"] = new_bullets

    return tailored


def _expand_one_bullet(bullet: str, title: str, company: str, jd_snippet: str, model: str) -> str:
    # Detect which verb the bullet currently starts with so we keep it
    first_word = bullet.split()[0] if bullet.split() else ""
    prompt = f"""Expand this resume bullet point to 25–30 words. Keep the opening verb exactly as-is.

Role: {title} at {company}
JD context: {jd_snippet}
Current bullet: {bullet}

Rewrite rules:
- Keep the first word "{first_word}" exactly
- Add specific technologies from the JD context (pick 1–2 that fit naturally)
- Add or strengthen the outcome — use %, user counts, cost savings, or team size
- NEVER use millisecond (ms) metrics — they are not credible
- Business outcomes preferred: "improving conversion by X%", "serving Xk users", "saving £X/month"
- Write ONE complete sentence, 25–30 words

Output ONLY the expanded bullet — nothing else:"""

    try:
        raw = call_ollama(prompt, model=model).strip().strip('"').strip("'")
        # Only accept if it's actually longer
        if len(raw.split()) >= min(20, len(bullet.split()) + 5):
            return raw
    except Exception:
        pass
    return bullet  # fall back to original if expansion fails


def _build_skeleton(resume: dict) -> dict:
    """Extract only the verified facts from the resume (companies, titles, dates, education)."""
    return {
        "name":      resume.get("name", ""),
        "email":     resume.get("email", ""),
        "phone":     resume.get("phone", ""),
        "location":  resume.get("location", ""),
        "linkedin":  resume.get("linkedin", ""),
        "github":    resume.get("github", ""),
        "portfolio": resume.get("portfolio", ""),
        "experience": [
            {
                "company":    e.get("company", ""),
                "title":      e.get("title", ""),
                "location":   e.get("location", ""),
                "start_date": e.get("start_date", ""),
                "end_date":   e.get("end_date", ""),
            }
            for e in resume.get("experience", []) if e.get("company")
        ],
        "education": [
            {
                "institution": e.get("institution", ""),
                "degree":      e.get("degree", ""),
                "start_date":  e.get("start_date", ""),
                "end_date":    e.get("end_date", ""),
            }
            for e in resume.get("education", []) if e.get("institution")
        ],
        "existing_skills": resume.get("skills", {}),
    }


def _enforce_skeleton(tailored: dict, original: dict) -> dict:
    """
    Hard-enforce the facts skeleton — make sure company names, titles, dates,
    and education are exactly what the user provided, not hallucinated versions.
    """
    orig_exps = {e["company"]: e for e in original.get("experience", []) if e.get("company")}
    tail_exps = {e.get("company", ""): e for e in tailored.get("experience", [])}

    final_exps = []
    for e in original.get("experience", []):
        if not e.get("company"):
            continue
        co = e["company"]
        # Take AI-generated bullets but enforce real facts for structural fields
        generated = tail_exps.get(co, {})
        final_exps.append({
            "company":    co,
            "title":      e.get("title") or generated.get("title", ""),
            "location":   e.get("location") or generated.get("location", ""),
            "start_date": e.get("start_date") or generated.get("start_date", ""),
            "end_date":   e.get("end_date") or generated.get("end_date", ""),
            "bullets":    generated.get("bullets") or e.get("bullets", []),
        })

    # Add any AI-generated entries for companies in tailored but not in original
    # (shouldn't happen but handle gracefully)
    known = {e["company"] for e in final_exps}
    for e in tailored.get("experience", []):
        if e.get("company") and e["company"] not in known:
            print(f"  [WARN] Removed hallucinated company: {e['company']}")

    tailored["experience"] = final_exps

    # Enforce education facts
    orig_edus = {e["institution"]: e for e in original.get("education", []) if e.get("institution")}
    tail_edus = {e.get("institution", ""): e for e in tailored.get("education", [])}
    final_edus = []
    for e in original.get("education", []):
        if not e.get("institution"):
            continue
        generated = tail_edus.get(e["institution"], {})
        final_edus.append({
            "institution": e["institution"],
            "degree":      e.get("degree") or generated.get("degree", ""),
            "start_date":  e.get("start_date") or generated.get("start_date", ""),
            "end_date":    e.get("end_date") or generated.get("end_date", ""),
            "gpa":         e.get("gpa", ""),
            "achievements": e.get("achievements", []),
        })
    tailored["education"] = final_edus

    # Ensure all original skills are present (AI adds JD skills on top)
    for category, orig_skills in original.get("skills", {}).items():
        tail_cat = tailored.get("skills", {}).get(category, [])
        for s in orig_skills:
            if s not in tail_cat:
                tail_cat.append(s)
        tailored.setdefault("skills", {})[category] = tail_cat

    # Enforce contact fields
    for field in ("name", "email", "phone", "location", "linkedin", "github", "portfolio"):
        if original.get(field):
            tailored[field] = original[field]

    return tailored

def cmd_tailor(args: argparse.Namespace) -> None:
    profile  = args.profile
    template = args.template
    company  = args.company or "job"
    model    = args.model or PRIMARY_MODEL

    # ── Get JD text ──────────────────────────────────────────────────────────
    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            print(f"[ERROR] File not found: {fpath}")
            sys.exit(1)
        jd_text = fpath.read_text(encoding="utf-8")
        print(f"  Loaded JD from: {fpath}")
    elif args.jd:
        jd_text = args.jd
    else:
        print("\nPaste the job description. Press Enter on a blank line then type END:\n")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        jd_text = "\n".join(lines).strip()

    if not jd_text.strip():
        print("[ERROR] Empty job description.")
        sys.exit(1)

    # ── Load & tailor ────────────────────────────────────────────────────────
    print(f"\n── Resume Tailor  [ profile: {profile} | template: {template} ] " + "─" * 20)
    master  = load_resume(profile)
    print(f"  Loaded resume for: {master.get('name', profile)}")

    tailored = tailor_with_ollama(master, jd_text, model=model)

    # ── Output paths ─────────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_co     = re.sub(r"[^\w]", "_", company)
    stem        = f"{safe_co}_{template}_{ts}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_dir = history_dir(profile)
    hist_dir.mkdir(parents=True, exist_ok=True)

    out_pdf  = OUTPUT_DIR / f"{stem}.pdf"
    out_json = OUTPUT_DIR / f"{stem}.json"
    hist_pdf = hist_dir   / f"{stem}.pdf"

    # Save tailored JSON for reference / re-render
    out_json.write_text(json.dumps(tailored, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render PDF
    render_pdf(tailored, template, out_pdf)

    # Copy to profile history
    if out_pdf.exists():
        shutil.copy2(out_pdf, hist_pdf)

    print(f"\n  [✓] Done!")
    print(f"      Profile  : {profile}")
    print(f"      Template : {template}")
    print(f"      Company  : {company}")
    print(f"      PDF      : {out_pdf}")
    print(f"      History  : {hist_pdf}\n")

# ── PDF RENDERING ─────────────────────────────────────────────────────────────

def render_pdf(resume: dict, template_name: str, out_path: Path) -> None:
    env      = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    template = env.get_template(f"{template_name}.html")
    html     = template.render(resume=resume)

    # Always save HTML alongside (useful for browser print-to-PDF)
    html_path = out_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    try:
        with open(str(out_path), "wb") as pdf_file:
            result = pisa.CreatePDF(html.encode("utf-8"), dest=pdf_file)
        if result.err:
            raise RuntimeError(f"xhtml2pdf reported {result.err} error(s)")
        print(f"  [✓] PDF  → {out_path.name}")
    except Exception as exc:
        print(f"  [WARN] PDF generation failed: {exc}")
        print(f"  [✓] HTML → {html_path.name}  (open in browser → Print → Save as PDF)")

# ── LIST COMMAND ──────────────────────────────────────────────────────────────

def cmd_list(_args: argparse.Namespace) -> None:
    if not PROFILES_DIR.exists() or not any(PROFILES_DIR.iterdir()):
        print("\nNo profiles found.  Run: python tailor.py setup\n")
        return

    print("\n── Profiles " + "─" * 50)
    for p in sorted(PROFILES_DIR.iterdir()):
        if not p.is_dir():
            continue
        rp   = p / "resume.json"
        hist = p / "history"
        name = json.loads(rp.read_text(encoding="utf-8")).get("name", "(unnamed)") if rp.exists() else "(not set up)"
        pdfs = sorted(hist.glob("*.pdf"), reverse=True) if hist.exists() else []

        print(f"\n  ● {p.name}  —  {name}")
        print(f"    Resume   : {'✓  ' + str(rp) if rp.exists() else '✗  (run setup)'}")
        print(f"    History  : {len(pdfs)} tailored resume(s)")
        for pdf in pdfs[:5]:
            print(f"               └ {pdf.name}")
        if len(pdfs) > 5:
            print(f"               └ … and {len(pdfs) - 5} more")
    print()

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tailor.py",
        description="Resume Tailor — AI-powered resume customisation with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
        Templates : {', '.join(TEMPLATES)}
        Models    : {PRIMARY_MODEL} (default), {FALLBACK_MODEL} (fallback)

        Examples:
          python tailor.py setup
          python tailor.py setup --profile alice --file my_resume.txt
          python tailor.py tailor "We are hiring a React engineer…" --company Stripe
          python tailor.py tailor --file jd.txt --template classic --company Google
          python tailor.py tailor "JD…" --profile alice --template tech_bold
          python tailor.py list
        """)
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    # ── setup ──────────────────────────────────────────────────────────────
    p_setup = sub.add_parser("setup", help="Create or update a profile's master resume")
    p_setup.add_argument("--profile", default=DEFAULT_PROFILE, metavar="NAME")
    p_setup.add_argument("--file", metavar="PATH", help="Load resume from text file instead of pasting")
    p_setup.add_argument("--model", metavar="MODEL", help=f"Ollama model (default: {PRIMARY_MODEL})")

    # ── tailor ─────────────────────────────────────────────────────────────
    p_tailor = sub.add_parser("tailor", help="Tailor a resume for a job description")
    p_tailor.add_argument("jd", nargs="?", metavar="JD_TEXT", help="Job description text (or omit to paste)")
    p_tailor.add_argument("--file",     metavar="PATH",     help="Load JD from a text file")
    p_tailor.add_argument("--profile",  default=DEFAULT_PROFILE, metavar="NAME")
    p_tailor.add_argument("--template", default=DEFAULT_TEMPLATE, choices=TEMPLATES)
    p_tailor.add_argument("--company",  metavar="NAME",     help="Company name for filename")
    p_tailor.add_argument("--model",    metavar="MODEL",    help=f"Ollama model (default: {PRIMARY_MODEL})")

    # ── list ───────────────────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List all profiles and tailored resume history")
    p_list  # no extra args

    args = parser.parse_args()

    if   args.command == "setup":  cmd_setup(args)
    elif args.command == "tailor": cmd_tailor(args)
    elif args.command == "list":   cmd_list(args)
    else:                          parser.print_help()

if __name__ == "__main__":
    main()
