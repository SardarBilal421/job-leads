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
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response["response"]
    except Exception as e:
        err = str(e).lower()
        is_fallback_err = any(kw in err for kw in [
            "not found", "pull", "does not exist",   # model missing
            "memory", "out of memory", "insufficient", # RAM/VRAM
            "500",                                     # generic server error
        ])
        next_model = _next_in_chain(model)
        if is_fallback_err and next_model:
            print(f"[WARN] {model} failed ({str(e).splitlines()[0]})")
            print(f"[WARN] Falling back to {next_model}…")
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

def tailor_with_ollama(resume: dict, jd: str, model: str) -> dict:
    exp_count    = len([e for e in resume.get("experience", []) if e.get("company")])
    edu_count    = len([e for e in resume.get("education",  []) if e.get("institution")])
    skills_total = sum(len(v) for v in resume.get("skills", {}).values() if isinstance(v, list))

    prompt = f"""You are a senior technical recruiter and ATS resume expert with 15 years experience.

YOUR ONLY JOB: Rewrite the resume below to be perfectly tailored for this specific job posting.

══ HARD RULES — violating any of these makes the output useless ══
1. PRESERVE ALL {exp_count} experience entries — do NOT remove or merge any job, even if it seems unrelated
2. PRESERVE ALL {edu_count} education entries — do NOT remove any degree
3. PRESERVE ALL skills — you may REORDER them (most relevant first) but NEVER delete any skill
4. NEVER invent companies, job titles, dates, or achievements that are not in the original
5. Fix any text artifacts like {{25%}} → 25% silently
6. Output MUST be valid JSON with the EXACT same structure as the input — no extra fields, no missing fields
7. Return ONLY the JSON object — absolutely no markdown, no explanation, no text before or after

══ WHAT TO CHANGE ══
SUMMARY: Rewrite completely to target this specific role. Use keywords from the JD. 2-3 sentences max. Start with a strong hook, not "I am".

BULLETS: For each job, rewrite bullets to:
  - Mirror exact keywords and phrases from the JD
  - Lead with a strong action verb (Built, Led, Architected, Reduced, Delivered, Scaled, Optimised)
  - Include measurable impact where already present in the original (keep all numbers)
  - Be concise — one impactful sentence each

SKILLS: Move JD-required technologies to the front of each category.

══ JOB DESCRIPTION ══
{jd}

══ RESUME TO TAILOR (JSON) ══
{json.dumps(resume, indent=2, ensure_ascii=False)}

Output the tailored resume JSON now:"""

    print(f"  Tailoring with {model} …")

    for attempt in range(3):
        try:
            raw      = call_ollama(prompt, model=model)
            tailored = extract_json(raw)
            # Validate and repair — never let the model silently drop content
            tailored = _validate_and_repair(tailored, resume)
            return tailored
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt < 2:
                print(f"  [WARN] JSON parse error on attempt {attempt + 1}, retrying…")
            else:
                raise RuntimeError(f"Model returned unparseable JSON after 3 attempts: {exc}") from exc
    return resume


def _validate_and_repair(tailored: dict, original: dict) -> dict:
    """
    Ensure the tailored resume has not silently dropped experiences, education,
    or skills. Merges back anything missing from the original.
    """
    # ── Experiences ──────────────────────────────────────────────────────────
    orig_exps = {e["company"]: e for e in original.get("experience", []) if e.get("company")}
    tail_exps = {e["company"]: e for e in tailored.get("experience", []) if e.get("company")}

    merged_exps = list(tailored.get("experience", []))
    for company, orig_entry in orig_exps.items():
        if company not in tail_exps:
            print(f"  [REPAIR] Restored dropped experience: {company}")
            merged_exps.append(orig_entry)

    # Preserve original ordering
    order = [e["company"] for e in original.get("experience", []) if e.get("company")]
    merged_exps.sort(key=lambda e: order.index(e["company"]) if e.get("company") in order else 999)
    tailored["experience"] = merged_exps

    # ── Education ────────────────────────────────────────────────────────────
    orig_edus = {e["institution"]: e for e in original.get("education", []) if e.get("institution")}
    tail_edus = {e["institution"]: e for e in tailored.get("education",  []) if e.get("institution")}

    merged_edus = list(tailored.get("education", []))
    for inst, orig_entry in orig_edus.items():
        if inst not in tail_edus:
            print(f"  [REPAIR] Restored dropped education: {inst}")
            merged_edus.append(orig_entry)
    tailored["education"] = merged_edus

    # ── Skills ───────────────────────────────────────────────────────────────
    for category, orig_skills in original.get("skills", {}).items():
        tail_skills = tailored.get("skills", {}).get(category, [])
        missing = [s for s in orig_skills if s not in tail_skills]
        if missing:
            print(f"  [REPAIR] Restored {len(missing)} dropped skill(s) in '{category}'")
            tailored.setdefault("skills", {})[category] = tail_skills + missing

    # ── Contact fields ───────────────────────────────────────────────────────
    for field in ("name", "email", "phone", "location", "linkedin", "github", "portfolio"):
        if not tailored.get(field) and original.get(field):
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
