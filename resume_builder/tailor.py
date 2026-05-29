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

def _generate_summary(skeleton: dict, jd: str, model: str) -> str:
    """Generate a rich professional summary in a dedicated focused call."""
    name        = skeleton.get("name", "The candidate")
    titles      = [e["title"] for e in skeleton.get("experience", []) if e.get("title")]
    companies   = [e["company"] for e in skeleton.get("experience", []) if e.get("company")]
    edu         = [e["degree"] for e in skeleton.get("education", []) if e.get("degree")]
    orig_skills = skeleton.get("existing_skills", {})
    all_skills  = [s for lst in orig_skills.values() for s in lst]

    prompt = f"""Write a professional resume summary paragraph for this candidate applying to the job below.

CANDIDATE FACTS:
- Name: {name}
- Roles held: {', '.join(titles[:4])}
- Companies: {', '.join(companies[:4])}
- Education: {', '.join(edu)}
- Core skills: {', '.join(all_skills[:15])}

JOB DESCRIPTION:
{jd}

REQUIREMENTS — follow every rule:
1. Write EXACTLY 5 to 6 sentences as one flowing paragraph (not a list, not bullet points)
2. Total length must be between 100 and 140 words — count carefully
3. Sentence 1: Job title from JD + years of experience + 3 core technologies from JD
4. Sentence 2: Describe the scale and type of systems built (products, users, requests)
5. Sentence 3: Highlight 4-5 specific technologies from the JD and how they were applied
6. Sentence 4: Mention the MSc in AI and how it adds value to this role
7. Sentence 5: Describe leadership, mentoring, or cross-functional collaboration
8. Sentence 6: State clearly what unique value the candidate brings to this specific company/role
9. Do NOT start with "I", "Experienced", or "Passionate"
10. Use confident, third-person professional tone

Output ONLY the paragraph text — no labels, no JSON, no markdown:"""

    print("  Generating summary …")
    try:
        raw = call_ollama(prompt, model=model)
        # Strip any accidental JSON or markdown wrapping
        summary = raw.strip().strip('"').strip("'")
        summary = re.sub(r"^summary[:\s]*", "", summary, flags=re.IGNORECASE).strip()
        return summary
    except Exception:
        return ""  # fall through to main prompt if this fails


def tailor_with_ollama(resume: dict, jd: str, model: str) -> dict:
    # Build a skeleton — only the verified facts we must keep
    skeleton = _build_skeleton(resume)

    # Generate summary separately for richer, more focused output
    summary = _generate_summary(skeleton, jd, model)

    prompt = f"""You are a world-class professional resume writer with 20 years of experience placing candidates at top tech companies.

You have two inputs:
1. THE CANDIDATE'S WORK HISTORY — real companies, titles, and dates. These are facts you must not change.
2. THE JOB DESCRIPTION — this is your complete content brief. Everything you write must serve this JD.

════════════════════════════════════════════════
CANDIDATE WORK HISTORY (facts — do not alter):
════════════════════════════════════════════════
{json.dumps(skeleton, indent=2, ensure_ascii=False)}

════════════════════════════════════════════════
JOB DESCRIPTION:
════════════════════════════════════════════════
{jd}

════════════════════════════════════════════════
INSTRUCTIONS — follow every point exactly:
════════════════════════════════════════════════

■ SUMMARY:
  Use EXACTLY this pre-written summary — copy it word for word into the "summary" field, do not shorten or rewrite it:
  "{summary if summary else 'Write a 5-6 sentence professional summary targeting this role.'}"

■ EXPERIENCE BULLETS — write 5 detailed bullets per role:
  • Every bullet must be a full, rich sentence (not a fragment)
  • Every bullet must reference at least one specific technology or methodology from the JD
  • Use a DIFFERENT strong action verb to start each bullet — choose from:
    Architected, Built, Engineered, Delivered, Optimised, Scaled, Integrated, Led, Deployed,
    Reduced, Increased, Automated, Refactored, Designed, Spearheaded, Implemented, Migrated
  • At least 3 of the 5 bullets must contain a quantified metric (%, ms, users, requests/day, hours saved, $ value)
  • Bullets should feel like a senior engineer wrote them — specific, technical, results-driven
  • Make bullets clearly relevant to the responsibilities and requirements in the JD

■ SKILLS — be comprehensive and thorough:
  • Start with all existing candidate skills
  • Extract and ADD every single technology, tool, framework, language, methodology, and concept mentioned anywhere in the JD
  • Also add closely related industry-standard skills implied by the JD (e.g. if JD says "React" also add "React Hooks", "JSX"; if "Node.js" also add "npm", "Express"; if "AWS" also add relevant AWS services)
  • Organise by category — most JD-relevant skills appear first in each list
  • languages: programming languages only
  • frameworks: frontend/backend frameworks and libraries
  • tools: devops, databases, testing, cloud, build tools
  • other: methodologies, soft skills, concepts (Agile, TDD, REST, GraphQL, Microservices, etc.)

■ EDUCATION: Keep exactly as provided — do not change any field

════════════════════════════════════════════════
OUTPUT FORMAT — return ONLY this JSON, nothing else:
════════════════════════════════════════════════
{{
  "name": "...", "email": "...", "phone": "...", "location": "...",
  "linkedin": "...", "github": "...", "portfolio": "...",
  "summary": "...",
  "experience": [
    {{
      "company": "...", "title": "...", "location": "...",
      "start_date": "...", "end_date": "...",
      "bullets": ["bullet 1", "bullet 2", "bullet 3", "bullet 4", "bullet 5"]
    }}
  ],
  "education": [{{"institution":"...","degree":"...","start_date":"...","end_date":"...","gpa":"","achievements":[]}}],
  "skills": {{
    "languages": ["lang1", "lang2"],
    "frameworks": ["fw1", "fw2"],
    "tools": ["tool1", "tool2"],
    "other": ["concept1", "concept2"]
  }},
  "projects": [],
  "certifications": []
}}

Generate the complete resume JSON now:"""

    print(f"  Generating with {model} …")

    for attempt in range(3):
        try:
            raw      = call_ollama(prompt, model=model)
            tailored = extract_json(raw)
            tailored = _enforce_skeleton(tailored, resume)
            # Always use the separately generated summary — never let the main
            # call overwrite it with a shorter version
            if summary:
                tailored["summary"] = summary
            return tailored
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt < 2:
                print(f"  [WARN] JSON parse error on attempt {attempt + 1}, retrying…")
            else:
                raise RuntimeError(f"Model returned unparseable JSON after 3 attempts: {exc}") from exc
    return resume


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
