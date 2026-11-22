#!/usr/bin/env python3
"""Validate FreeTPU Agent Skill packaging, links, clean-room hygiene, and Python templates.

Checks performed:
  1. `SKILL.md` YAML frontmatter (`name`, `description`) and line-budget hygiene.
  2. Relative Markdown links in `skills/freetpu/**/*.md` resolve to existing files.
  3. Zero proprietary binaries, private paths, internal hostnames, or secret tokens.
  4. Python AST syntax validity across `skills/freetpu/templates/*.py` (plus optional `--run-self-tests`).
  5. `README.md` documents the `skills/freetpu` Agent Skill quick-install section.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BAD_EXTS = (".so", ".ko", ".img", ".apk", ".gguf", ".tflite", ".dgc0", ".dmg0", ".bin")
MAX_SKILL_MD_LINES = 300


def _forbidden_text_patterns() -> Tuple[Tuple[str, str], ...]:
    # Split literals so this validator does not trigger on its own source.
    return (
        (r"/" + r"Users/", "macOS user home path"),
        (r"bobby" + r"bones", "private username"),
        (r"saari" + r"-co", "private org reference"),
        (r"\bspark" + r"-[12]\b", "internal host reference"),
        (r"ro\.boot\.hw\." + r"soc\.id", "device-specific property"),
        (r"file" + r"://", "local file URI"),
        (r"-----BEGIN [A-Z ]*PRIVATE " + r"KEY-----", "private key material"),
        (r"\bghp_[A-Za-z0-9]{20,}\b", "GitHub personal access token"),
        (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "GitHub fine-grained token"),
        (r"\bAIza[0-9A-Za-z\-_]{30,}\b", "Google API key"),
    )


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Parse YAML frontmatter using Python stdlib only."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter delimiter '---'")
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        raise ValueError("SKILL.md is missing closing YAML frontmatter delimiter '---'")

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :])
    data: Dict[str, str] = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", line)
        if not m:
            raise ValueError(f"Unsupported YAML frontmatter line: {line!r}")
        key, rest = m.group(1), m.group(2).strip()
        if rest in (">-", ">", "|-", "|"):
            block: List[str] = []
            i += 1
            while i < len(fm_lines) and (fm_lines[i].startswith((" ", "\t")) or not fm_lines[i].strip()):
                block.append(fm_lines[i].strip())
                i += 1
            sep = " " if rest.startswith(">") else "\n"
            data[key] = sep.join(part for part in block if part).strip()
            continue
        if (rest.startswith('"') and rest.endswith('"')) or (rest.startswith("'") and rest.endswith("'")):
            rest = rest[1:-1]
        data[key] = rest
        i += 1
    return data, body


def strip_code_fences(md_text: str) -> str:
    """Remove fenced code blocks before scanning markdown links."""
    out: List[str] = []
    in_fence = False
    for line in md_text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    if in_fence:
        raise ValueError("Unclosed markdown code fence ('```')")
    return "\n".join(out)


def validate_skill(
    repo_root: Path,
    skill_dir: Path,
    readme_path: Path,
    run_self_tests: bool = False,
) -> List[str]:
    errors: List[str] = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return [f"Missing required skill file: {skill_md}"]

    raw_skill = skill_md.read_text(encoding="utf-8")
    skill_lines = raw_skill.splitlines()
    if len(skill_lines) > MAX_SKILL_MD_LINES:
        errors.append(
            f"{skill_md}: {len(skill_lines)} lines exceeds {MAX_SKILL_MD_LINES}-line progressive-disclosure budget"
        )

    try:
        fm, _ = parse_frontmatter(raw_skill)
    except ValueError as exc:
        errors.append(f"{skill_md}: {exc}")
        fm = {}

    name = fm.get("name", "")
    desc = fm.get("description", "")
    if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", name):
        errors.append(f"{skill_md}: invalid frontmatter name {name!r} (expected lowercase hyphenated)")
    elif name != skill_dir.name:
        errors.append(f"{skill_md}: frontmatter name {name!r} != directory name {skill_dir.name!r}")

    if len(desc) < 40 or len(desc) > 1024:
        errors.append(f"{skill_md}: description length {len(desc)} out of bounds [40, 1024]")

    # Markdown link, fence, binary, and private-leak checks across skill_dir
    link_re = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    forbidden = _forbidden_text_patterns()

    files = sorted(p for p in skill_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if not files:
        errors.append(f"No files found in {skill_dir}")

    for path in files:
        rel = path.relative_to(repo_root)
        if path.name.endswith(BAD_EXTS) or ".so." in path.name:
            errors.append(f"Forbidden binary file in skill directory: {rel}")
            continue

        raw_bytes = path.read_bytes()
        if raw_bytes[:4] in (b"\x7fELF", b"TFL3", b"GGUF") or raw_bytes[4:8] in (
            b"DGC0",
            b"DWS0",
            b"DGD0",
            b"DMG0",
        ):
            errors.append(f"Forbidden binary magic in {rel}")
            continue

        text = raw_bytes.decode("utf-8", errors="strict")
        for pat, label in forbidden:
            if re.search(pat, text):
                errors.append(f"Clean-room leak ({label}) in {rel}")

        if path.suffix.lower() == ".md":
            try:
                stripped = strip_code_fences(text)
            except ValueError as exc:
                errors.append(f"{rel}: {exc}")
                stripped = text
            for match in link_re.finditer(stripped):
                target = match.group(1).strip().split()[0]
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target_path = target.split("#", 1)[0]
                if not target_path:
                    continue
                resolved = (path.parent / target_path).resolve()
                try:
                    resolved.relative_to(repo_root.resolve())
                except ValueError:
                    errors.append(f"Relative link escapes repo root in {rel}: {target}")
                    continue
                if not resolved.exists():
                    errors.append(f"Broken relative link in {rel}: {target}")

    # Validate templates Python AST + optional self-tests
    templates_dir = skill_dir / "templates"
    templates = sorted(templates_dir.glob("*.py")) if templates_dir.is_dir() else []
    if not templates:
        errors.append(f"No Python templates found in {templates_dir}")

    for tpl in templates:
        rel = tpl.relative_to(repo_root)
        src = tpl.read_text(encoding="utf-8")
        try:
            ast.parse(src, filename=str(rel))
        except SyntaxError as exc:
            errors.append(f"Syntax error in template {rel}: {exc}")
            continue

        if run_self_tests:
            proc = subprocess.run(
                [sys.executable, str(tpl), "--self-test"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                errors.append(
                    f"Template self-test failed ({rel}, rc={proc.returncode}): "
                    f"{(proc.stderr or proc.stdout).strip()[-400:]}"
                )
                continue
            try:
                payload = json.loads(proc.stdout)
                if payload.get("status") != "ok":
                    errors.append(f"Template self-test unexpected status in {rel}: {payload!r}")
            except json.JSONDecodeError as exc:
                errors.append(f"Template self-test did not emit valid JSON ({rel}): {exc}")

    # Validate README quick-install section
    if not readme_path.is_file():
        errors.append(f"Missing README file: {readme_path}")
    else:
        readme_text = readme_path.read_text(encoding="utf-8")
        if "skills/freetpu" not in readme_text or "Agent Skill" not in readme_text:
            errors.append(f"{readme_path}: missing Agent Skill quick-install section referencing skills/freetpu")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate FreeTPU Agent Skill packaging and templates")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--skill-dir", type=Path, default=None)
    ap.add_argument("--readme", type=Path, default=None)
    ap.add_argument("--run-self-tests", action="store_true", help="Execute `--self-test` on every template")
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    skill_dir = (args.skill_dir or (repo_root / "skills" / "freetpu")).resolve()
    readme_path = (args.readme or (repo_root / "README.md")).resolve()

    errors = validate_skill(
        repo_root=repo_root,
        skill_dir=skill_dir,
        readme_path=readme_path,
        run_self_tests=args.run_self_tests,
    )
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    tpl_count = len(list((skill_dir / "templates").glob("*.py")))
    print(
        f"Skill validation PASSED: {skill_dir.relative_to(repo_root)} "
        f"({tpl_count} templates, self_tests={'ran' if args.run_self_tests else 'ast-only'})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
