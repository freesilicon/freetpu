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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - stdlib fallback when PyYAML is not installed
    yaml = None

BAD_EXTS = (".so", ".ko", ".img", ".apk", ".gguf", ".tflite", ".dgc0", ".dmg0", ".bin")
MAX_SKILL_MD_LINES = 300
MAX_SKILL_NAME_LEN = 64


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


def _extract_frontmatter_block(text: str) -> Tuple[str, str]:
    """Split a Markdown file with YAML frontmatter into `(frontmatter_yaml, body)`."""
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
    return "\n".join(lines[1:end_idx]), "\n".join(lines[end_idx + 1 :])


def _strip_inline_comment(raw: str) -> str:
    """Strip an unquoted YAML `# ...` comment (preceded by whitespace or at start)."""
    in_single = False
    in_double = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if in_double:
            if ch == "\\" and i + 1 < len(raw):
                i += 2
                continue
            if ch == '"':
                in_double = False
        elif in_single:
            if ch == "'":
                if i + 1 < len(raw) and raw[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
        else:
            if ch == '"':
                in_double = True
            elif ch == "'":
                in_single = True
            elif ch == "#" and (i == 0 or raw[i - 1].isspace()):
                return raw[:i].rstrip()
        i += 1
    return raw.strip()


def _flow_depth(raw: str, open_ch: str, close_ch: str) -> int:
    """Return unclosed bracket depth for a YAML flow collection, ignoring quoted characters."""
    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if in_double:
            if ch == "\\" and i + 1 < len(raw):
                i += 2
                continue
            if ch == '"':
                in_double = False
        elif in_single:
            if ch == "'":
                if i + 1 < len(raw) and raw[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
        else:
            if ch == '"':
                in_double = True
            elif ch == "'":
                in_single = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth < 0:
                    raise ValueError(f"Unbalanced YAML flow collection: {raw!r}")
        i += 1
    return depth


def _parse_single_quoted_yaml(val: str) -> str:
    """Parse a complete single-quoted YAML scalar (rejecting trailing non-comment text)."""
    chars: List[str] = []
    i = 1
    while i < len(val):
        ch = val[i]
        if ch == "'":
            if i + 1 < len(val) and val[i + 1] == "'":
                chars.append("'")
                i += 2
                continue
            if i == len(val) - 1:
                return "".join(chars)
            raise ValueError(f"Unexpected trailing characters after single-quoted YAML scalar: {val!r}")
        chars.append(ch)
        i += 1
    raise ValueError(f"Unterminated single-quoted YAML scalar: {val!r}")


def _parse_yaml_scalar_stdlib(raw: str) -> Any:
    """Parse an inline YAML value into its typed Python object (stdlib fallback)."""
    val = _strip_inline_comment(raw)
    if not val:
        return None
    if val.startswith('"'):
        try:
            decoded = json.loads(val)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid double-quoted YAML scalar {val!r}: {exc}") from exc
        if not isinstance(decoded, str):
            raise ValueError(f"Expected string for double-quoted YAML scalar: {val!r}")
        return decoded
    if val.startswith("'"):
        return _parse_single_quoted_yaml(val)
    if val.startswith("["):
        if _flow_depth(val, "[", "]") != 0 or not val.endswith("]"):
            raise ValueError(f"Malformed YAML flow sequence: {val!r}")
        return [val]
    if val.startswith("{"):
        if _flow_depth(val, "{", "}") != 0 or not val.endswith("}"):
            raise ValueError(f"Malformed YAML flow mapping: {val!r}")
        return {"_raw": val}
    if val == "-" or val.startswith(("- ", "!", "&", "*", "%", "@", "`", ",", "?", ":")):
        raise ValueError(f"Unsupported YAML frontmatter value syntax: {val!r}")

    lower = val.lower()
    if lower in ("null", "~"):
        return None
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False
    if re.match(r"^[+-]?(?:0|[1-9][0-9_]*)$", val):
        return int(val.replace("_", ""), 10)
    if re.match(r"^[+-]?0x[0-9a-fA-F_]+$", val):
        return int(val.replace("_", ""), 16)
    if re.match(r"^[+-]?0o[0-7_]+$", val):
        return int(val.replace("_", ""), 8)
    if re.match(r"^[+-]?(?:[0-9][0-9_]*\.[0-9_]*(?:[eE][+-]?[0-9_]+)?|\.[0-9_]+(?:[eE][+-]?[0-9_]+)?|[0-9][0-9_]*[eE][+-]?[0-9_]+)$", val):
        return float(val.replace("_", ""))
    if re.match(r"^[+-]?\.(?:inf|nan)$", lower):
        return float(lower.replace(".", ""))
    return val


def _parse_frontmatter_stdlib(fm_text: str) -> Dict[str, Any]:
    """Strict stdlib YAML frontmatter parser preserving non-string YAML types."""
    fm_lines = fm_text.splitlines()
    data: Dict[str, Any] = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line[0].isspace():
            raise ValueError(f"Unexpected indentation in YAML frontmatter: {line!r}")
        m = re.match(r"^([A-Za-z0-9_-]+)\s*:(?:\s+(.*)|\s*)$", line)
        if not m:
            raise ValueError(f"Unsupported YAML frontmatter line: {line!r}")
        key = m.group(1)
        rest = (m.group(2) or "").strip()

        block_m = re.match(r"^([>|])([0-9]*[-+]?)(?:\s+#.*)?$", rest)
        if block_m:
            style = block_m.group(1)
            block: List[str] = []
            i += 1
            while i < len(fm_lines) and (fm_lines[i].startswith((" ", "\t")) or not fm_lines[i].strip()):
                block.append(fm_lines[i].strip())
                i += 1
            sep = " " if style == ">" else "\n"
            data[key] = sep.join(part for part in block if part).strip()
            continue

        cleaned = _strip_inline_comment(rest)
        if not cleaned:
            i += 1
            indented: List[str] = []
            while i < len(fm_lines) and (fm_lines[i].startswith((" ", "\t")) or not fm_lines[i].strip()):
                stripped = fm_lines[i].strip()
                if stripped and not stripped.startswith("#"):
                    indented.append(stripped)
                i += 1
            if not indented:
                data[key] = None
            elif indented[0] == "-" or indented[0].startswith("- "):
                data[key] = indented
            elif re.match(r"^[A-Za-z0-9_-]+\s*:", indented[0]):
                data[key] = {"_lines": indented}
            else:
                raise ValueError(f"Unsupported indented block under {key!r}: {indented[0]!r}")
            continue

        if cleaned.startswith("[") or cleaned.startswith("{"):
            open_ch = cleaned[0]
            close_ch = "]" if open_ch == "[" else "}"
            acc = cleaned
            i += 1
            while _flow_depth(acc, open_ch, close_ch) > 0:
                if i >= len(fm_lines):
                    raise ValueError(f"Unterminated YAML flow collection for {key!r}")
                acc = f"{acc} {_strip_inline_comment(fm_lines[i])}".strip()
                i += 1
            data[key] = _parse_yaml_scalar_stdlib(acc)
            continue

        val = _parse_yaml_scalar_stdlib(rest)
        i += 1
        if isinstance(val, str) and not rest.startswith(('"', "'")):
            cont: List[str] = [val]
            while i < len(fm_lines) and fm_lines[i].startswith((" ", "\t")):
                part = _strip_inline_comment(fm_lines[i])
                if part:
                    if part == "-" or part.startswith("- ") or re.match(r"^[A-Za-z0-9_-]+\s*:", part):
                        raise ValueError(f"Unsupported nested YAML structure under {key!r}: {part!r}")
                    cont.append(part)
                i += 1
            val = " ".join(cont)
        data[key] = val
    return data


def parse_frontmatter(text: str, *, require_yaml: bool = False) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter with PyYAML when available (or strict typed stdlib fallback)."""
    fm_text, body = _extract_frontmatter_block(text)
    if yaml is not None:
        try:
            parsed = yaml.safe_load(fm_text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML frontmatter: {exc}") from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError(f"YAML frontmatter must be a mapping, got {type(parsed).__name__}")
        return parsed, body
    if require_yaml:
        raise ValueError("PyYAML is required (--require-yaml) to parse SKILL.md frontmatter")
    return _parse_frontmatter_stdlib(fm_text), body


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
    require_yaml: bool = False,
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

    fm: Optional[Dict[str, Any]] = None
    try:
        fm, _ = parse_frontmatter(raw_skill, require_yaml=require_yaml)
    except ValueError as exc:
        errors.append(f"{skill_md}: {exc}")

    if fm is not None:
        if "name" not in fm:
            errors.append(f"{skill_md}: missing required frontmatter field 'name'")
        elif not isinstance(fm["name"], str):
            errors.append(
                f"{skill_md}: frontmatter 'name' must be a string, got {type(fm['name']).__name__}"
            )
        else:
            name = fm["name"]
            if len(name) > MAX_SKILL_NAME_LEN or not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", name):
                errors.append(
                    f"{skill_md}: invalid frontmatter name {name!r} "
                    f"(expected lowercase hyphenated, <= {MAX_SKILL_NAME_LEN} chars)"
                )
            elif name != skill_dir.name:
                errors.append(f"{skill_md}: frontmatter name {name!r} != directory name {skill_dir.name!r}")

        if "description" not in fm:
            errors.append(f"{skill_md}: missing required frontmatter field 'description'")
        elif not isinstance(fm["description"], str):
            errors.append(
                f"{skill_md}: frontmatter 'description' must be a string, got {type(fm['description']).__name__}"
            )
        else:
            desc = fm["description"].strip()
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


def run_validator_self_test(*, require_yaml: bool = False) -> None:
    """Verify frontmatter YAML type checks reject non-string metadata (lists, dicts, scalars)."""
    if require_yaml and yaml is None:
        raise AssertionError("PyYAML is required (--require-yaml) for validator self-test")

    def _check_skill_md(md_text: str, should_pass: bool, expected_substr: str = "") -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sdir = root / "skills" / "freetpu"
            (sdir / "templates").mkdir(parents=True)
            (sdir / "SKILL.md").write_text(md_text, encoding="utf-8")
            (sdir / "templates" / "noop.py").write_text("x = 1\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text("# Agent Skill\nInstall `skills/freetpu`.\n", encoding="utf-8")
            errs = validate_skill(root, sdir, readme, run_self_tests=False, require_yaml=require_yaml)
            if should_pass:
                assert not errs, f"Expected pass, got errors: {errs}"
            else:
                assert errs, f"Expected validation failure for:\n{md_text}"
                if expected_substr:
                    assert any(expected_substr in e for e in errs), (
                        f"Expected substring {expected_substr!r} in errors: {errs}"
                    )

    # Valid folded and quoted string descriptions pass
    _check_skill_md(
        "---\nname: freetpu\ndescription: >-\n  Valid folded string description that easily exceeds forty characters.\n---\nBody\n",
        should_pass=True,
    )
    _check_skill_md(
        '---\nname: freetpu\ndescription: "[Quoted string with brackets that easily exceeds forty characters]"\n---\nBody\n',
        should_pass=True,
    )

    # Non-string descriptions must be rejected (inline list, multi-line list, block list, mapping, scalar)
    invalid_cases = [
        (
            '---\nname: freetpu\ndescription: ["inline list entry one is quite long", "inline list entry two"]\n---\n',
            "must be a string",
        ),
        (
            '---\nname: freetpu\ndescription: [\n  "multiline flow list entry one is quite long",\n  "entry two"\n]\n---\n',
            "must be a string",
        ),
        (
            "---\nname: freetpu\ndescription:\n  - block list item one that is definitely longer than forty characters\n---\n",
            "must be a string",
        ),
        (
            '---\nname: freetpu\ndescription: {summary: "inline mapping value that is longer than forty characters"}\n---\n',
            "must be a string",
        ),
        (
            "---\nname: freetpu\ndescription:\n  summary: block mapping value that is longer than forty characters\n---\n",
            "must be a string",
        ),
        (
            "---\nname: freetpu\ndescription: 123456789012345678901234567890123456789012345\n---\n",
            "must be a string",
        ),
        (
            "---\nname: freetpu\ndescription: true\n---\n",
            "must be a string",
        ),
        (
            "---\nname: freetpu\ndescription: null\n---\n",
            "must be a string",
        ),
        (
            '---\nname: [freetpu]\ndescription: "Valid string description that is definitely longer than forty characters."\n---\n',
            "'name' must be a string",
        ),
        (
            '---\nname: freetpu\ndescription: "unclosed or extra", "quoted strings exceeding forty characters"\n---\n',
            "",
        ),
        (
            '---\nname: freetpu\ndescription: "Valid length description closed with YAML document-end dots instead of ---."\n...\nBody\n',
            "missing closing YAML frontmatter delimiter",
        ),
    ]
    for md_text, expected in invalid_cases:
        _check_skill_md(md_text, should_pass=False, expected_substr=expected)

    # Also exercise the stdlib parser directly so both PyYAML and stdlib paths stay verified.
    fm_list = _parse_frontmatter_stdlib('name: freetpu\ndescription: ["a long list item", "another long item"]\n')
    assert isinstance(fm_list["description"], list), f"stdlib parser returned {type(fm_list['description'])}"
    fm_block = _parse_frontmatter_stdlib("name: freetpu\ndescription:\n  - item one\n  - item two\n")
    assert isinstance(fm_block["description"], list), f"stdlib parser returned {type(fm_block['description'])}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate FreeTPU Agent Skill packaging and templates")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--skill-dir", type=Path, default=None)
    ap.add_argument("--readme", type=Path, default=None)
    ap.add_argument("--run-self-tests", action="store_true", help="Execute `--self-test` on every template")
    ap.add_argument("--self-test", action="store_true", help="Run validator frontmatter & type-check unit tests")
    ap.add_argument("--require-yaml", action="store_true", help="Fail if PyYAML is not installed")
    args = ap.parse_args()

    if args.self_test:
        run_validator_self_test(require_yaml=args.require_yaml)
        parser_name = "pyyaml" if yaml is not None else "stdlib"
        print(f"Skill validator self-test PASSED (parser={parser_name}).")
        return 0

    repo_root = args.repo_root.resolve()
    skill_dir = (args.skill_dir or (repo_root / "skills" / "freetpu")).resolve()
    readme_path = (args.readme or (repo_root / "README.md")).resolve()

    errors = validate_skill(
        repo_root=repo_root,
        skill_dir=skill_dir,
        readme_path=readme_path,
        run_self_tests=args.run_self_tests,
        require_yaml=args.require_yaml,
    )
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    tpl_count = len(list((skill_dir / "templates").glob("*.py")))
    parser_name = "pyyaml" if yaml is not None else "stdlib"
    print(
        f"Skill validation PASSED: {skill_dir.relative_to(repo_root)} "
        f"({tpl_count} templates, parser={parser_name}, self_tests={'ran' if args.run_self_tests else 'ast-only'})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
