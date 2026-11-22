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


def _credential_patterns() -> Tuple[Tuple[str, str, str], ...]:
    # Single source of truth for both clean-room credential detection and diagnostic redaction.
    # Split literals so this validator does not trigger on its own source.
    return (
        (
            r"-----BEGIN [A-Z ]*PRIVATE " + r"KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE " + r"KEY-----",
            "private key material",
            "[REDACTED_PRIVATE_KEY]",
        ),
        (
            r"-----BEGIN [A-Z ]*PRIVATE " + r"KEY-----[\s\S]*",
            "private key material",
            "[REDACTED_UNTERMINATED_PRIVATE_KEY_BLOCK]",
        ),
        (r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "GitHub token", "[REDACTED_GITHUB_TOKEN]"),
        (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "GitHub fine-grained token", "[REDACTED_GITHUB_TOKEN]"),
        (r"\bAIza[0-9A-Za-z\-_]{30,}\b", "Google API key", "[REDACTED_API_KEY]"),
        (r"\bsk-[A-Za-z0-9_-]{20,}\b", "API secret key", "[REDACTED_API_KEY]"),
        (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key ID", "[REDACTED_AWS_KEY]"),
    )


def _forbidden_text_patterns() -> Tuple[Tuple[str, str], ...]:
    # Split literals so this validator does not trigger on its own source.
    return (
        (r"/" + r"Users/", "macOS user home path"),
        (r"bobby" + r"bones", "private username"),
        (r"saari" + r"-co", "private org reference"),
        (r"\bspark" + r"-[12]\b", "internal host reference"),
        (r"ro\.boot\.hw\." + r"soc\.id", "device-specific property"),
        (r"file" + r"://", "local file URI"),
    ) + tuple((pat, label) for pat, label, _ in _credential_patterns())


def redact_secrets(text: str) -> str:
    """Scrub detected secret tokens/keys before emitting diagnostics."""
    redacted = text
    for pat, _, repl in _credential_patterns():
        redacted = re.sub(pat, repl, redacted)
    return redacted


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
                    raise ValueError("Invalid YAML frontmatter: unbalanced flow collection")
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
            raise ValueError("Invalid YAML frontmatter: unexpected trailing characters after single-quoted scalar")
        chars.append(ch)
        i += 1
    raise ValueError("Invalid YAML frontmatter: unterminated single-quoted scalar")


def _parse_yaml_scalar_stdlib(raw: str) -> Any:
    """Parse an inline YAML value into its typed Python object (stdlib fallback)."""
    val = _strip_inline_comment(raw)
    if not val:
        return None
    if val.startswith('"'):
        try:
            decoded = json.loads(val)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid YAML frontmatter: invalid double-quoted scalar ({exc.msg})") from exc
        if not isinstance(decoded, str):
            raise ValueError("Invalid YAML frontmatter: expected string for double-quoted scalar")
        return decoded
    if val.startswith("'"):
        return _parse_single_quoted_yaml(val)
    if val.startswith("["):
        if _flow_depth(val, "[", "]") != 0 or not val.endswith("]"):
            raise ValueError("Invalid YAML frontmatter: malformed flow sequence")
        return [val]
    if val.startswith("{"):
        if _flow_depth(val, "{", "}") != 0 or not val.endswith("}"):
            raise ValueError("Invalid YAML frontmatter: malformed flow mapping")
        return {"_raw": val}
    if val == "-" or val.startswith(("- ", "!", "&", "*", "%", "@", "`", ",", "?", ":")):
        raise ValueError("Invalid YAML frontmatter: unsupported value syntax")

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
    if re.search(r":(?:\s|$)", val):
        raise ValueError("Invalid YAML frontmatter: unquoted plain scalar contains ': ' or trailing ':'")
    return val


def _parse_frontmatter_stdlib(fm_text: str) -> Dict[str, Any]:
    """Strict stdlib YAML frontmatter parser preserving non-string YAML types."""
    fm_lines = fm_text.splitlines()
    data: Dict[str, Any] = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        line_no = i + 2  # 1-based line number in SKILL.md (line 1 is opening '---')
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line[0].isspace():
            raise ValueError(f"Invalid YAML frontmatter: unexpected indentation at line {line_no}")
        m = re.match(r"^([A-Za-z0-9_-]+)\s*:(?:\s+(.*)|\s*)$", line)
        if not m:
            raise ValueError(f"Invalid YAML frontmatter: unsupported syntax at line {line_no}")
        key = m.group(1)
        if key in data:
            raise ValueError(f"Invalid YAML frontmatter: duplicate key at line {line_no}")
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
            elif re.match(r"^[A-Za-z0-9_-]+\s*:(?:\s|$)", indented[0]):
                seen_nested = set()
                for sub_line in indented:
                    sub_m = re.match(r"^([A-Za-z0-9_-]+)\s*:(?:\s|$)", sub_line)
                    if sub_m:
                        sub_key = sub_m.group(1)
                        if sub_key in seen_nested:
                            raise ValueError(f"Invalid YAML frontmatter: duplicate nested key near line {line_no}")
                        seen_nested.add(sub_key)
                data[key] = {"_lines": indented}
            else:
                # Valid YAML permits an indented scalar or flow collection on the line after `key:`.
                first_cleaned = _strip_inline_comment(indented[0])
                if first_cleaned.startswith("[") or first_cleaned.startswith("{"):
                    acc = " ".join(_strip_inline_comment(item) for item in indented).strip()
                    data[key] = _parse_yaml_scalar_stdlib(acc)
                elif len(indented) == 1:
                    data[key] = _parse_yaml_scalar_stdlib(indented[0])
                elif first_cleaned.startswith(('"', "'")):
                    acc = " ".join(_strip_inline_comment(item) for item in indented).strip()
                    data[key] = _parse_yaml_scalar_stdlib(acc)
                else:
                    parts: List[str] = []
                    for item in indented:
                        part = _strip_inline_comment(item)
                        if not part:
                            continue
                        if part == "-" or part.startswith("- ") or re.search(r":(?:\s|$)", part):
                            raise ValueError(
                                f"Invalid YAML frontmatter: invalid indented scalar near line {line_no}"
                            )
                        parts.append(part)
                    if not parts:
                        data[key] = None
                    elif len(parts) == 1:
                        data[key] = _parse_yaml_scalar_stdlib(parts[0])
                    else:
                        first_val = _parse_yaml_scalar_stdlib(parts[0])
                        if not isinstance(first_val, str):
                            first_val = parts[0]
                        data[key] = " ".join([first_val, *parts[1:]])
            continue

        if cleaned.startswith("[") or cleaned.startswith("{"):
            open_ch = cleaned[0]
            close_ch = "]" if open_ch == "[" else "}"
            acc = cleaned
            i += 1
            while _flow_depth(acc, open_ch, close_ch) > 0:
                if i >= len(fm_lines):
                    raise ValueError(f"Invalid YAML frontmatter: unterminated flow collection at line {line_no}")
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
                    if part == "-" or part.startswith("- ") or re.search(r":(?:\s|$)", part):
                        raise ValueError(
                            f"Invalid YAML frontmatter: invalid plain scalar continuation near line {i + 2}"
                        )
                    cont.append(part)
                i += 1
            val = " ".join(cont)
        data[key] = val
    return data


def _load_yaml_unique_keys(fm_text: str) -> Any:
    """Parse YAML via PyYAML SafeLoader while rejecting duplicate mapping keys."""

    class _UniqueKeySafeLoader(yaml.SafeLoader):
        def construct_mapping(self, node: Any, deep: bool = False) -> Dict[Any, Any]:
            if not isinstance(node, yaml.MappingNode):
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"expected a mapping node, but found {node.id}",
                    node.start_mark,
                )
            self.flatten_mapping(node)
            mapping: Dict[Any, Any] = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    is_dup = key in mapping
                except TypeError as exc:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found unhashable key",
                        key_node.start_mark,
                    ) from exc
                if is_dup:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
                mapping[key] = self.construct_object(value_node, deep=deep)
            return mapping

    return yaml.load(fm_text, Loader=_UniqueKeySafeLoader)


def _format_yaml_error(exc: Exception) -> str:
    """Format a YAML error with error type and location only (never raw source excerpts)."""
    err_name = type(exc).__name__
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    loc = ""
    if mark is not None and hasattr(mark, "line") and hasattr(mark, "column"):
        loc = f" at line {mark.line + 2}, col {mark.column + 1}"
    problem = getattr(exc, "problem", None)
    if isinstance(problem, str) and problem.startswith("found duplicate key"):
        return f"Invalid YAML frontmatter: duplicate key ({err_name}{loc})"
    return f"Invalid YAML frontmatter ({err_name}{loc})"


def parse_frontmatter(text: str, *, require_yaml: bool = False) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter with PyYAML when available (or strict typed stdlib fallback)."""
    fm_text, body = _extract_frontmatter_block(text)
    if yaml is not None:
        try:
            parsed = _load_yaml_unique_keys(fm_text)
        except yaml.YAMLError as exc:
            raise ValueError(_format_yaml_error(exc)) from exc
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
            safe_name = redact_secrets(name)
            if len(name) > MAX_SKILL_NAME_LEN or not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", name):
                errors.append(
                    f"{skill_md}: invalid frontmatter name {safe_name!r} "
                    f"(expected lowercase hyphenated, <= {MAX_SKILL_NAME_LEN} chars)"
                )
            elif name != skill_dir.name:
                errors.append(f"{skill_md}: frontmatter name {safe_name!r} != directory name {skill_dir.name!r}")

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
            # `colab run` prepends executable statements before script text and compiles as one cell,
            # which fails with SyntaxError if a template uses `from __future__ import ...`.
            compile("import sys\n" + src, str(rel), "exec")
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
                redacted_out = redact_secrets((proc.stderr or proc.stdout).strip())
                errors.append(
                    f"Template self-test failed ({rel}, rc={proc.returncode}): "
                    f"{redacted_out[-400:]}"
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

    return [redact_secrets(err) for err in errors]


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
    _check_skill_md(
        '---\nname: freetpu\ndescription: "Build and verify TPU kernels: run parity before timing."\n---\nBody\n',
        should_pass=True,
    )
    _check_skill_md(
        "---\nname: freetpu\ndescription: Build and verify TPU kernels via https://example.com before timing.\n---\nBody\n",
        should_pass=True,
    )
    _check_skill_md(
        "---\nname: freetpu\ndescription:\n  Indented plain string description that easily exceeds forty characters.\n---\nBody\n",
        should_pass=True,
    )
    _check_skill_md(
        "---\nname: freetpu\ndescription:\n  Indented plain string description\n  continued on the next line cleanly.\n---\nBody\n",
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
        (
            '---\nname: freetpu\ndescription: "First valid description that is definitely longer than forty characters."\ndescription: "Second valid description that is definitely longer than forty characters."\n---\n',
            "duplicate key",
        ),
        (
            '---\nname: freetpu\nname: freetpu\ndescription: "Valid description that is definitely longer than forty characters."\n---\n',
            "duplicate key",
        ),
        (
            "---\nname: freetpu\ndescription: Build and verify TPU kernels: run parity before timing.\n---\n",
            "Invalid YAML frontmatter",
        ),
        (
            "---\nname: freetpu\ndescription: Build and verify TPU kernels with parity before timing:\n---\n",
            "Invalid YAML frontmatter",
        ),
    ]
    for md_text, expected in invalid_cases:
        _check_skill_md(md_text, should_pass=False, expected_substr=expected)

    # Also exercise the stdlib parser directly so both PyYAML and stdlib paths stay verified.
    fm_list = _parse_frontmatter_stdlib('name: freetpu\ndescription: ["a long list item", "another long item"]\n')
    assert isinstance(fm_list["description"], list), f"stdlib parser returned {type(fm_list['description'])}"
    fm_block = _parse_frontmatter_stdlib("name: freetpu\ndescription:\n  - item one\n  - item two\n")
    assert isinstance(fm_block["description"], list), f"stdlib parser returned {type(fm_block['description'])}"
    try:
        _parse_frontmatter_stdlib(
            'name: freetpu\n'
            'description: "First valid description that is definitely longer than forty characters."\n'
            'description: "Second valid description that is definitely longer than forty characters."\n'
        )
    except ValueError as exc:
        assert "duplicate key" in str(exc), f"Unexpected duplicate-key error: {exc}"
    else:
        raise AssertionError("stdlib parser failed to reject duplicate description key")
    try:
        _parse_frontmatter_stdlib(
            "name: freetpu\ndescription: Build and verify TPU kernels: run parity before timing.\n"
        )
    except ValueError as exc:
        assert "Invalid YAML frontmatter" in str(exc), f"Unexpected plain-scalar error: {exc}"
    else:
        raise AssertionError("stdlib parser failed to reject plain scalar containing ': '")

    fm_indented = _parse_frontmatter_stdlib(
        "name: freetpu\ndescription:\n  Indented plain string.\n"
    )
    assert fm_indented["description"] == "Indented plain string.", f"Unexpected indented value: {fm_indented!r}"
    fm_indented_multi = _parse_frontmatter_stdlib(
        "name: freetpu\ndescription:\n  Indented plain string\n  across two lines.\n"
    )
    assert fm_indented_multi["description"] == "Indented plain string across two lines.", fm_indented_multi

    # Verify accidentally pasted secrets in frontmatter are redacted from all emitted diagnostics.
    fake_token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdir = root / "skills" / "freetpu"
        (sdir / "templates").mkdir(parents=True)
        (sdir / "SKILL.md").write_text(
            f"---\nname: {fake_token}\ndescription: Valid string description that is definitely longer than forty characters.\n---\n",
            encoding="utf-8",
        )
        (sdir / "templates" / "noop.py").write_text("x = 1\n", encoding="utf-8")
        readme = root / "README.md"
        readme.write_text("# Agent Skill\nInstall `skills/freetpu`.\n", encoding="utf-8")
        errs = validate_skill(root, sdir, readme, run_self_tests=False, require_yaml=require_yaml)
        joined = "\n".join(errs)
        assert errs and fake_token not in joined, f"Secret leaked in diagnostics: {joined!r}"
        assert "[REDACTED_GITHUB_TOKEN]" in joined, f"Expected redaction marker in: {joined!r}"
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--repo-root",
                str(root),
                "--skill-dir",
                str(sdir),
                "--readme",
                str(readme),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        assert fake_token not in proc.stdout and fake_token not in proc.stderr, (
            f"Secret leaked in CLI output: stdout={proc.stdout!r}, stderr={proc.stderr!r}"
        )
        assert "[REDACTED_GITHUB_TOKEN]" in proc.stderr, f"Expected redaction marker in stderr: {proc.stderr!r}"

    # Verify every recognized credential family is caught in Markdown body text AND redacted.
    body_secrets = (
        ("gho_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4", "GitHub token", "[REDACTED_GITHUB_TOKEN]"),
        ("sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4", "API secret key", "[REDACTED_API_KEY]"),
        ("AKIA" + "0123456789ABCDEF", "AWS access key ID", "[REDACTED_AWS_KEY]"),
    )
    for tok, label, placeholder in body_secrets:
        assert redact_secrets(f"token={tok}") == f"token={placeholder}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sdir = root / "skills" / "freetpu"
            (sdir / "templates").mkdir(parents=True)
            (sdir / "SKILL.md").write_text(
                "---\nname: freetpu\ndescription: Valid string description that is definitely longer than forty characters.\n---\n"
                f"Body accidentally containing {tok}\n",
                encoding="utf-8",
            )
            (sdir / "templates" / "noop.py").write_text("x = 1\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text("# Agent Skill\nInstall `skills/freetpu`.\n", encoding="utf-8")
            errs = validate_skill(root, sdir, readme, run_self_tests=False, require_yaml=require_yaml)
            joined = "\n".join(errs)
            assert any(label in e for e in errs), f"Expected clean-room detection for {label}, got: {errs}"
            assert tok not in joined, f"Credential leaked in diagnostic: {joined!r}"

    # Verify failed self-test output is redacted in FULL before the [-400:] tail cut.
    pem_body_secret = "MIIE_SYNTHETIC_SECRET_BODY_DO_NOT_LEAK_9876543210"
    tok_tail_secret = "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdir = root / "skills" / "freetpu"
        (sdir / "templates").mkdir(parents=True)
        (sdir / "SKILL.md").write_text(
            "---\nname: freetpu\ndescription: Valid string description that is definitely longer than forty characters.\n---\n",
            encoding="utf-8",
        )
        # Emit a PEM block whose header starts before the last 400 chars and a token straddling the -400 cutoff.
        (sdir / "templates" / "leaky_fail.py").write_text(
            "import sys\n"
            f"pem = '-----BEGIN RSA PRIVATE ' + 'KEY-----\\n' + ('A' * 300) + '{pem_body_secret}\\n-----END RSA PRIVATE ' + 'KEY-----\\n'\n"
            f"tok = 'ghp_' + '{tok_tail_secret}'\n"
            "sys.stderr.write(pem + tok + ('x' * 390))\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        readme = root / "README.md"
        readme.write_text("# Agent Skill\nInstall `skills/freetpu`.\n", encoding="utf-8")
        errs = validate_skill(root, sdir, readme, run_self_tests=True, require_yaml=require_yaml)
        joined = "\n".join(errs)
        assert any("Template self-test failed" in e for e in errs), f"Expected self-test failure, got: {errs}"
        assert pem_body_secret not in joined, f"PEM body leaked across [-400:] cut: {joined!r}"
        assert tok_tail_secret not in joined, f"Token suffix leaked across [-400:] cut: {joined!r}"
        assert "[REDACTED_GITHUB_TOKEN]" in joined, f"Expected token redaction marker in: {joined!r}"

    # Verify unterminated private-key blocks (no END marker) suppress all trailing body bytes.
    unterm_secret = "UNTERMINATED_KEY_BODY_SECRET_6677889900"
    unterm_text = "prefix\n-----BEGIN RSA PRIVATE " + "KEY-----\n" + unterm_secret + "\nMORE_KEY_BYTES\n"
    redacted_unterm = redact_secrets(unterm_text)
    assert unterm_secret not in redacted_unterm and "MORE_KEY_BYTES" not in redacted_unterm, redacted_unterm
    assert "[REDACTED_UNTERMINATED_PRIVATE_KEY_BLOCK]" in redacted_unterm, redacted_unterm
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdir = root / "skills" / "freetpu"
        (sdir / "templates").mkdir(parents=True)
        (sdir / "SKILL.md").write_text(
            "---\nname: freetpu\ndescription: Valid string description that is definitely longer than forty characters.\n---\n",
            encoding="utf-8",
        )
        (sdir / "templates" / "unterm_key_fail.py").write_text(
            "import sys\n"
            f"sys.stderr.write('err\\n-----BEGIN PRIVATE ' + 'KEY-----\\n{unterm_secret}\\nTAIL_SECRET_LINE\\n')\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        readme = root / "README.md"
        readme.write_text("# Agent Skill\nInstall `skills/freetpu`.\n", encoding="utf-8")
        errs = validate_skill(root, sdir, readme, run_self_tests=True, require_yaml=require_yaml)
        joined = "\n".join(errs)
        assert any("Template self-test failed" in e for e in errs), errs
        assert unterm_secret not in joined and "TAIL_SECRET_LINE" not in joined, (
            f"Unterminated private-key body leaked in diagnostic: {joined!r}"
        )
        assert "[REDACTED_UNTERMINATED_PRIVATE_KEY_BLOCK]" in joined, joined

    # Verify malformed YAML frontmatter with a pasted secret never leaks raw source excerpts or clipped suffixes.
    clipped_suffix = "CLIPPED_SECRET_SUFFIX_554433221100"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdir = root / "skills" / "freetpu"
        (sdir / "templates").mkdir(parents=True)
        (sdir / "SKILL.md").write_text(
            "---\n"
            "name: freetpu\n"
            f'description: "prefix {"x" * 60} ghp_{clipped_suffix}" trailing_junk\n'
            "---\n",
            encoding="utf-8",
        )
        (sdir / "templates" / "noop.py").write_text("x = 1\n", encoding="utf-8")
        readme = root / "README.md"
        readme.write_text("# Agent Skill\nInstall `skills/freetpu`.\n", encoding="utf-8")
        errs = validate_skill(root, sdir, readme, run_self_tests=False, require_yaml=require_yaml)
        joined = "\n".join(errs)
        assert errs and clipped_suffix not in joined, f"Raw YAML excerpt leaked secret suffix: {joined!r}"

    # Also verify _format_yaml_error directly with a MarkedYAMLError-shaped exception whose __str__ leaks a clipped excerpt.
    class _FakeMark:
        line = 1
        column = 12

    class _FakeMarkedYAMLError(Exception):
        problem = "expected <block end>, but found '<scalar>'"
        problem_mark = _FakeMark()

        def __str__(self) -> str:
            return f"scanner error\n  ...{clipped_suffix}\" trailing_junk\n"

    formatted = _format_yaml_error(_FakeMarkedYAMLError())
    assert clipped_suffix not in formatted, f"_format_yaml_error leaked excerpt: {formatted!r}"
    assert "line 3, col 13" in formatted and "_FakeMarkedYAMLError" in formatted, formatted

    # Verify templates using `from __future__ import annotations` are rejected (breaks `colab run` cell prepending).
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdir = root / "skills" / "freetpu"
        (sdir / "templates").mkdir(parents=True)
        (sdir / "SKILL.md").write_text(
            "---\nname: freetpu\ndescription: Valid string description that is definitely longer than forty characters.\n---\n",
            encoding="utf-8",
        )
        (sdir / "templates" / "bad_future.py").write_text(
            "from __future__ import annotations\nx: int = 1\n", encoding="utf-8"
        )
        readme = root / "README.md"
        readme.write_text("# Agent Skill\nInstall `skills/freetpu`.\n", encoding="utf-8")
        errs = validate_skill(root, sdir, readme, run_self_tests=False, require_yaml=require_yaml)
        assert any("Syntax error in template" in e for e in errs), (
            f"Expected SyntaxError rejection for future import template, got: {errs}"
        )


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
        if not args.run_self_tests:
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
            print(f"ERROR: {redact_secrets(err)}", file=sys.stderr)
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
