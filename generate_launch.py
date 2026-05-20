#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib


GROUP_HEADER_RE = re.compile(r"^([A-Z0-9_]+)=\(\s*$")
FLAG_LINE_RE = re.compile(r"^(\s*)(--[A-Za-z0-9][A-Za-z0-9-]*)(?:\s+.*)?$")
CONFIG_NAME_PLACEHOLDER = "__LAUNCH_CONFIG_NAME__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate launch.sh from launch.template.sh and a TOML flag config."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="launch-config.toml",
        help="Path to the TOML config file.",
    )
    parser.add_argument(
        "--template",
        default="launch.template.sh",
        help="Path to the launch template.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the generated launch script.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, dict[str, object]]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"Invalid TOML in {path}: {exc}") from exc

    config: dict[str, dict[str, object]] = {}
    for group_name, group_values in data.items():
        if not isinstance(group_values, dict):
            raise SystemExit(
                f"Top-level entry '{group_name}' must be a TOML table of flags."
            )

        config[group_name] = {}
        for flag, value in group_values.items():
            validate_flag_name(group_name, flag)
            validate_value_type(group_name, flag, value)
            config[group_name][flag] = value

    return config


def validate_flag_name(group_name: str, flag: str) -> None:
    if not flag.startswith("--"):
        raise SystemExit(
            f"Invalid flag '{flag}' in [{group_name}]. Flag names must start with '--'."
        )


def validate_value_type(group_name: str, flag: str, value: object) -> None:
    if isinstance(value, (str, int, float, bool)):
        return

    if isinstance(value, list) and value:
        if all(isinstance(item, (str, int, float)) for item in value):
            return

    raise SystemExit(
        f"Unsupported value for '{flag}' in [{group_name}]. "
        "Use true/false, a string/number, or a non-empty array of strings/numbers."
    )


def render_flag_line(indent: str, flag: str, value: object) -> str:
    if value is True:
        return f"{indent}{flag}\n"

    if isinstance(value, list):
        rendered_values = " ".join(str(item) for item in value)
        return f"{indent}{flag} {rendered_values}\n"

    return f"{indent}{flag} {value}\n"


def sanitize_config_name(config_path: Path) -> str:
    config_label = config_path.stem or config_path.name
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", config_label).strip("-")
    if not sanitized:
        raise SystemExit(f"Could not derive a safe config name from: {config_path}")
    return sanitized


def default_output_path(config_path: Path) -> Path:
    return config_path.with_name(f"{sanitize_config_name(config_path)}_launch.sh")


def apply_group_overrides(
    body_lines: list[str], group_config: dict[str, object]
) -> list[str]:
    indent = "    "
    for line in body_lines:
        match = FLAG_LINE_RE.match(line.rstrip("\n"))
        if match:
            indent = match.group(1)
            break

    updated_lines: list[str] = []
    handled_flags: set[str] = set()

    for line in body_lines:
        match = FLAG_LINE_RE.match(line.rstrip("\n"))
        if not match:
            updated_lines.append(line)
            continue

        flag = match.group(2)
        if flag not in group_config:
            updated_lines.append(line)
            continue

        if flag in handled_flags:
            continue

        handled_flags.add(flag)
        value = group_config[flag]
        if value is False:
            continue

        updated_lines.append(render_flag_line(match.group(1), flag, value))

    for flag, value in group_config.items():
        if flag in handled_flags or value is False:
            continue
        updated_lines.append(render_flag_line(indent, flag, value))

    return updated_lines


def generate(
    template_path: Path,
    output_path: Path,
    config: dict[str, dict[str, object]],
    config_name: str,
) -> None:
    if template_path.resolve() == output_path.resolve():
        raise SystemExit("Template and output path must be different.")

    try:
        lines = template_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Template file not found: {template_path}") from exc

    available_groups: set[str] = set()
    generated: list[str] = []
    idx = 0

    while idx < len(lines):
        header_match = GROUP_HEADER_RE.match(lines[idx].rstrip("\n"))
        if not header_match:
            generated.append(lines[idx])
            idx += 1
            continue

        group_name = header_match.group(1)
        available_groups.add(group_name)
        generated.append(lines[idx])
        idx += 1

        body_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip() != ")":
            body_lines.append(lines[idx])
            idx += 1

        if idx >= len(lines):
            raise SystemExit(f"Unterminated group '{group_name}' in template {template_path}.")

        if group_name in config:
            generated.extend(apply_group_overrides(body_lines, config[group_name]))
        else:
            generated.extend(body_lines)

        generated.append(lines[idx])
        idx += 1

    unknown_groups = sorted(set(config) - available_groups)
    if unknown_groups:
        raise SystemExit(
            "Unknown config section(s): "
            + ", ".join(f"[{group}]" for group in unknown_groups)
        )

    output_text = "".join(generated).replace(CONFIG_NAME_PLACEHOLDER, config_name)
    output_path.write_text(output_text, encoding="utf-8")
    output_path.chmod(template_path.stat().st_mode)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_path = Path(args.output) if args.output else default_output_path(config_path)
    generate(
        Path(args.template),
        output_path,
        config,
        sanitize_config_name(config_path),
    )
    print(f"Generated {output_path} from {args.template} using {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
