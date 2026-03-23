"""
Convert JSON instances (data/anon_jsons/*.json) to MiniZinc `.dzn` files.

This version uses the repository helper `utils.read_instance()` for all data
wrangling, then exports a `.dzn` that matches `scheduling_model.mzn`.

Key shapes:
- `compatible`: array[Task,Team] of int (0/1)
- `same_allocation`: list of sets of `Task` (one set per "group")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Any

import pandas as pd  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import read_instance  # noqa: E402


def trunc_int(x: Any) -> int:
    """Mimic CPMPy usage: `int(value)` truncates towards 0 for non-negative times."""
    return int(float(x))


def format_enum_def(enum_name: str, prefix: str, n: int) -> str:
    if n <= 0:
        raise ValueError(f"{enum_name} must have positive cardinality, got {n}")
    values = ", ".join(f"{prefix}_{i}" for i in range(1, n + 1))
    # Note: no `enum` keyword here because the enum type is declared in the `.mzn`.
    return f"{enum_name} = {{{values}}};"


def format_int_list(xs: list[int]) -> str:
    return ", ".join(str(x) for x in xs)


def format_set_of_enum(cases: Iterable[str]) -> str:
    cases = list(cases)
    if not cases:
        # Empty set is legal in MiniZinc: {}.
        return "{}"
    return "{" + ", ".join(cases) + "}"


def build_instance_dzn(json_path: Path) -> str:
    tasks_df, teams_df, same_allocation_groups = read_instance(str(json_path))
    print(tasks_df[tasks_df['successors'].apply(lambda x: len(x) > 0)])

    # Stable ordering: map each input id to a compact enum case.
    task_ids = sorted(tasks_df["task_id"].tolist(), key=lambda x: str(x))
    # `utils.read_instance()` only populates `teams_df` with *unavailable* intervals.
    # For some instances it can be empty, so we derive team ids from `tasks_df`.
    all_teams_from_tasks = set()
    for team_set in tasks_df["team_ids"].tolist():
        all_teams_from_tasks |= set(team_set)
    team_ids = sorted(all_teams_from_tasks, key=lambda x: str(x))

    task_index = {tid: i for i, tid in enumerate(task_ids, start=1)}  # enum case number
    team_index = {team_id: j for j, team_id in enumerate(team_ids, start=1)}

    def task_case(tid: str) -> str:
        return f"task_{task_index[tid]}"

    def team_case(team_id: str) -> str:
        return f"team_{team_index[team_id]}"

    # Model expects `horizon`, Time = 0..horizon.
    horizon = int(trunc_int(tasks_df["due_date"].max()))

    # Arrays over enum Task.
    duration = [trunc_int(tasks_df.loc[tasks_df["task_id"] == tid, "duration"].iloc[0]) for tid in task_ids]
    released = [trunc_int(tasks_df.loc[tasks_df["task_id"] == tid, "release_date"].iloc[0]) for tid in task_ids]
    due = [trunc_int(tasks_df.loc[tasks_df["task_id"] == tid, "due_date"].iloc[0]) for tid in task_ids]
    original_start = [trunc_int(tasks_df.loc[tasks_df["task_id"] == tid, "original_start"].iloc[0]) for tid in task_ids]

    # Compatibility matrix (0/1) aligned to enum case order.
    compatible_flat: list[int] = []
    tasks_by_id = {row["task_id"]: row for _, row in tasks_df.iterrows()}
    for tid in task_ids:
        allowed_teams = tasks_by_id[tid]["team_ids"]  # set of team_id strings
        for team_id in team_ids:
            compatible_flat.append(1 if team_id in allowed_teams else 0)

    compatible = f"array2d(Task, Team, [{format_int_list(compatible_flat)}])"

    # same_allocation: list of sets of Task enum cases.
    # utils.read_instance returns a `list[set[task_id]]`.
    same_groups_cases: list[str] = []
    for group in same_allocation_groups:
        group_cases = [task_case(tid) for tid in sorted(group, key=lambda x: str(x)) if tid in task_index]
        same_groups_cases.append(format_set_of_enum(group_cases))

    n_same_groups = len(same_groups_cases)

    lines: list[str] = []
    lines.append(f"% Generated from {json_path.name}")
    lines.append(format_enum_def("Team", "team", len(team_ids)))
    lines.append(format_enum_def("Task", "task", len(task_ids)))
    lines.append("")

    lines.append(f"horizon = {horizon};")
    lines.append(f"duration = [{format_int_list(duration)}];")
    lines.append(f"released = [{format_int_list(released)}];")
    lines.append(f"due = [{format_int_list(due)}];")
    lines.append(f"original_start = [{format_int_list(original_start)}];")
    lines.append(f"compatible = {compatible};")
    lines.append("")

    # scheduling_model expects: array[1..nSameGroups] of set of Task
    lines.append(f"same_allocation = [{', '.join(same_groups_cases)}];")
    lines.append("")

    for _, row in tasks_df[tasks_df['successors'].apply(lambda x: len(x) > 0)].iterrows():
        task = task_case(row['task_id'])
        lines.append(f"    (task: {task}, successors: {format_set_of_enum([task_case(succ) for succ in row['successors']])}),")
    lines.append("];")
    
    lines.append(f"calendars = [")
    if len(teams_df) > 0: # there is some calendar data
        for team_id in team_ids:
            calendar = teams_df[teams_df["team_id"] == team_id]
            if len(calendar):
                lines.append(f"    (team: {team_case(team_id)}, start_unavailable: {int(calendar['start_unavailable'].iloc[0])}, dur_unavailable: {int(calendar['end_unavailable'].iloc[0] - calendar['start_unavailable'].iloc[0])}),")
    lines.append("];")

    return "\n".join(lines)


def convert_json_file(json_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dzn = build_instance_dzn(json_path)
    output_path.write_text(dzn, encoding="utf-8")


def iter_input_files(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON instances to MiniZinc .dzn files (utils.read_instance-based).")
    parser.add_argument("--input-dir", type=Path, default=Path("data/anon_jsons"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dzn"))
    parser.add_argument("--pattern", type=str, default="instance_*.json")
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.input_file is not None:
        json_files = [args.input_file]
    else:
        json_files = list(iter_input_files(args.input_dir, args.pattern))

    if not json_files:
        raise FileNotFoundError(f"No input JSON files found (dir={args.input_dir}, pattern={args.pattern})")

    for jp in json_files:
        out_path = args.output_dir / f"{jp.stem}.dzn"
        convert_json_file(jp, out_path, overwrite=args.overwrite)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

