from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tools.clang_tidy import run_clang_tidy
from tools.cmake_runner import build_cmake, configure_cmake
from tools.glslang import run_glslang_validator
from tools.spirv_validator import run_spirv_val


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    stdout: str
    stderr: str
    faults: list[str]


def _faults(results: Iterable) -> list[str]:
    return [result.fault for result in results if getattr(result, "fault", None)]


def validate_cpp_source(source_file: str | Path, build_dir: str | Path, parallel_workers: int | None = None) -> ValidationResult:
    tidy = run_clang_tidy(source_file, build_dir=build_dir)
    if tidy.fault:
        return ValidationResult(False, tidy.stdout, tidy.stderr, [tidy.fault])
    configure = configure_cmake(Path(source_file).parents[1], build_dir)
    if configure.fault:
        return ValidationResult(False, configure.stdout, configure.stderr, [configure.fault])
    build = build_cmake(build_dir, parallel_workers=parallel_workers)
    return ValidationResult(
        passed=build.returncode == 0,
        stdout="\n".join(filter(None, [tidy.stdout, configure.stdout, build.stdout])),
        stderr="\n".join(filter(None, [tidy.stderr, configure.stderr, build.stderr])),
        faults=_faults([tidy, configure, build]),
    )


def validate_spirv(shader_files: Iterable[str | Path]) -> ValidationResult:
    results = [run_spirv_val(path) for path in shader_files]
    return ValidationResult(False if any(result.fault for result in results) else True, "", "", _faults(results))


def validate_glsl(shader_files: Iterable[str | Path]) -> ValidationResult:
    results = [run_glslang_validator(path) for path in shader_files]
    return ValidationResult(False if any(result.fault for result in results) else True, "", "", _faults(results))
