"""Validate Lab 18 deliverables before submission."""

import json
import os
import re
import subprocess
import sys

SOURCE_FILES = [
    "src/m1_chunking.py",
    "src/m2_search.py",
    "src/m3_rerank.py",
    "src/m4_eval.py",
    "src/m5_enrichment.py",
    "src/pipeline.py",
]


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  ✅ {path}")
        return True
    if required:
        print(f"  ❌ THIẾU: {path}")
        return False
    print(f"  ⚠️  Optional: {path}")
    return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"  ❌ {path} — {exc}")
        return False

    missing = [key for key in required_keys if key not in data]
    if missing:
        print(f"  ❌ {path} thiếu keys: {missing}")
        return False
    print(f"  ✅ {path} — keys OK")
    return True


def check_text_not_template(path: str, forbidden_markers: tuple[str, ...]) -> bool:
    try:
        with open(path, encoding="utf-8") as file_obj:
            text = file_obj.read()
    except OSError as exc:
        print(f"  ❌ {path} — {exc}")
        return False
    found = [marker for marker in forbidden_markers if marker in text]
    if found:
        print(f"  ❌ {path} vẫn còn template markers: {found}")
        return False
    print(f"  ✅ {path} — nội dung đã được điền")
    return True


def check_todos() -> int:
    """Count unresolved starter markers in graded module files."""
    count = 0
    for path in SOURCE_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as file_obj:
            count += sum(1 for line in file_obj if "# TODO:" in line)
    return count


def run_tests() -> tuple[int, int, int]:
    """Run pytest and return (passed, failed, exit_code)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  ❌ pytest error: {exc}")
        return 0, 0, 1

    output = f"{result.stdout}\n{result.stderr}"
    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    if result.returncode != 0:
        print(output[-3000:])
    return passed, failed, result.returncode


def validate() -> int:
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    print("📁 Source code:")
    for path in SOURCE_FILES:
        if not check_file(path):
            errors += 1

    print("\n📊 Reports:")
    for report_path in ("reports/ragas_report.json", "ragas_report.json"):
        if not check_file(report_path) or not check_json(
            report_path, ["aggregate", "num_questions", "failures"]
        ):
            errors += 1

    baseline_path = "reports/naive_baseline_report.json"
    if not check_file(baseline_path) or not check_json(
        baseline_path, ["aggregate", "num_questions"]
    ):
        errors += 1

    latency_path = "reports/latency_report.json"
    if not check_file(latency_path) or not check_json(latency_path, ["unit", "timings"]):
        errors += 1
    if not check_file("reports/latency_report.md"):
        errors += 1

    print("\n📝 Analysis:")
    failure_path = "analysis/failure_analysis.md"
    if not check_file(failure_path):
        errors += 1
    elif not check_text_not_template(
        failure_path,
        ("[Tên nhóm]", "[Tên 1", "(copy template)", "**Question:**\n- **Expected:**"),
    ):
        errors += 1

    group_path = "analysis/group_report.md"
    if not check_file(group_path):
        errors += 1
    elif not check_text_not_template(
        group_path,
        ("**Nhóm:** [Tên]", "| Faithfulness | | | |", "☐", "**Biggest improvement:**\n"),
    ):
        errors += 1

    print("\n👤 Individual reflection:")
    assignment_reflection = "analysis/reflection_ChuNguyenTuanAnh.md"
    if not check_file(assignment_reflection):
        errors += 1

    reflection_dir = "analysis/reflections"
    reflections = []
    if os.path.isdir(reflection_dir):
        reflections = sorted(
            filename
            for filename in os.listdir(reflection_dir)
            if filename.startswith("reflection_") and filename.endswith(".md")
        )
    if not reflections:
        print(f"  ❌ Chưa có reflection cá nhân trong {reflection_dir}/")
        errors += 1
    else:
        for reflection in reflections:
            print(f"  ✅ {reflection_dir}/{reflection}")

    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO marker trong graded modules")
    else:
        print(f"  ❌ Còn {todo_count} TODO marker")
        errors += 1

    print("\n🧪 Auto-tests:")
    passed, failed, test_exit = run_tests()
    if test_exit == 0 and failed == 0 and passed > 0:
        print(f"  ✅ {passed} tests passed, 0 failed")
    else:
        print(f"  ❌ pytest exit={test_exit}, passed={passed}, failed={failed}")
        errors += 1

    print("\n" + "=" * 50)
    if errors == 0:
        print("🚀 Bài lab sẵn sàng để nộp!")
    else:
        print(f"❌ Có {errors} lỗi cần sửa trước khi nộp.")
    print("=" * 50)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(validate())
