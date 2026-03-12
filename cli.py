from __future__ import annotations

import os
from pathlib import Path

from .config import DB_TABLE_OPTIONS, DEFAULT_CHORE_JAR, PROJECT_ROOT, PipelineConfig, SCREEN_OPTIONS
from .pipeline import run_pipeline


def main() -> None:
    config = prompt_for_config()
    run_pipeline(config)
    print(f"Pipeline complete. Outputs written to {config.output_dir}")


def prompt_for_config() -> PipelineConfig:
    print("MWT one-click ETL setup")
    raw_data_dir = Path(input("Raw experiment root directory: ").strip()).expanduser().resolve()
    output_dir_text = input("Output directory [./one_click_etl_output]: ").strip()
    output_dir = Path(output_dir_text).expanduser().resolve() if output_dir_text else (PROJECT_ROOT / "one_click_etl_output")

    print("Available screens:")
    for index, screen in enumerate(SCREEN_OPTIONS, start=1):
        print(f"  {index}. {screen}")
    print("  0. Custom screen name")
    screen = prompt_screen()

    run_chore = parse_bool(input("Run Chore.jar extraction first? [y/N]: ").strip() or "n")
    number_of_taps = 30
    isi_seconds = 10
    first_tap_seconds = 600
    java_heap = detect_max_java_heap() if run_chore else "13g"
    adjust_advanced = parse_bool(input("Adjust more advanced settings? [y/N]: ").strip() or "n")
    if adjust_advanced:
        number_of_taps = prompt_int("Number of taps", default=30)
        isi_seconds = prompt_int("ISI in seconds", default=10)
        first_tap_seconds = prompt_int("First tap time in seconds", default=600)
        if run_chore:
            java_heap = input(f"Java heap for Chore [{java_heap}]: ").strip() or java_heap

    upload_to_db = parse_bool(input("Upload final tables to PostgreSQL? [y/N]: ").strip() or "n")
    database_ini = None
    db_upload_mode = "append"
    replace_tables: tuple[str, ...] = ()
    if upload_to_db:
        database_ini_text = input("Path to database.ini [./database.ini]: ").strip()
        database_ini = Path(database_ini_text).expanduser().resolve() if database_ini_text else (PROJECT_ROOT / "database.ini")
        db_upload_mode = prompt_upload_mode()
        if db_upload_mode == "replace":
            replace_tables = prompt_replace_tables()
            confirm_replace_tables(replace_tables)

    return PipelineConfig(
        raw_data_dir=raw_data_dir,
        screen=screen,
        output_dir=output_dir,
        chore_jar=DEFAULT_CHORE_JAR,
        run_chore=run_chore,
        upload_to_db=upload_to_db,
        database_ini=database_ini,
        db_upload_mode=db_upload_mode,
        replace_tables=replace_tables,
        number_of_taps=number_of_taps,
        isi_seconds=isi_seconds,
        first_tap_seconds=first_tap_seconds,
        java_heap=java_heap,
    )


def parse_bool(value: str) -> bool:
    return value.lower() in {"y", "yes", "true", "1"}


def prompt_int(label: str, default: int) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            print(f"{label} must be an integer.")
            continue
        if parsed <= 0:
            print(f"{label} must be greater than 0.")
            continue
        return parsed


def detect_max_java_heap() -> str:
    total_bytes = get_total_memory_bytes()
    if total_bytes is None:
        return "13g"
    total_gib = max(1, total_bytes // (1024 ** 3))
    return f"{total_gib}g"


def get_total_memory_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf"):
            if "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                page_size = os.sysconf("SC_PAGE_SIZE")
                phys_pages = os.sysconf("SC_PHYS_PAGES")
                if isinstance(page_size, int) and isinstance(phys_pages, int) and page_size > 0 and phys_pages > 0:
                    return page_size * phys_pages
    except (ValueError, OSError, AttributeError):
        return None
    return None


def prompt_screen() -> str:
    while True:
        value = input("Screen number or 0 for custom: ").strip()
        if value == "0":
            custom_screen = input("Enter custom screen name: ").strip()
            if custom_screen:
                return custom_screen
            print("Custom screen name cannot be empty.")
            continue
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(SCREEN_OPTIONS):
                return SCREEN_OPTIONS[index - 1]
            print("Screen number is out of range.")
            continue
        if value:
            return value
        print("Please select a screen or enter a custom name.")


def prompt_upload_mode() -> str:
    while True:
        value = input("Database upload mode: [A]ppend with ON CONFLICT DO NOTHING or [R]eplace? [A]: ").strip().lower()
        if value in {"", "a", "append"}:
            return "append"
        if value in {"r", "replace"}:
            return "replace"
        print("Please enter append or replace.")


def prompt_replace_tables() -> tuple[str, ...]:
    print("Select the database tables to replace:")
    for index, table_name in enumerate(DB_TABLE_OPTIONS, start=1):
        print(f"  {index}. {table_name}")

    while True:
        raw_value = input("Enter comma-separated table numbers to replace: ").strip()
        selections = [item.strip() for item in raw_value.split(",") if item.strip()]
        if not selections:
            print("You must select at least one table.")
            continue
        try:
            indexes = sorted({int(item) for item in selections})
        except ValueError:
            print("Selections must be numbers.")
            continue
        if any(index < 1 or index > len(DB_TABLE_OPTIONS) for index in indexes):
            print("One or more selections are out of range.")
            continue
        return tuple(DB_TABLE_OPTIONS[index - 1] for index in indexes)


def confirm_replace_tables(replace_tables: tuple[str, ...]) -> None:
    table_list = ", ".join(replace_tables)

    first = input(f"Replace mode will overwrite these tables: {table_list}\nType REPLACE to continue: ").strip()
    if first != "REPLACE":
        raise SystemExit("Replace upload cancelled.")

    second = input(f"Type the number of selected tables ({len(replace_tables)}) to confirm: ").strip()
    if second != str(len(replace_tables)):
        raise SystemExit("Replace upload cancelled.")

    third = input(f"Final confirmation. Type this exact table list:\n{table_list}\n> ").strip()
    if third != table_list:
        raise SystemExit("Replace upload cancelled.")


if __name__ == "__main__":
    main()
