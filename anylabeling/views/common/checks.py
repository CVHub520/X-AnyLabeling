from pathlib import Path
from typing import Dict
from termcolor import colored

import sys

sys.path.append(str(Path(__file__).parent.parent))

from anylabeling.app_info import __appname__, __version__, __preferred_device__
from anylabeling.views.labeling.utils.general import collect_system_info


def _print_section(data: Dict[str, str], title: str) -> None:
    print(colored(f"{title}", "cyan", attrs=["bold"]))
    print(colored("─" * 60, "cyan"))
    if data:
        max_key_len = max(len(str(k)) for k in data.keys())
        for key, value in data.items():
            key_colored = colored(
                f"{key}:".ljust(max_key_len + 1), "white", attrs=["bold"]
            )
            print(f"  {key_colored}  {value}")
    else:
        print("  No data available")
    print(colored("─" * 60, "cyan"))


def run_checks() -> None:
    try:
        app_info = {
            "App Name": __appname__,
            "App Version": __version__,
            "Preferred Device": __preferred_device__,
        }
        system_info, pkg_info = collect_system_info()

        _print_section(app_info, "Application")
        _print_section(system_info, "System")
        _print_section(pkg_info, "Packages")

    except Exception as e:
        print(colored(f"✗ Error: {e}", "red"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_checks()
