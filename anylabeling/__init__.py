from .app_info import __appdescription__, __appname__, __version__

def __getattr__(name: str):
    if name == "checks":
        from anylabeling.views.common.checks import run_checks

        return run_checks
    if name in {"convert", "SUPPORTED_TASKS", "list_supported_tasks"}:
        from anylabeling.views.common.converter import (
            SUPPORTED_TASKS,
            list_supported_tasks,
            run_conversion,
        )

        if name == "convert":
            return run_conversion
        if name == "SUPPORTED_TASKS":
            return SUPPORTED_TASKS
        return list_supported_tasks
    raise AttributeError(name)

__all__ = (
    "__version__",
    "__appname__",
    "__appdescription__",
    "checks",
    "convert",
    "list_supported_tasks",
    "SUPPORTED_TASKS",
)
