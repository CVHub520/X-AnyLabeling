from .app_info import __appdescription__, __appname__, __version__

from anylabeling.views.common.checks import run_checks as checks

__all__ = (
    "__version__",
    "checks",
)
