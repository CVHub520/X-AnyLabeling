__appname__ = "X-AnyLabeling"
__appdescription__ = "Advanced Auto Labeling Solution with Added Features"
__version__ = "3.3.0-beta"
__url__ = "https://github.com/CVHub520/X-AnyLabeling"

CLI_HELP_MSG = """
    Usage: xanylabeling [COMMAND] [OPTIONS]

    Available Commands:
        help              Show this help message
        checks            Display system and package information
        version           Show version information
        config            Show config file path

    Launch Options:
        xanylabeling                                    Launch the GUI application
        xanylabeling --filename IMAGE                   Open specific image/folder
        xanylabeling --output DIR                       Set output directory
        xanylabeling --config FILE                      Use custom config file
        xanylabeling --reset-config                     Reset Qt config

    Examples:
        1. Launch the app:
            xanylabeling

        2. Open an image:
            xanylabeling --filename /path/to/image.jpg

        3. Check system information:
            xanylabeling checks

        4. Show version:
            xanylabeling version

    For more options, use: xanylabeling --help

    Docs: https://github.com/CVHub520/X-AnyLabeling/tree/main/docs
    Examples: https://github.com/CVHub520/X-AnyLabeling/tree/main/examples/
    GitHub: https://github.com/CVHub520/X-AnyLabeling
"""


def __getattr__(name):
    if name == "__preferred_device__":
        from anylabeling.views.common.device_manager import (
            get_preferred_device,
        )

        return get_preferred_device()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
