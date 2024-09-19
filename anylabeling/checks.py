import logging
import pprint
from app_info import __appname__, __version__, __preferred_device__
from views.labeling.utils.general import (
    collect_system_info,
    format_bold,
    format_color,
    indent_text,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    app_info = {
        "App name": __appname__,
        "App version": __version__,
        "Device": __preferred_device__,
    }
    system_info, pkg_info = collect_system_info()

    logger.info(format_color(format_bold("Application Information:"), '36'))
    logger.info(indent_text(pprint.pformat(app_info)))

    logger.info(format_color(format_bold("\nSystem Information:"), '36'))
    logger.info(indent_text(pprint.pformat(system_info)))

    logger.info(format_color(format_bold("\nPackage Information:"), '36'))
    logger.info(indent_text(pprint.pformat(pkg_info)))