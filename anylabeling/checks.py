from app_info import (
    __appname__,
    __version__,
    __preferred_device__
)
from views.labeling.utils.general import collect_system_info
import pprint

if __name__ == '__main__':
    app_info = {
        "App name": __appname__,
        "App version": __version__,
        "Device": __preferred_device__
    }
    system_info, pkg_info = collect_system_info()
    pp = pprint.PrettyPrinter(indent=2)

    print("Application Information:")
    pp.pprint(app_info)

    print("\nSystem Information:")
    pp.pprint(system_info)

    print("\nPackage Information:")
    pp.pprint(pkg_info)
