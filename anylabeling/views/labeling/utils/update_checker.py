import requests
import threading
from packaging import version

from anylabeling.app_info import __version__
from anylabeling.views.labeling.logger import logger


def check_for_updates_async(callback=None, timeout=10):
    """
    Check for updates asynchronously without blocking the main thread

    Args:
        callback: Optional callback function to receive update info
        timeout: Network request timeout in seconds
    """

    def update_check_thread():
        try:
            headers = {"Accept": "application/vnd.github.v3+json"}
            response = requests.get(
                "https://api.github.com/repos/CVHub520/X-AnyLabeling/releases/latest",
                headers=headers,
                timeout=timeout,
            )

            if response.status_code == 200:
                data = response.json()
                latest_version = data["tag_name"].lstrip("v")
                current_version = __version__

                if version.parse(latest_version) > version.parse(
                    current_version
                ):
                    update_info = {
                        "has_update": True,
                        "current_version": current_version,
                        "latest_version": latest_version,
                        "download_url": data["html_url"],
                        "release_notes": data.get("body", ""),
                        "published_at": data.get("published_at", ""),
                    }

                    logger.info(
                        "🎉 Update available: "
                        f"{current_version} → {latest_version}\n"
                        "🔧 To upgrade: `git pull origin main`\n"
                        f"🌐 Or visit: {data['html_url']}"
                    )

                    if callback:
                        callback(update_info)
                else:
                    # Already latest version
                    update_info = {
                        "has_update": False,
                        "current_version": current_version,
                        "latest_version": latest_version,
                    }
                    if callback:
                        callback(update_info)

        except Exception:
            # Silently ignore all errors - just treat as no update available
            pass

    thread = threading.Thread(target=update_check_thread, daemon=True)
    thread.start()


def check_for_updates_sync(timeout=10):
    """
    Check for updates synchronously and return update info

    Args:
        timeout: Network request timeout in seconds

    Returns:
        dict: Update info with has_update field, None if error
    """
    try:
        headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(
            "https://api.github.com/repos/CVHub520/X-AnyLabeling/releases/latest",
            headers=headers,
            timeout=timeout,
        )

        if response.status_code == 200:
            data = response.json()
            latest_version = data["tag_name"].lstrip("v")
            current_version = __version__

            return {
                "has_update": version.parse(latest_version)
                > version.parse(current_version),
                "current_version": current_version,
                "latest_version": latest_version,
                "download_url": data["html_url"],
                "release_notes": data.get("body", ""),
                "published_at": data.get("published_at", ""),
            }
        else:
            return None

    except Exception:
        return None
