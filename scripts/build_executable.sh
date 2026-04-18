#!/bin/bash
set -euo pipefail

system=${1:-}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SPEC_DIR="${ROOT_DIR}/packaging/pyinstaller/specs"
DIST_DIR="${ROOT_DIR}/dist"
export X_ANYLABELING_ROOT="${ROOT_DIR}"

package_macos_release_zip() {
    local apps app_path app_name exe_name exe_path arch_info arch zip_name zip_path sha_path

    apps=("${DIST_DIR}"/X-AnyLabeling-v*-macOS.app)
    if [ ! -e "${apps[0]}" ]; then
        echo "No macOS .app bundle found in ${DIST_DIR}."
        exit 1
    fi

    app_path=$(ls -td "${apps[@]}" | head -n 1)
    app_name=$(basename "${app_path}" .app)

    exe_name=$(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "${app_path}/Contents/Info.plist" 2>/dev/null || true)
    exe_path="${app_path}/Contents/MacOS/${exe_name}"
    if [ -z "${exe_name}" ] || [ ! -f "${exe_path}" ]; then
        echo "Failed to locate app executable in ${app_path}."
        exit 1
    fi

    arch_info=$(lipo -info "${exe_path}" 2>/dev/null || file "${exe_path}")
    if [[ "${arch_info}" == *"arm64"* && "${arch_info}" == *"x86_64"* ]]; then
        arch="universal2"
    elif [[ "${arch_info}" == *"arm64"* ]]; then
        arch="arm64"
    elif [[ "${arch_info}" == *"x86_64"* ]]; then
        arch="x86_64"
    else
        arch=$(uname -m)
    fi

    zip_name="${app_name}-${arch}-unsigned.zip"
    zip_path="${DIST_DIR}/${zip_name}"
    sha_path="${zip_path}.sha256"

    echo "Packaging macOS release artifact..."
    rm -f "${zip_path}" "${sha_path}"
    ditto -c -k --norsrc --keepParent "${app_path}" "${zip_path}"

    (
        cd "${DIST_DIR}"
        shasum -a 256 "${zip_name}" > "${zip_name}.sha256"
    )

    echo "Created release zip: ${zip_path}"
    echo "Created checksum: ${sha_path}"
}

if [ "$system" = "win-cpu" ]; then
    echo "Building Windows CPU version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-win-cpu.spec"
elif [ "$system" = "win-gpu" ];then
    echo "Building Windows GPU version..."
    export X_ANYLABELING_DEVICE=GPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-win-gpu.spec"
elif [ "$system" = "linux-cpu" ];then
    echo "Building Linux CPU version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-linux-cpu.spec"
elif [ "$system" = "linux-gpu" ];then
    echo "Building Linux GPU version..."
    export X_ANYLABELING_DEVICE=GPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-linux-gpu.spec"
elif [ "$system" = "macos" ];then
    echo "Building macOS version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-macos.spec"
    package_macos_release_zip
else
    echo "System value '$system' is not recognized."
fi
