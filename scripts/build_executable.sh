#!/bin/bash
set -euo pipefail

system=${1:-}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SPEC_DIR="${ROOT_DIR}/packaging/pyinstaller/specs"
DIST_DIR="${ROOT_DIR}/dist"
export X_ANYLABELING_ROOT="${ROOT_DIR}"

usage() {
    cat <<EOF
Usage: $(basename "$0") {win-cpu|win-gpu|linux-cpu|linux-gpu|macos}
       $(basename "$0") {win-cpu-installer|win-gpu-installer|linux-cpu-bundle|linux-gpu-bundle|macos-installer}

Build X-AnyLabeling executable artifacts with PyInstaller.
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command '$1' was not found."
}

require_file() {
    [ -f "$1" ] || fail "Required file '$1' was not found."
}

build_with_spec() {
    local label device spec_path

    label=$1
    device=$2
    spec_path=$3

    require_command pyinstaller
    require_file "${spec_path}"

    echo "Building ${label} version..."
    export X_ANYLABELING_DEVICE="${device}"
    pyinstaller --noconfirm "${spec_path}"
}

package_linux_bundle() {
    local version variant name
    version=$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "${ROOT_DIR}/anylabeling/app_info.py")
    variant=${X_ANYLABELING_VARIANT:-CPU}
    [[ "${variant}" =~ ^(CPU|CUDA11|CUDA12|CUDA13)$ ]] || fail "Invalid X_ANYLABELING_VARIANT."
    name="X-AnyLabeling-v${version}-Linux-${variant}-Portable"
    require_file "${DIST_DIR}/X-AnyLabeling/X-AnyLabeling"
    cp "${ROOT_DIR}/packaging/linux/install.sh" "${DIST_DIR}/X-AnyLabeling/install.sh"
    cp "${ROOT_DIR}/anylabeling/resources/images/icon.png" "${DIST_DIR}/X-AnyLabeling/icon.png"
    printf '%s\n%s\n' "${version}" "${variant}" > "${DIST_DIR}/X-AnyLabeling/build-info"
    chmod +x "${DIST_DIR}/X-AnyLabeling/install.sh"
    prepare_update_bundle "${DIST_DIR}/X-AnyLabeling" "${variant}"
    tar -czf "${DIST_DIR}/${name}.tar.gz" -C "${DIST_DIR}" X-AnyLabeling
    (cd "${DIST_DIR}" && sha256sum "${name}.tar.gz" > "${name}.tar.gz.sha256")
}

prepare_update_bundle() {
    local bundle variant helper_name
    bundle=$1
    variant=$2
    helper_name=xanylabeling-updater
    if [[ "${system}" = win-* ]]; then
        helper_name=xanylabeling-updater.exe
    fi
    pyinstaller --noconfirm --clean --onefile --noupx \
        --paths "${ROOT_DIR}/anylabeling/services/updates" \
        --name xanylabeling-updater --specpath "${ROOT_DIR}/build/updater-spec" \
        --workpath "${ROOT_DIR}/build/updater" --distpath "${ROOT_DIR}/build/updater-dist" \
        "${ROOT_DIR}/anylabeling/services/updates/installer.py"
    "${ROOT_DIR}/build/updater-dist/${helper_name}" --help > /dev/null
    python "${SCRIPT_DIR}/package_updates.py" prepare --bundle "${bundle}" \
        --variant "${variant}" --helper "${ROOT_DIR}/build/updater-dist/${helper_name}"
    if [[ "${system}" = macos* ]]; then
        codesign --force --deep --sign "${X_ANYLABELING_MAC_SIGN_IDENTITY:--}" "${bundle}"
        codesign --verify --deep --strict "${bundle}"
    fi
}

package_update_bundle() {
    python "${SCRIPT_DIR}/package_updates.py" package --bundle "$1" --artifact "$2" --output "${DIST_DIR}"
}

package_macos_dmg() {
    local version stage name
    version=$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "${ROOT_DIR}/anylabeling/app_info.py")
    name="X-AnyLabeling-v${version}-macOS-$(uname -m)-unsigned"
    require_command hdiutil
    require_command ditto
    [ -d "${DIST_DIR}/X-AnyLabeling.app" ] || fail "No macOS app bundle found."
    stage=$(mktemp -d)
    ditto "${DIST_DIR}/X-AnyLabeling.app" "${stage}/X-AnyLabeling.app"
    ln -s /Applications "${stage}/Applications"
    hdiutil create -ov -fs HFS+ -volname X-AnyLabeling -srcfolder "${stage}" -format UDZO "${DIST_DIR}/${name}.dmg"
    rm -rf "${stage}"
    (cd "${DIST_DIR}" && shasum -a 256 "${name}.dmg" > "${name}.dmg.sha256")
}

case "${system}" in
    win-cpu-installer|win-gpu-installer)
        export X_ANYLABELING_PACKAGE_MODE=onedir
        variant=${X_ANYLABELING_VARIANT:-CPU}
        if [ "${system}" = win-gpu-installer ]; then
            variant=${X_ANYLABELING_VARIANT:-CUDA12}
            [[ "${variant}" =~ ^CUDA(11|12|13)$ ]] || fail "GPU installers require a CUDA variant."
            build_with_spec "Windows GPU folder" "GPU" "${SPEC_DIR}/x-anylabeling-win-gpu.spec"
        else
            [ "${variant}" = CPU ] || fail "CPU installers require the CPU variant."
            build_with_spec "Windows CPU folder" "CPU" "${SPEC_DIR}/x-anylabeling-win-cpu.spec"
        fi
        prepare_update_bundle "${DIST_DIR}/X-AnyLabeling" "${variant}"
        require_command powershell.exe
        powershell.exe -NoProfile -ExecutionPolicy Bypass -File "${SCRIPT_DIR}/package_windows.ps1" -Variant "${variant}"
        version=$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "${ROOT_DIR}/anylabeling/app_info.py")
        package_update_bundle "${DIST_DIR}/X-AnyLabeling" "${DIST_DIR}/X-AnyLabeling-v${version}-Windows-${variant}-Setup.exe"
        ;;
    linux-cpu-bundle|linux-gpu-bundle)
        export X_ANYLABELING_PACKAGE_MODE=onedir
        if [ "${system}" = linux-gpu-bundle ]; then
            export X_ANYLABELING_VARIANT=${X_ANYLABELING_VARIANT:-CUDA12}
            [[ "${X_ANYLABELING_VARIANT}" =~ ^CUDA(11|12|13)$ ]] || fail "GPU bundles require a CUDA variant."
            build_with_spec "Linux GPU folder" "GPU" "${SPEC_DIR}/x-anylabeling-linux-gpu.spec"
        else
            export X_ANYLABELING_VARIANT=${X_ANYLABELING_VARIANT:-CPU}
            [ "${X_ANYLABELING_VARIANT}" = CPU ] || fail "CPU bundles require the CPU variant."
            build_with_spec "Linux CPU folder" "CPU" "${SPEC_DIR}/x-anylabeling-linux-cpu.spec"
        fi
        package_linux_bundle
        ;;
    macos|macos-installer)
        export X_ANYLABELING_PACKAGE_MODE=onedir
        build_with_spec "macOS app folder" "CPU" "${SPEC_DIR}/x-anylabeling-macos.spec"
        prepare_update_bundle "${DIST_DIR}/X-AnyLabeling.app" CPU
        package_macos_dmg
        version=$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "${ROOT_DIR}/anylabeling/app_info.py")
        package_update_bundle "${DIST_DIR}/X-AnyLabeling.app" "${DIST_DIR}/X-AnyLabeling-v${version}-macOS-$(uname -m)-unsigned.dmg"
        ;;
    win-cpu)
        build_with_spec "Windows CPU" "CPU" "${SPEC_DIR}/x-anylabeling-win-cpu.spec"
        ;;
    win-gpu)
        build_with_spec "Windows GPU" "GPU" "${SPEC_DIR}/x-anylabeling-win-gpu.spec"
        ;;
    linux-cpu|linux-gpu)
        export X_ANYLABELING_PACKAGE_MODE=onefile
        variant=${X_ANYLABELING_VARIANT:-CPU}
        device=CPU
        if [ "${system}" = linux-gpu ]; then
            variant=${X_ANYLABELING_VARIANT:-CUDA12}
            [[ "${variant}" =~ ^CUDA(11|12|13)$ ]] || fail "GPU builds require a CUDA variant."
            device=GPU
        else
            [ "${variant}" = CPU ] || fail "CPU builds require the CPU variant."
        fi
        export X_ANYLABELING_UPDATE_RESOURCES="${ROOT_DIR}/build/linux-update-resources"
        mkdir -p "${X_ANYLABELING_UPDATE_RESOURCES}"
        prepare_update_bundle "${X_ANYLABELING_UPDATE_RESOURCES}" "${variant}"
        build_with_spec "Linux ${variant}" "${device}" "${SPEC_DIR}/x-anylabeling-${system}.spec"
        version=$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "${ROOT_DIR}/anylabeling/app_info.py")
        artifact="${DIST_DIR}/X-AnyLabeling-v${version}-Linux-${variant}"
        if [ "${device}" = GPU ]; then
            mv "${DIST_DIR}/X-AnyLabeling-v${version}-Linux-GPU" "${artifact}"
        fi
        package_update_bundle "${X_ANYLABELING_UPDATE_RESOURCES}" "${artifact}"
        ;;
    -h|--help|help)
        usage
        ;;
    "")
        usage
        fail "Missing system value."
        ;;
    *)
        usage
        fail "System value '${system}' is not recognized."
        ;;
esac
