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

package_macos_release_zip() {
    local apps app_path latest_app app_name exe_name exe_path arch_info arch zip_name zip_path sha_path

    apps=("${DIST_DIR}"/X-AnyLabeling-v*-macOS.app)
    if [ ! -e "${apps[0]}" ]; then
        fail "No macOS .app bundle found in ${DIST_DIR}."
    fi

    latest_app=${apps[0]}
    for app_path in "${apps[@]}"; do
        if [ "${app_path}" -nt "${latest_app}" ]; then
            latest_app=${app_path}
        fi
    done

    app_path=${latest_app}
    app_name=$(basename "${app_path}" .app)

    require_command ditto
    require_command file
    require_command shasum
    [ -x /usr/libexec/PlistBuddy ] || fail "Required command '/usr/libexec/PlistBuddy' was not found."

    exe_name=$(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "${app_path}/Contents/Info.plist" 2>/dev/null || true)
    exe_path="${app_path}/Contents/MacOS/${exe_name}"
    if [ -z "${exe_name}" ] || [ ! -f "${exe_path}" ]; then
        fail "Failed to locate app executable in ${app_path}."
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
    if [ "${system}" = macos-installer ]; then
        codesign --force --deep --sign "${X_ANYLABELING_MAC_SIGN_IDENTITY:--}" "${bundle}"
        codesign --verify --deep --strict "${bundle}"
    fi
}

package_update_bundle() {
    python "${SCRIPT_DIR}/package_updates.py" package --bundle "$1" --output "${DIST_DIR}"
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
    ditto -c -k --norsrc --keepParent "${DIST_DIR}/X-AnyLabeling.app" "${DIST_DIR}/${name}-Portable.zip"
    (cd "${DIST_DIR}" && shasum -a 256 "${name}.dmg" > "${name}.dmg.sha256" && shasum -a 256 "${name}-Portable.zip" > "${name}-Portable.zip.sha256")
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
        package_update_bundle "${DIST_DIR}/X-AnyLabeling"
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
        package_update_bundle "${DIST_DIR}/X-AnyLabeling"
        ;;
    macos-installer)
        export X_ANYLABELING_PACKAGE_MODE=onedir
        build_with_spec "macOS app folder" "CPU" "${SPEC_DIR}/x-anylabeling-macos.spec"
        prepare_update_bundle "${DIST_DIR}/X-AnyLabeling.app" CPU
        package_macos_dmg
        package_update_bundle "${DIST_DIR}/X-AnyLabeling.app"
        ;;
    win-cpu)
        build_with_spec "Windows CPU" "CPU" "${SPEC_DIR}/x-anylabeling-win-cpu.spec"
        ;;
    win-gpu)
        build_with_spec "Windows GPU" "GPU" "${SPEC_DIR}/x-anylabeling-win-gpu.spec"
        ;;
    linux-cpu)
        build_with_spec "Linux CPU" "CPU" "${SPEC_DIR}/x-anylabeling-linux-cpu.spec"
        ;;
    linux-gpu)
        build_with_spec "Linux GPU" "GPU" "${SPEC_DIR}/x-anylabeling-linux-gpu.spec"
        ;;
    macos)
        build_with_spec "macOS" "CPU" "${SPEC_DIR}/x-anylabeling-macos.spec"
        package_macos_release_zip
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
