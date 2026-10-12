#!/bin/bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "$0")" && pwd)
version=$(sed -n '1p' "${bundle_dir}/build-info")
variant=$(sed -n '2p' "${bundle_dir}/build-info")
[[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.-]*)$ ]] || exit 1
[[ "${variant}" =~ ^(CPU|CUDA11|CUDA12|CUDA13)$ ]] || exit 1
install_dir=${1:-"${HOME}/.local/opt/X-AnyLabeling-${variant}"}
mkdir -p "${install_dir}"
install_dir=$(cd "${install_dir}" && pwd)
if [[ "${install_dir}" = "${bundle_dir}" || "${install_dir}" = "${bundle_dir}/"* ]]; then
    echo 'Install outside the extracted bundle directory.' >&2
    exit 1
fi
case "${install_dir}" in
    *[\$\`\"\\%]*|*$'\n'*|*$'\r'*)
        echo 'The installation path contains unsupported desktop-entry characters.' >&2
        exit 1
        ;;
esac
if [ -e "${install_dir}/current" ] && [ ! -L "${install_dir}/current" ]; then
    echo 'The current installation path must be a symbolic link.' >&2
    exit 1
fi
mkdir -p "${install_dir}/versions"
stage=$(mktemp -d "${install_dir}/versions/${version}.XXXXXX")
cp -a "${bundle_dir}/." "${stage}/"
ln -s "${stage}" "${install_dir}/.current-$$"
mv -Tf "${install_dir}/.current-$$" "${install_dir}/current"
desktop_dir=${XDG_DATA_HOME:-"${HOME}/.local/share"}/applications
mkdir -p "${desktop_dir}"
cat > "${desktop_dir}/xanylabeling-${variant}.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=X-AnyLabeling (${variant})
Exec="${install_dir}/current/X-AnyLabeling"
Icon=${install_dir}/current/icon.png
Terminal=false
Categories=Graphics;
EOF
echo "Installed X-AnyLabeling ${version} (${variant}) in ${install_dir}"
echo 'Previous versions and user data have been preserved.'
