# Installation and automatic updates

This packaging path supports Windows, Linux, and macOS alongside the existing
single-file releases. Installed builds with a pinned update public key check for
updates on startup, show a quiet toolbar indicator, download a verified package, and restart
through an independent updater. Source checkouts and older single-file builds
retain their existing update behavior.

The release workflow builds installation packages alongside the existing portable
executables. Windows and Linux support CPU, CUDA11, CUDA12, and CUDA13; macOS uses CPU.
Builds without `UPDATE_PUBLIC_KEY` have automatic installation disabled.
This prevents accepting untrusted update metadata.

## Local builds

Build on the destination operating system with the matching Python dependencies
installed. Run these commands from the repository root in Bash (Git Bash on
Windows):

```bash
bash scripts/build_executable.sh win-cpu-installer
bash scripts/build_executable.sh linux-cpu-bundle
bash scripts/build_executable.sh macos-installer
```

Windows requires Inno Setup 6 and Windows PowerShell. GPU builds require the
matching ONNX Runtime dependencies and an explicit variant:

```bash
X_ANYLABELING_VARIANT=CUDA12 bash scripts/build_executable.sh win-gpu-installer
X_ANYLABELING_VARIANT=CUDA12 bash scripts/build_executable.sh linux-gpu-bundle
```

Use the `gpu-cu11`, `gpu`, or `gpu-cu13` dependency extra respectively for CUDA
11, 12, or 13. The variant labels packages; it does not install CUDA dependencies.
The original `win-cpu`, `win-gpu`, `linux-cpu`, `linux-gpu`, and `macos` targets
continue to produce their existing packages.

## Installation and upgrades

### Processor architectures

macOS has separate Apple Silicon (`arm64`) and Intel (`x86_64`) release jobs.
Each job uses native Python and produces its own DMG, portable ZIP, and update
archive. Both need native installation/update validation before public release.

Windows packages currently contain x64 binaries. Windows 11 on ARM can run x64
applications through emulation; use the CPU package for compatibility testing.
Update selection follows the Python executable architecture, so an emulated x64
application continues to select x64 update packages. This path still requires
validation on a Windows ARM device.

Native Windows ARM64 packages are not configured: `opencv-contrib-python-headless`
and `pyclipper` currently have no compatible ARM64 wheels for the Python 3.12
build. Providing a native package requires building and validating these
dependencies from source; the remaining CPU/development dependency set resolves
with ARM64 wheels. CUDA variants are not proposed for Windows ARM.

**Windows:** Run `*-Setup.exe`. The default location is under the current user's
LocalAppData, without requiring administrator privileges. Choose another writable
directory if needed. A Start menu shortcut and uninstall entry are created;
a desktop shortcut is optional. CPU and CUDA variants use separate application
identities and default directories. Close the app before running a newer installer
of the same variant; the installer remembers the previous directory. The portable
ZIP contains `X-AnyLabeling.exe` and `_internal`; keep them together.

**Linux:** Extract `*-Portable.tar.gz` and run `X-AnyLabeling/X-AnyLabeling`, or
run `bash X-AnyLabeling/install.sh [INSTALL_DIRECTORY]`. The installer defaults
to `~/.local/opt/X-AnyLabeling-CPU` (or the CUDA variant), installs a versioned
copy, and creates an application menu entry under the XDG data directory. Running
the installer from a new bundle switches the `current` symbolic link after copying
the package. Old versions are retained for manual rollback. To uninstall, remove
the dedicated installation directory and its
`~/.local/share/applications/xanylabeling-CPU.desktop` entry (adjust for the
variant and `XDG_DATA_HOME`). Builds on Ubuntu 22.04 still require a compatible
Linux distribution and system graphics libraries; this is not a universal package.

**macOS:** Open the DMG and drag `X-AnyLabeling.app` into Applications. Quit the
existing app before replacing it with a newer bundle. The executable and app
names are stable in this packaging mode. The ZIP provides an alternative download.
Packages are unsigned and not notarized; Gatekeeper behavior must be checked on
a real Mac before public distribution. Delete the app bundle to uninstall.

The default `.xanylabelingrc` and `xanylabeling_data` remain in the user's home
directory. These packages do not delete configuration, model caches, or annotation
files. SHA-256 sidecars detect damaged downloads; they are not publisher signatures.
Windows upgrades currently overlay packaged files and do not provide automatic
rollback when the user manually reruns Setup. Automatic updates instead replace
the application directory, preserve files outside the package's managed paths
(including the uninstaller), and restore the previous directory if startup fails.
The Windows uninstall entry is updated after a successful automatic update.

## Enable signed releases

Generate an Ed25519 key once, outside the repository, and keep a secure backup:

```bash
source .venv-cpu/bin/activate
mkdir -p ~/.config/xanylabeling
python scripts/generate_update_key.py --output ~/.config/xanylabeling/update-signing.key
```

The command prints the public key, never the private key. Configure these GitHub
repository settings before enabling installation packages in the release workflow:

| Setting | Type | Value |
| --- | --- | --- |
| `UPDATE_SIGNING_KEY` | Actions secret | Contents of the private key file |
| `UPDATE_PUBLIC_KEY` | Actions variable | Public key printed by the generator |
| `UPDATE_FEED_URL` | Actions variable, optional | HTTPS address of the signed manifest |
| `AUTO_UPDATE_ENABLED` | Actions variable | `true` after native acceptance checks |

The default feed is
`https://github.com/CVHub520/X-AnyLabeling/releases/latest/download/update-manifest.json`.
Forks should set their own feed. A CDN can serve the same manifest and update ZIPs;
set `X_ANYLABELING_RELEASE_BASE_URL` during package builds to change archive URLs.
Both feed and archive addresses must use HTTPS.

When enabled, the release workflow builds all ten configured platform/variant
combinations, adds installation and update packages, verifies their hashes and
signing-key identities, and signs one manifest. The release is assembled as a
draft and published only after all assets are uploaded. Website synchronization
is called after publication when `WEBSITE_SYNC_ENABLED` is configured. The website
shows direct installation downloads only for releases containing the update manifest.

Use the same signing key for subsequent releases. Key rotation is not implemented
in this protocol. Users of an older single-file executable must install an update-enabled
version once; automatic updates then take over. Initial package signing, Windows
SmartScreen reputation, and macOS notarization are separate from the update manifest's
signature. The macOS build restores a valid ad-hoc bundle signature after adding the
updater; `X_ANYLABELING_MAC_SIGN_IDENTITY` can select an installed signing identity.
Developer ID signing and notarization must be validated before public macOS distribution.

## Update behavior and recovery

The version manifest is verified with the installed public key before displaying
release notes or selecting a download. Packages must match the operating system,
architecture, and CPU/CUDA variant. Downloads are streamed, cancellable, and checked
against the signed size and SHA-256. The updater rechecks the signature and archive
integrity before installing and rejects unsafe extraction paths or symlinks.

After download, the app uses its normal close checks for unsaved annotations and
training tasks. Cancelling those checks postpones installation. The updater waits
for the process and other instances using that installation to exit. It then stages
the new package on the same filesystem and switches the installation. Linux user
installs switch `current` atomically and retain older versions; Windows, macOS, and
Linux portable builds retain a backup until the new GUI confirms startup. The new
process inherits launch arguments and a fresh PyInstaller environment.

Failed startup terminates the new process, restores the old installation, and
relaunches it. The old client shows the recorded failure. Non-writable installation
directories require manual installation into a writable location; automatic elevation
is not implemented. A missing compatible package causes no automatic installation.

A process crash or power loss during a directory replacement is distinct from a
normal installation failure. A journal is saved under the user's cache directory
(`X-AnyLabeling/updates/<installation-id>/result.json`). Close all app instances and
recover an unfinished transaction using the cached updater or, from this checkout:

```bash
source .venv-cpu/bin/activate
python anylabeling/services/updates/installer.py --recover /path/to/result.json
```

Recovery preserves the interrupted new directory rather than deleting it. This
manual recovery path must be exercised on Windows and macOS before release; recovery
from power loss is not guaranteed to be automatic. Successful downloads are deleted
after installation and the cached helper is removed on a subsequent startup.

## Update validation

Run the failure and UI checks with:

```bash
source .venv-cpu/bin/activate
QT_QPA_PLATFORM=offscreen pytest tests/test_updates
```

## Native acceptance checks

For each platform, check a first installation, launching through its shortcut or
app bundle, loading a model, and saving annotations. Install a second version
after closing the app and verify configuration and model caches are retained.
Check paths with spaces and non-ASCII characters, uninstallation, and coexistence
of CPU and CUDA variants on Windows and Linux. Test failed or interrupted upgrades
before enabling automatic updates. Windows and macOS package compilation and GUI
behavior require their native runners; Linux-side static checks cannot certify them.
