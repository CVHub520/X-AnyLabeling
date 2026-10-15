# Packaging

Build from the repository root on the target operating system with the matching Python dependencies installed. Use Bash, or Git Bash on Windows.

## Build

```bash
bash scripts/build_executable.sh win-cpu-installer
bash scripts/build_executable.sh linux-cpu
bash scripts/build_executable.sh macos-installer
```

Windows requires PowerShell and Inno Setup 6.7.3 or later. For Windows and Linux GPU builds, use `win-gpu-installer` or `linux-gpu` and set `X_ANYLABELING_VARIANT` to `CUDA11`, `CUDA12`, or `CUDA13`. Install the corresponding dependency extra: `gpu-cu11`, `gpu`, or `gpu-cu13`.

Build outputs go to `dist/`. PyInstaller specifications are in `packaging/pyinstaller/specs`; platform installers are in `packaging/windows` and `packaging/linux`.

## Automatic updates

Generate a signing key once and keep it outside the repository:

```bash
source .venv-cpu/bin/activate
mkdir -p ~/.config/xanylabeling
python scripts/generate_update_key.py --output ~/.config/xanylabeling/update-signing.key
```

Configure these GitHub Actions settings:

| Name | Type | Value |
| --- | --- | --- |
| `UPDATE_SIGNING_KEY` | Secret | Private key file contents |
| `UPDATE_PUBLIC_KEY` | Variable | Public key printed by the generator |
| `UPDATE_FEED_URL` | Variable | Optional HTTPS manifest URL for forks or custom hosting |
| `AUTO_UPDATE_ENABLED` | Variable | Set to `true` after testing installation and updates on each target platform |

Keep a secure backup of the signing key and reuse it for future releases; key rotation is not supported. Builds without `UPDATE_PUBLIC_KEY` cannot install updates automatically. Linux single-file builds embed their update metadata and helper. Source checkouts do not install updates automatically.

Updates reuse the published Windows `Setup.exe`, macOS DMG, and Linux executable. No separate `Update.zip` is generated or uploaded. The signed manifest records the format, size, and SHA-256 digest of each release artifact.

The Windows installer supports `/UPDATESTAGE=1` for extracting into an isolated staging directory without changing shortcuts or uninstall registration. The updater verifies the staged application, replaces the installation, and restores the previous version if startup fails. Linux replaces the existing executable in place, preserving its filename; macOS mounts the DMG read-only, copies the application bundle to staging with `ditto`, and detaches the image before replacing the installed application.

Update signatures do not replace Windows code signing or macOS signing and notarization.

## Test

```bash
source .venv-cpu/bin/activate
QT_QPA_PLATFORM=offscreen pytest tests/test_updates
```

On Windows, set `INNO_SETUP_COMPILER` to the full path of `ISCC.exe` to run the Setup tests. Before release, test installation, upgrades, failed-update recovery, and uninstallation on each target platform.
