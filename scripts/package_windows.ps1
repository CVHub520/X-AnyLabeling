param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^(CPU|CUDA11|CUDA12|CUDA13)$')]
    [string]$Variant
)

$ErrorActionPreference = 'Stop'
$rootDir = Split-Path $PSScriptRoot -Parent
$distDir = Join-Path $rootDir 'dist'
$sourceDir = Join-Path $distDir 'X-AnyLabeling'
$appInfo = Get-Content (Join-Path $rootDir 'anylabeling/app_info.py') -Raw
$versionMatch = [regex]::Match($appInfo, '(?m)^__version__\s*=\s*"([^"]+)"')
if (-not $versionMatch.Success) { throw 'Unable to read application version.' }
$appVersion = $versionMatch.Groups[1].Value
if (-not (Test-Path (Join-Path $sourceDir 'X-AnyLabeling.exe'))) {
    throw 'Build the Windows onedir bundle before packaging.'
}

$iscc = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue
$compiler = if ($iscc) { $iscc.Source } else {
    Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6/ISCC.exe'
}
if (-not (Test-Path $compiler)) { throw 'Install Inno Setup 6 before packaging.' }

& $compiler "/DAppVersion=$appVersion" "/DVariant=$Variant" "/DSourceDir=$sourceDir" "/DOutputDir=$distDir" (Join-Path $rootDir 'packaging/windows/installer.iss')
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }

$baseName = "X-AnyLabeling-v$appVersion-Windows-$Variant"
$artifact = Join-Path $distDir "$baseName-Setup.exe"
$hash = (Get-FileHash $artifact -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $(Split-Path $artifact -Leaf)" | Set-Content "$artifact.sha256" -Encoding ASCII
