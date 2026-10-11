#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef Variant
  #error Variant is required
#endif
#ifndef SourceDir
  #error SourceDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif

[Setup]
AppId=X-AnyLabeling-{#Variant}
AppName=X-AnyLabeling ({#Variant})
AppVersion={#AppVersion}
AppPublisher=CVHub
AppPublisherURL=https://xanylabeling.com
DefaultDirName={localappdata}\Programs\X-AnyLabeling-{#Variant}
DefaultGroupName=X-AnyLabeling ({#Variant})
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=X-AnyLabeling-v{#AppVersion}-Windows-{#Variant}-Setup
UninstallDisplayIcon={app}\X-AnyLabeling.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\X-AnyLabeling ({#Variant})"; Filename: "{app}\X-AnyLabeling.exe"; WorkingDir: "{%USERPROFILE}"
Name: "{userdesktop}\X-AnyLabeling ({#Variant})"; Filename: "{app}\X-AnyLabeling.exe"; WorkingDir: "{%USERPROFILE}"; Tasks: desktopicon

[Run]
Filename: "{app}\X-AnyLabeling.exe"; Description: "Launch X-AnyLabeling"; WorkingDir: "{%USERPROFILE}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Code]
procedure BrowseInstallDirectory(Sender: TObject);
var
  Directory: String;
begin
  Directory := WizardForm.DirEdit.Text;
  if BrowseForFolder('Choose an installation folder', Directory, True) then
    WizardForm.DirEdit.Text := Directory;
end;

procedure InitializeWizard;
begin
  WizardForm.DirBrowseButton.OnClick := @BrowseInstallDirectory;
end;
