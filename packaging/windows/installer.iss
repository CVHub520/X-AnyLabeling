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
AppId=X-AnyLabeling
AppName=X-AnyLabeling
AppVersion={#AppVersion}
AppPublisher=CVHub
AppPublisherURL=https://xanylabeling.com
DefaultDirName={localappdata}\Programs\X-AnyLabeling
DefaultGroupName=X-AnyLabeling
UsePreviousGroup=no
DisableProgramGroupPage=yes
DisableDirPage=no
DisableWelcomePage=yes
DisableReadyPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=X-AnyLabeling-v{#AppVersion}-Windows-{#Variant}-Setup
Uninstallable=not IsUpdateStage
CreateUninstallRegKey=not IsUpdateStage
UninstallDisplayIcon={app}\X-AnyLabeling.exe
UninstallDisplayName=X-AnyLabeling
Compression=lzma2/fast
SolidCompression=yes
SetupIconFile=..\..\anylabeling\resources\images\icon.ico
WizardStyle=modern light windows11 excludelightbuttons hidebevels
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\..\anylabeling\resources\images\icon.png"; DestDir: "{tmp}"; Flags: dontcopy noencryption
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs solidbreak

[Messages]
SetupWindowTitle=X-AnyLabeling
UninstallAppFullTitle=X-AnyLabeling

[Icons]
Name: "{group}\X-AnyLabeling"; Filename: "{app}\X-AnyLabeling.exe"; WorkingDir: "{%USERPROFILE}"; Flags: uninsneveruninstall; Check: not IsUpdateStage
Name: "{userdesktop}\X-AnyLabeling"; Filename: "{app}\X-AnyLabeling.exe"; WorkingDir: "{%USERPROFILE}"; Check: ShouldCreateDesktopShortcut and not IsUpdateStage; Flags: uninsneveruninstall

[Run]
Filename: "{app}\X-AnyLabeling.exe"; Description: "Launch X-AnyLabeling"; WorkingDir: "{%USERPROFILE}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[InstallDelete]
Type: files; Name: "{userdesktop}\X-AnyLabeling ({#Variant}).lnk"; Check: not IsUpdateStage
Type: files; Name: "{userprograms}\X-AnyLabeling ({#Variant})\X-AnyLabeling ({#Variant}).lnk"; Check: not IsUpdateStage
Type: dirifempty; Name: "{userprograms}\X-AnyLabeling ({#Variant})"; Check: not IsUpdateStage

[Code]
function GetWizardSystemMenu(Window: HWND; Revert: Integer): HWND;
  external 'GetSystemMenu@user32.dll stdcall';
function EnableWizardMenuItem(Menu: HWND; Item, Flags: Integer): Integer;
  external 'EnableMenuItem@user32.dll stdcall';
function RedrawWizardMenu(Window: HWND): Integer;
  external 'DrawMenuBar@user32.dll stdcall';

var
  DesktopShortcut, LaunchApp: TNewCheckBox;
  InstallLocationLabel, ProgressPercent: TNewStaticText;
  ProgressTrack, ProgressFill: TPanel;
  BackupDirectory: String;
  BackupPaths: TStringList;
  InstallationCommitted: Boolean;

function IsUpdateStage: Boolean;
begin
  Result := ExpandConstant('{param:UPDATESTAGE|0}') = '1';
end;

procedure WizardCloseQuery(Sender: TObject; var CanClose: Boolean);
begin
  if WizardForm.CurPageID = wpFinished then
  begin
    CanClose := False;
    LaunchApp.Checked := False;
    DesktopShortcut.Checked := False;
    WizardForm.RunList.Checked[0] := False;
    WizardForm.NextButton.OnClick(WizardForm.NextButton);
  end;
end;

procedure RemoveVersionDirectories(const KeepVersion: String);
var
  Entry: TFindRec;
  Directory: String;
  Metadata: AnsiString;
begin
  if FindFirst(ExpandConstant('{app}\*'), Entry) then
  begin
    try
      repeat
        if (Entry.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0) and
           (Entry.Attributes and FILE_ATTRIBUTE_REPARSE_POINT = 0) and
           (Entry.Name <> '.') and (Entry.Name <> '..') and
           (Entry.Name <> KeepVersion) then
        begin
          Directory := ExpandConstant('{app}\') + Entry.Name;
          if LoadStringFromFile(Directory + '\build-info.json', Metadata) and
             (Pos('"schema": 1,', Metadata) > 0) and
             (Pos('"version": "' + Entry.Name + '"', Metadata) > 0) and
             (Pos('"platform": "Windows"', Metadata) > 0) then
            if not DelTree(Directory, True, True, True) then
            begin
              if KeepVersion = '' then
                RaiseException('Unable to remove application version: ' + Directory);
              Log('Unable to remove application version: ' + Directory);
            end;
        end;
      until not FindNext(Entry);
    finally
      FindClose(Entry);
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if DirExists(ExpandConstant('{app}\.setup-backup')) or
     FileExists(ExpandConstant('{app}\.setup-backup')) then
    Result := 'A previous installation backup exists in ' +
      ExpandConstant('{app}\.setup-backup') +
      '. Restore or preserve this backup before removing it and retrying Setup.';
end;

procedure BackupInstalledPath(const Name: String);
var
  Source: String;
begin
  Source := ExpandConstant('{app}\') + Name;
  if not FileExists(Source) and not DirExists(Source) then
    Exit;
  if not RenameFile(Source, BackupDirectory + '\' + Name) then
    RaiseException('Unable to back up ' + Source +
      '. Close X-AnyLabeling and try again.');
  BackupPaths.Add(Name);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
  begin
    BackupDirectory := ExpandConstant('{app}\.setup-backup');
    if not ForceDirectories(ExpandConstant('{app}')) or
       not CreateDir(BackupDirectory) then
      RaiseException('Unable to create installation backup: ' + BackupDirectory);
    BackupPaths := TStringList.Create;
    BackupInstalledPath('X-AnyLabeling.exe');
    BackupInstalledPath('{#AppVersion}');
    BackupInstalledPath('_internal');
    BackupInstalledPath('build-info.json');
    BackupInstalledPath('xanylabeling-updater.exe');
  end
  else if CurStep = ssPostInstall then
  begin
    InstallationCommitted := True;
    RemoveVersionDirectories('{#AppVersion}');
    if not DelTree(BackupDirectory, True, True, True) then
      Log('Unable to remove installation backup: ' + BackupDirectory);
  end;
end;

procedure DeinitializeSetup;
var
  Index: Integer;
  Source, Destination: String;
  Restored: Boolean;
begin
  if BackupPaths = nil then
    Exit;
  if not InstallationCommitted then
  begin
    Restored := True;
    for Index := BackupPaths.Count - 1 downto 0 do
    begin
      Source := BackupDirectory + '\' + BackupPaths[Index];
      Destination := ExpandConstant('{app}\') + BackupPaths[Index];
      if DirExists(Destination) then
        DelTree(Destination, True, True, True)
      else if FileExists(Destination) then
        DeleteFile(Destination);
      if not RenameFile(Source, Destination) then
      begin
        Restored := False;
        Log('Unable to restore installation file: ' + Destination);
      end;
    end;
    if Restored then
      RemoveDir(BackupDirectory)
    else
      SuppressibleMsgBox('Setup could not restore all previous application files. ' +
        'Your backup is preserved in ' + BackupDirectory +
        '. Close X-AnyLabeling and restore these files before starting it.',
        mbError, MB_OK, IDOK);
  end;
  BackupPaths.Free;
end;

procedure RemoveApplicationShortcut(const Filename: String);
var
  Shell, Shortcut: Variant;
begin
  if FileExists(Filename) then
  begin
    try
      Shell := CreateOleObject('WScript.Shell');
      Shortcut := Shell.CreateShortcut(Filename);
      if CompareText(Shortcut.TargetPath, ExpandConstant('{app}\X-AnyLabeling.exe')) = 0 then
        DeleteFile(Filename);
    except
      Log('Unable to inspect shortcut: ' + Filename);
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    RemoveApplicationShortcut(ExpandConstant('{group}\X-AnyLabeling.lnk'));
    RemoveApplicationShortcut(ExpandConstant('{userdesktop}\X-AnyLabeling.lnk'));
    RemoveDir(ExpandConstant('{group}'));
    RemoveVersionDirectories('');
  end;
end;

procedure BrowseInstallDirectory(Sender: TObject);
var
  Directory: String;
begin
  Directory := WizardForm.DirEdit.Text;
  if BrowseForFolder('Choose an installation folder', Directory, True) then
    WizardForm.DirEdit.Text := Directory;
end;

procedure FocusInstallDirectory(Sender: TObject);
begin
  WizardForm.ActiveControl := WizardForm.DirEdit;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := PageID = wpSelectTasks;
end;

function ShouldCreateDesktopShortcut: Boolean;
begin
  Result := WizardSilent and WizardIsTaskSelected('desktopicon');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  ShortcutPath: String;
begin
  Result := True;
  if (CurPageID = wpFinished) and not WizardSilent then
  begin
    if DesktopShortcut.Checked then
    begin
      ShortcutPath := ExpandConstant('{userdesktop}\X-AnyLabeling.lnk');
      try
        if not ForceDirectories(ExtractFileDir(ShortcutPath)) then
          RaiseException('Unable to access the desktop folder.');
        CreateShellLink(ShortcutPath, 'X-AnyLabeling',
          ExpandConstant('{app}\X-AnyLabeling.exe'), '',
          ExpandConstant('{%USERPROFILE}'), '', 0, SW_SHOWNORMAL);
      except
        MsgBox('Unable to create the desktop shortcut. Please try again or clear this option.', mbError, MB_OK);
        Result := False;
        Exit;
      end;
    end;
    WizardForm.RunList.Checked[0] := LaunchApp.Checked;
  end;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  WizardForm.Caption := 'X-AnyLabeling';
  WizardForm.BackButton.Visible := False;
  WizardForm.PageNameLabel.Caption := 'X-AnyLabeling';
  WizardForm.PageDescriptionLabel.Visible := False;
  if CurPageID = wpSelectDir then
  begin
    WizardForm.NextButton.Caption := 'Install';
    if not WizardSilent then
      WizardForm.ActiveControl := WizardForm.NextButton;
  end;
  if CurPageID = wpInstalling then
  begin
    WizardForm.NextButton.Visible := False;
    WizardForm.CancelButton.Left := WizardForm.NextButton.Left;
  end
  else
    WizardForm.CancelButton.Left := WizardForm.NextButton.Left -
      WizardForm.CancelButton.Width - ScaleX(12);
  if CurPageID = wpFinished then
  begin
    WizardForm.FinishedHeadingLabel.Caption := 'Installation complete';
    WizardForm.FinishedLabel.Caption := 'X-AnyLabeling is ready to use.';
    WizardForm.FinishedLabel.SetBounds(ScaleX(32), ScaleY(190), WizardForm.ClientWidth - ScaleX(64), ScaleY(26));
    WizardForm.RunList.Visible := False;
    WizardForm.RunList.Checked[0] := False;
    WizardForm.NextButton.Caption := 'Finish';
    WizardForm.NextButton.Enabled := True;
    EnableWizardMenuItem(GetWizardSystemMenu(WizardForm.Handle, 0), $F060, 0);
    RedrawWizardMenu(WizardForm.Handle);
  end;
end;

procedure CurInstallProgressChanged(CurProgress, MaxProgress: Integer);
var
  Percent: Integer;
begin
  if MaxProgress > 0 then
  begin
    Percent := Round(100.0 * CurProgress / MaxProgress);
    ProgressPercent.Caption := IntToStr(Percent) + '%';
    ProgressFill.Width := Round(1.0 * ProgressTrack.Width * CurProgress / MaxProgress);
  end;
end;

procedure InitializeWizard;
var
  BrandImage, FinishedImage: TBitmapImage;
  FinishedTitle: TNewStaticText;
  DirectoryBorder, DirectoryBackground: TPanel;
  FontMetrics: TBitmap;
  Width, Height, ContentWidth, ButtonWidth, ControlHeight, TextHeight: Integer;
begin
  Width := ScaleX(640);
  Height := ScaleY(348);
  ContentWidth := Width - ScaleX(64);
  ButtonWidth := ScaleX(104);
  ControlHeight := ScaleY(34);
  WizardForm.ClientWidth := Width;
  WizardForm.ClientHeight := Height;
  WizardForm.Font.Name := 'Segoe UI';
  WizardForm.Font.Size := 10;
  WizardForm.OnCloseQuery := @WizardCloseQuery;
  WizardForm.OuterNotebook.SetBounds(0, 0, Width, Height - ScaleY(70));
  WizardForm.MainPanel.SetBounds(0, 0, Width, ScaleY(152));
  WizardForm.InnerNotebook.SetBounds(ScaleX(32), ScaleY(152), ContentWidth, Height - ScaleY(222));
  WizardForm.WizardSmallBitmapImage.Visible := False;
  WizardForm.WizardBitmapImage.Visible := False;
  WizardForm.WizardBitmapImage2.Visible := False;
  WizardForm.SelectDirBitmapImage.Visible := False;
  WizardForm.SelectDirLabel.Visible := False;
  WizardForm.SelectDirBrowseLabel.Visible := False;
  WizardForm.PageNameLabel.AutoSize := False;
  WizardForm.PageNameLabel.Font.Size := 18;
  WizardForm.PageNameLabel.Font.Style := [fsBold];
  WizardForm.PageNameLabel.Alignment := taCenter;
  WizardForm.PageNameLabel.SetBounds(ScaleX(32), ScaleY(92), ContentWidth, ScaleY(40));
  WizardForm.PageDescriptionLabel.Visible := False;
  ExtractTemporaryFile('icon.png');
  BrandImage := TBitmapImage.Create(WizardForm);
  BrandImage.Parent := WizardForm.MainPanel;
  BrandImage.SetBounds((Width - ScaleX(52)) div 2, ScaleY(24), ScaleX(52), ScaleY(52));
  BrandImage.PngImage.LoadFromFile(ExpandConstant('{tmp}\icon.png'));
  BrandImage.Stretch := True;
  InstallLocationLabel := TNewStaticText.Create(WizardForm);
  InstallLocationLabel.Parent := WizardForm.SelectDirPage;
  InstallLocationLabel.Caption := 'Install location';
  InstallLocationLabel.SetBounds(0, 0, ContentWidth, ScaleY(24));
  DirectoryBorder := TPanel.Create(WizardForm);
  DirectoryBorder.Parent := WizardForm.SelectDirPage;
  DirectoryBorder.TabOrder := WizardForm.DirEdit.TabOrder;
  DirectoryBorder.SetBounds(0, ScaleY(33), ContentWidth - ButtonWidth - ScaleX(12), ControlHeight - ScaleY(2));
  DirectoryBorder.BevelOuter := bvNone;
  DirectoryBorder.StyleElements := [];
  DirectoryBorder.Color := $00D1D5DB;
  DirectoryBorder.OnClick := @FocusInstallDirectory;
  DirectoryBackground := TPanel.Create(WizardForm);
  DirectoryBackground.Parent := DirectoryBorder;
  DirectoryBackground.SetBounds(ScaleX(1), ScaleY(1), DirectoryBorder.Width - ScaleX(2), DirectoryBorder.Height - ScaleY(2));
  DirectoryBackground.BevelOuter := bvNone;
  DirectoryBackground.StyleElements := [];
  DirectoryBackground.Color := clWhite;
  DirectoryBackground.OnClick := @FocusInstallDirectory;
  WizardForm.DirEdit.Parent := DirectoryBackground;
  WizardForm.DirEdit.StyleName := 'Windows';
  WizardForm.DirEdit.StyleElements := [];
  WizardForm.DirEdit.BorderStyle := bsNone;
  WizardForm.DirEdit.Font.Assign(WizardForm.Font);
  WizardForm.DirEdit.Color := clWhite;
  FontMetrics := TBitmap.Create;
  FontMetrics.Canvas.Font.Assign(WizardForm.DirEdit.Font);
  TextHeight := FontMetrics.Canvas.TextHeight('Ag');
  FontMetrics.Free;
  WizardForm.DirEdit.AutoSize := False;
  WizardForm.DirEdit.SetBounds(ScaleX(8),
    (DirectoryBackground.Height - TextHeight) div 2,
    DirectoryBackground.Width - ScaleX(16), TextHeight);
  WizardForm.DirBrowseButton.SetBounds(ContentWidth - ButtonWidth, ScaleY(32), ButtonWidth, ControlHeight);
  WizardForm.DirBrowseButton.Caption := 'Browse...';
  WizardForm.DirBrowseButton.OnClick := @BrowseInstallDirectory;
  WizardForm.DiskSpaceLabel.SetBounds(0, ScaleY(78), ContentWidth, ScaleY(24));
  WizardForm.DiskSpaceLabel.Font.Size := 9;
  WizardForm.DiskSpaceLabel.StyleElements := [seClient, seBorder];
  WizardForm.DiskSpaceLabel.Font.Color := $0063554B;
  WizardForm.StatusLabel.SetBounds(0, ScaleY(42), ContentWidth - ScaleX(64), ScaleY(28));
  WizardForm.StatusLabel.Font.Style := [fsBold];
  WizardForm.FilenameLabel.SetBounds(0, ScaleY(78), ContentWidth, ScaleY(28));
  WizardForm.FilenameLabel.Font.Size := 9;
  WizardForm.FilenameLabel.StyleElements := [seClient, seBorder];
  WizardForm.FilenameLabel.Font.Color := $0063554B;
  WizardForm.ProgressGauge.Visible := False;
  ProgressTrack := TPanel.Create(WizardForm);
  ProgressTrack.Parent := WizardForm.InstallingPage;
  ProgressTrack.SetBounds(0, ScaleY(16), ContentWidth, ScaleY(6));
  ProgressTrack.BevelOuter := bvNone;
  ProgressTrack.StyleElements := [];
  ProgressTrack.Color := $00E5E7EB;
  ProgressFill := TPanel.Create(WizardForm);
  ProgressFill.Parent := ProgressTrack;
  ProgressFill.SetBounds(0, 0, 0, ProgressTrack.Height);
  ProgressFill.BevelOuter := bvNone;
  ProgressFill.StyleElements := [];
  ProgressFill.Color := $00466AFF;
  ProgressPercent := TNewStaticText.Create(WizardForm);
  ProgressPercent.Parent := WizardForm.InstallingPage;
  ProgressPercent.Alignment := taRightJustify;
  ProgressPercent.Font.Style := [fsBold];
  ProgressPercent.SetBounds(ContentWidth - ScaleX(52), ScaleY(42), ScaleX(52), ScaleY(28));
  ProgressPercent.Caption := '0%';
  FinishedImage := TBitmapImage.Create(WizardForm);
  FinishedImage.Parent := WizardForm.FinishedPage;
  FinishedImage.SetBounds(BrandImage.Left, BrandImage.Top, BrandImage.Width, BrandImage.Height);
  FinishedImage.PngImage.Assign(BrandImage.PngImage);
  FinishedImage.Stretch := True;
  FinishedTitle := TNewStaticText.Create(WizardForm);
  FinishedTitle.Parent := WizardForm.FinishedPage;
  FinishedTitle.AutoSize := False;
  FinishedTitle.Caption := 'X-AnyLabeling';
  FinishedTitle.Font.Size := 18;
  FinishedTitle.Font.Style := [fsBold];
  FinishedTitle.Alignment := taCenter;
  FinishedTitle.SetBounds(ScaleX(32), ScaleY(92), ContentWidth, ScaleY(40));
  WizardForm.FinishedHeadingLabel.Font.Size := 12;
  WizardForm.FinishedHeadingLabel.Font.Style := [fsBold];
  WizardForm.FinishedHeadingLabel.SetBounds(ScaleX(32), ScaleY(152), ContentWidth, ScaleY(28));
  WizardForm.FinishedLabel.Font.Size := 10;
  WizardForm.FinishedLabel.StyleElements := [seClient, seBorder];
  WizardForm.FinishedLabel.Font.Color := $0063554B;
  WizardForm.FinishedLabel.SetBounds(ScaleX(32), ScaleY(190), ContentWidth, ScaleY(26));
  LaunchApp := TNewCheckBox.Create(WizardForm);
  LaunchApp.Parent := WizardForm.FinishedPage;
  LaunchApp.SetBounds(ScaleX(32), ScaleY(218), ContentWidth, ScaleY(24));
  LaunchApp.Caption := 'Run X-AnyLabeling';
  LaunchApp.Checked := True;
  DesktopShortcut := TNewCheckBox.Create(WizardForm);
  DesktopShortcut.Parent := WizardForm.FinishedPage;
  DesktopShortcut.SetBounds(ScaleX(32), ScaleY(248), ContentWidth, ScaleY(24));
  DesktopShortcut.Caption := 'Create a desktop shortcut';
  DesktopShortcut.Checked := True;
  WizardForm.NextButton.SetBounds(Width - ScaleX(32) - ButtonWidth, Height - ScaleY(54), ButtonWidth, ControlHeight);
  WizardForm.CancelButton.SetBounds(WizardForm.NextButton.Left - ScaleX(12) - ButtonWidth, WizardForm.NextButton.Top, ButtonWidth, ControlHeight);
end;
