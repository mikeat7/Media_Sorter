; Media Sorter — Inno Setup Script
; Compile with Inno Setup 6+ from https://jrsoftware.org/isdl.php
; Output: dist\MediaSorter_Setup.exe  (~5 MB, downloads AI models on first run)

#define AppName      "Media Sorter"
#define AppVersion   "3.0"
#define AppPublisher "mikeat7"
#define AppURL       "https://github.com/mikeat7/Media_Sorter"

[Setup]
AppId={{A3F72C8D-4B1E-4F9A-8C3D-2E6F7A0B3C4E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=dist
OutputBaseFilename=MediaSorter_Setup
SetupIconFile=
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Run without admin — installs to user profile if admin not available
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Core application
Source: "app.py";                   DestDir: "{app}";           Flags: ignoreversion
Source: "requirements.txt";         DestDir: "{app}";           Flags: ignoreversion
Source: "Launch Media Sorter.bat";  DestDir: "{app}";           Flags: ignoreversion
Source: "README.md";                DestDir: "{app}";           Flags: ignoreversion isreadme

; Web UI templates
Source: "templates\*";             DestDir: "{app}\templates"; Flags: ignoreversion recursesubdirs createallsubdirs

; NOTE: md_v5a.0.0.pt (~290 MB) and CLIP model (~350 MB) are NOT bundled.
;       They download automatically on first sort run — one time only.

[Icons]
; Start menu
Name: "{group}\{#AppName}";              Filename: "{app}\Launch Media Sorter.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#AppName}";    Filename: "{uninstallexe}"

; Desktop (optional)
Name: "{autodesktop}\{#AppName}";        Filename: "{app}\Launch Media Sorter.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Install Python packages
Filename: "python"; Parameters: "-m pip install --upgrade pip --quiet"; \
  StatusMsg: "Preparing package manager..."; Flags: runhidden waituntilterminated

Filename: "python"; Parameters: "-m pip install -r ""{app}\requirements.txt"" --quiet"; \
  StatusMsg: "Installing required packages (5-15 minutes depending on speed)..."; \
  Flags: runhidden waituntilterminated

Filename: "python"; Parameters: "-m pip install facenet-pytorch --no-deps --quiet"; \
  StatusMsg: "Installing face recognition support..."; \
  Flags: runhidden waituntilterminated

Filename: "python"; Parameters: "-m pip install pillow-heif --quiet"; \
  StatusMsg: "Installing iPhone HEIC photo support..."; \
  Flags: runhidden waituntilterminated

; Offer to launch after install
Filename: "{app}\Launch Media Sorter.bat"; \
  Description: "&Launch Media Sorter now"; \
  WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent shellexec

[UninstallDelete]
; Remove generated files on uninstall (user data in C:\MediaSorted is never touched)
Type: filesandordirs; Name: "{app}\__pycache__"
Type: files;          Name: "{app}\custom_categories.json"
Type: files;          Name: "{app}\category_thresholds.json"

[Code]
{ ── Python detection ── }
function PythonFound(): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('python', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode)
            and (ResultCode = 0);
end;

function InitializeSetup(): Boolean;
var
  MsgResult: Integer;
begin
  Result := True;
  if not PythonFound() then
  begin
    MsgResult := MsgBox(
      'Python 3.10 or newer is required but was not found.' + #13#10 + #13#10 +
      'Please download and install Python from:' + #13#10 +
      '  https://www.python.org/downloads/' + #13#10 + #13#10 +
      'IMPORTANT: During install, check the box that says:' + #13#10 +
      '  "Add Python to PATH"' + #13#10 + #13#10 +
      'Then run this installer again.' + #13#10 + #13#10 +
      'Click OK to open the Python download page now.',
      mbError, MB_OKCANCEL);
    if MsgResult = IDOK then
    begin
      ShellExec('open', 'https://www.python.org/downloads/', '', '', SW_SHOW,
                ewNoWait, MsgResult);
    end;
    Result := False;
  end;
end;

{ Friendly finish page message }
function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine +
            'AI models (~640 MB total) will download automatically' + NewLine +
            'the first time you run a sort — one time only.' + NewLine + NewLine +
            MemoTasksInfo;
end;
