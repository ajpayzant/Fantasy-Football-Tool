' Fantasy Mock Draft - silent launcher (no flashing console window).
' Double-click this, or create a shortcut to it and pin THAT to the taskbar.
' It runs launch_mock_draft.bat hidden and opens the app on port 8502.
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = scriptDir
' 0 = hidden window, True = don't wait (Streamlit keeps running).
sh.Run """" & scriptDir & "\launch_mock_draft.bat""", 0, False
