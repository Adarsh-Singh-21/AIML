param()
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment.' }
}
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $taskPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize pip.' }
& $taskPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install numerical project dependencies.' }
& $taskPython -m lung_tabular doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
