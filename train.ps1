param([string]$RunName = 'numerical_repeat')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 first.' }
if (-not (Test-Path -LiteralPath 'data\numerical\akashnath29\lung_cancer.csv')) {
    & $taskPython -m lung_tabular download-data
    if ($LASTEXITCODE -ne 0) { throw 'Numerical dataset download failed.' }
}
& $taskPython -m lung_tabular prepare-data
if ($LASTEXITCODE -ne 0) { throw 'Numerical dataset preparation failed.' }
& $taskPython -m lung_tabular train --run-name $RunName
if ($LASTEXITCODE -ne 0) { throw 'Numerical model training failed.' }
Write-Output 'The trained numerical model is now active. Later runs preserve the viewed test partition and report CV only.'
