param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 first to install the project environment.' }
Write-Output "Open http://localhost:$Port to use the classifier. Press Ctrl+C to stop it."
& $taskPython -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
