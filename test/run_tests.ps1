# Test script location: c:\Users\treyr\ORAMA\test\run_tests.ps1

# Navigate to the project root from the script's location (test/ -> root)
Push-Location ".."

Write-Host "Starting ORAMA module tests from root: $(Get-Location)"
Write-Host "----------------------------------------------------"

$ErrorActionPreference = "Continue" # Continue even if a command fails, but $? will be false

# Ensure Python execution policy allows scripts if needed, or that python is in PATH
# For example, in some environments: Set-ExecutionPolicy RemoteSigned -Scope Process -Force

$python_modules = @(
 "action_system.py",
 "core_engine.py",
 "interface.py",
 "main.py",
 "memory_engine.py",
 "orchestrator.py",
 "system_manager.py",
 "system_utils.py",
 "orama/debug_manager.py",
 "orama/resource_manager.py",
 "orama/__init__.py"
)

$all_tests_passed = $true

# 1. Python version check
Write-Section "Python Version Check"
python --version
if ($LASTEXITCODE -ne 0) {
 Write-Host "[FATAL] Python is not installed or not in PATH." -ForegroundColor Red
 exit 1
}

# 2. Dependency check (requirements.txt)
Write-Section "Dependency Check (requirements.txt)"
if (Test-Path "requirements.txt") {
 $missing = @()
 $reqs = Get-Content requirements.txt | Where-Object { $_ -notmatch "^#" -and $_.Trim() -ne "" }
 foreach ($req in $reqs) {
  $pkg = $req -replace ".*([a-zA-Z0-9\-_]+).*", '$1'
  if ($pkg -eq "dataclasses" -or $pkg -eq "asyncio") { continue }
  python -c "import $pkg" 2>$null
  if ($LASTEXITCODE -ne 0) { $missing += $pkg }
 }
 if ($missing.Count -eq 0) {
  Write-Result "All required packages importable" $true
 }
 else {
  Write-Result "Missing packages: $($missing -join ", ")" $false
 }
 Write-Host "Running 'pip check'..."
 pip check
 Write-Host "Running 'pip list --outdated'..."
 pip list --outdated
}
else {
 Write-Result "requirements.txt not found" $false
}

# 3. Directory and file presence check
Write-Section "File/Directory Presence Check"
$expected = @(
 "action_system.py", "core_engine.py", "interface.py", "main.py", "memory_engine.py", "orchestrator.py", "system_manager.py", "system_utils.py",
 "orama/debug_manager.py", "orama/resource_manager.py", "orama/__init__.py",
 "data/vector_store/.gitkeep", "data/knowledge_graph/.gitkeep"
)
$all_present = $true
foreach ($f in $expected) {
 if (Test-Path $f) {
  Write-Result "$f exists" $true
 }
 else {
  Write-Result "$f missing" $false
  $all_present = $false
 }
}

# 4. Static analysis (flake8 if available)
Write-Section "Static Analysis (flake8)"
python -m flake8 .
if ($LASTEXITCODE -eq 0) {
 Write-Result "flake8: No issues found" $true
}
else {
 Write-Result "flake8: Issues found" $false
}

# 5. Syntax check and import test for all modules
Write-Section "Syntax & Import Test"
$modules = @(
 "action_system", "core_engine", "interface", "main", "memory_engine", "orchestrator", "system_manager", "system_utils",
 "orama.debug_manager", "orama.resource_manager"
)
foreach ($m in $modules) {
 Write-Host "Testing import: $m"
 python -c "import $m"
 Write-Result "Import $m" ($LASTEXITCODE -eq 0)
}

# 6. Class instantiation smoke test (where possible)
Write-Section "Class Instantiation Smoke Test"
$smoke_classes = @(
 @{mod = "core_engine"; cls = "CognitiveEngine" },
 @{mod = "action_system"; cls = "ActionSystem" },
 @{mod = "memory_engine"; cls = "MemoryEngine" },
 @{mod = "orchestrator"; cls = "Orchestrator" },
 @{mod = "system_manager"; cls = "SystemManager" },
 @{mod = "system_utils"; cls = "SystemMonitor" },
 @{mod = "orama.resource_manager"; cls = "ResourceManager" },
 @{mod = "orama.debug_manager"; cls = "DebugManager" }
)
foreach ($item in $smoke_classes) {
 $mod = $item.mod; $cls = $item.cls
 Write-Host "Trying: from $mod import $cls; $cls({})"
 python -c "from $mod import $cls; $cls({})"
 Write-Result "Instantiate $cls from $mod" ($LASTEXITCODE -eq 0)
}

# 7. Run __main__ blocks for demo/test code
Write-Section "__main__ Demo/Test Execution"
$main_modules = @(
 "core_engine.py", "interface.py", "main.py", "memory_engine.py", "orchestrator.py", "system_utils.py", "system_manager.py"
)
foreach ($f in $main_modules) {
 if (Test-Path $f) {
  Write-Host "Running: python $f (timeout 30s)"
  $job = Start-Job { param($f) python $f } -ArgumentList $f
  Wait-Job $job -Timeout 30 | Out-Null
  if ($job.State -eq "Completed") {
   Receive-Job $job | Write-Host
   Write-Result "Ran $f __main__ block" $true
  }
  else {
   Write-Result "$f __main__ block timed out or failed" $false
   Stop-Job $job | Out-Null
  }
  Remove-Job $job
 }
}

# 8. Pytest discovery (if any test_*.py files exist)
Write-Section "Pytest Discovery"
$testfiles = Get-ChildItem -Recurse -Include "test_*.py"
if ($testfiles.Count -gt 0) {
 Write-Host "Running pytest..."
 pytest
 Write-Result "pytest run" ($LASTEXITCODE -eq 0)
}
else {
 Write-Host "No test_*.py files found. Skipping pytest."
}

# 9. Data directory check
Write-Section "Data Directory Check"
$datadirs = @("data/vector_store", "data/knowledge_graph", "data/documents", "data/parameters")
foreach ($d in $datadirs) {
 if (Test-Path $d) {
  Write-Result "$d exists" $true
 }
 else {
  Write-Result "$d missing" $false
 }
}

# 10. Summary
Write-Section "Test Summary"
Write-Host "If all [PASS] above, your ORAMA system is ready for development!" -ForegroundColor Green

Pop-Location
