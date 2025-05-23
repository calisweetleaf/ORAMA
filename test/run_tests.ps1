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

foreach ($module_file_rel_path in $python_modules) {
 # Construct absolute path for clarity, though relative should work from root
 $module_file_abs_path = Join-Path (Get-Location).Path $module_file_rel_path
    
 Write-Host ""
 Write-Host "Testing module: $module_file_rel_path"
 Write-Host "Full path: $module_file_abs_path"
 Write-Host "------------------------------------"

 # Test 1: Python syntax check
 Write-Host "Running syntax check (py_compile) for $module_file_rel_path ..."
 python -m py_compile $module_file_abs_path
 if ($LASTEXITCODE -ne 0) {
  Write-Error "Syntax check FAILED for $module_file_rel_path."
  $all_tests_passed = $false
 }
 else {
  Write-Host "Syntax check PASSED for $module_file_rel_path."
 }

 # Test 2: Attempt to run specific modules if they have a testable __main__
 if ($module_file_rel_path -eq "core_engine.py") {
  Write-Host "Attempting to run $module_file_rel_path (expects test_cognitive_engine)..."
  python $module_file_abs_path
  if ($LASTEXITCODE -ne 0) {
   Write-Error "Execution FAILED for $module_file_rel_path."
   $all_tests_passed = $false
  }
  else {
   Write-Host "Execution of $module_file_rel_path finished."
  }
 }
 elseif ($module_file_rel_path -eq "interface.py") {
  Write-Host "Attempting to run '$module_file_rel_path help'..."
  python $module_file_abs_path help # Assumes 'help' is a valid command for interface.py
  if ($LASTEXITCODE -ne 0) {
   Write-Error "Execution FAILED for '$module_file_rel_path help'."
   $all_tests_passed = $false
  }
  else {
   Write-Host "Execution of '$module_file_rel_path help' finished."
  }
 }
 # For other modules like main.py, memory_engine.py, orchestrator.py, etc., 
 # only the syntax check is performed by this basic script, as their __main__
 # blocks might require specific setup, arguments, or start larger processes.
}

Write-Host ""
Write-Host "--------------------------------"
if ($all_tests_passed) {
 Write-Host "All basic tests PASSED." -ForegroundColor Green
}
else {
 Write-Error "Some tests FAILED."
}
Write-Host "--------------------------------"

Pop-Location
