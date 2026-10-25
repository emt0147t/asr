# PowerShell build script for KWS Mobile Frontend
# Compiles shared library (.dll), CLI runner (.exe), and unit test suite (.exe) using g++

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$binDir = Join-Path $scriptDir "bin"
$libDir = Join-Path $scriptDir "lib"
if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir | Out-Null }
if (-not (Test-Path $libDir)) { New-Item -ItemType Directory -Path $libDir | Out-Null }

Write-Host "=== Building KWS Mobile Frontend ==="
$cflags = @("-O3", "-Wall", "-Wextra", "-std=c++17", "-Iinclude")

# 1. Compile Shared Library (DLL)
Write-Host "1. Compiling kws_mobile_frontend.dll..."
$dllOut = Join-Path $binDir "kws_mobile_frontend.dll"
$implibOut = "-Wl,--out-implib," + (Join-Path $libDir "libkws_mobile_frontend.a")

& g++ -shared -DKWS_FRONTEND_BUILD_DLL @cflags src/kws_mobile_frontend.cpp -o $dllOut $implibOut
if ($LASTEXITCODE -ne 0) { throw "Failed compiling DLL" }
Write-Host "   Created: $dllOut"

# Also copy DLL to script directory for easy Python ctypes loading
Copy-Item $dllOut (Join-Path $scriptDir "kws_mobile_frontend.dll") -Force

# 2. Compile Unit Test Suite
Write-Host "2. Compiling test_cxx_frontend.exe..."
$testExe = Join-Path $binDir "test_cxx_frontend.exe"
& g++ @cflags tests/test_cxx_frontend.cpp src/kws_mobile_frontend.cpp -o $testExe
if ($LASTEXITCODE -ne 0) { throw "Failed compiling unit tests" }
Write-Host "   Created: $testExe"

# 3. Compile Standalone CLI Runner
Write-Host "3. Compiling run_frontend_cli.exe..."
$cliExe = Join-Path $binDir "run_frontend_cli.exe"
& g++ @cflags tools/run_frontend_cli.cpp src/kws_mobile_frontend.cpp -o $cliExe
if ($LASTEXITCODE -ne 0) { throw "Failed compiling CLI runner" }
Write-Host "   Created: $cliExe"

Write-Host "=== Build Completed Successfully ==="
