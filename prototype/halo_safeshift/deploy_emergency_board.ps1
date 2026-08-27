param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactZip,

    [Parameter(Mandatory = $true)][string]$Device,
    [string]$RemoteDir = "/home/arduino/halo-safeshift-emergency",
    [int]$Repeat = 500
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ArtifactZip -PathType Leaf)) {
    throw "Artifact ZIP not found: $ArtifactZip"
}

$adbState = (& adb -s $Device get-state 2>&1 | Out-String).Trim()
if ($adbState -ne "device") {
    throw "UNO Q $Device is not connected (adb state: $adbState)"
}

$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$tempRoot = Join-Path $env:TEMP "halo-safeshift-$stamp"
New-Item -ItemType Directory -Path $tempRoot | Out-Null

try {
    Expand-Archive -LiteralPath $ArtifactZip -DestinationPath $tempRoot -Force

    $required = @("edge_model.json", "edge_runner.py", "replay_samples.json", "manifest.json", "metrics.json")
    foreach ($name in $required) {
        $path = Join-Path $tempRoot $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Artifact ZIP is missing required file: $name"
        }
    }

    $model = Get-Content -LiteralPath (Join-Path $tempRoot "edge_model.json") -Raw | ConvertFrom-Json
    if ($model.model_type -notin @("mlp_sparse_int8", "persistence")) {
        throw "Unsupported emergency model type: $($model.model_type)"
    }

    & adb -s $Device shell "rm -rf '$RemoteDir.new' && mkdir -p '$RemoteDir.new'"
    if ($LASTEXITCODE -ne 0) { throw "Failed to create staging directory on UNO Q" }

    foreach ($name in $required) {
        & adb -s $Device push (Join-Path $tempRoot $name) "$RemoteDir.new/$name"
        if ($LASTEXITCODE -ne 0) { throw "ADB push failed for $name" }
    }

    & adb -s $Device shell "rm -rf '$RemoteDir.previous'; if [ -d '$RemoteDir' ]; then mv '$RemoteDir' '$RemoteDir.previous'; fi; mv '$RemoteDir.new' '$RemoteDir'"
    if ($LASTEXITCODE -ne 0) { throw "Failed to activate staged UNO Q bundle" }

    $remoteHashes = (& adb -s $Device shell "cd '$RemoteDir' && sha256sum edge_model.json edge_runner.py replay_samples.json metrics.json" | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "Remote SHA-256 verification command failed" }

    $runtimeJson = (& adb -s $Device shell "cd '$RemoteDir' && python3 edge_runner.py --model edge_model.json --samples replay_samples.json --repeat $Repeat" | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "UNO Q inference command failed`n$runtimeJson" }

    $runtime = $runtimeJson | ConvertFrom-Json
    if ($runtime.runtime -ne "Arduino UNO Q") {
        throw "Runtime provenance mismatch: expected Arduino UNO Q, got $($runtime.runtime)"
    }
    if ($runtime.repeat -ne $Repeat) {
        throw "Runtime repeat mismatch: expected $Repeat, got $($runtime.repeat)"
    }
    if ([double]$runtime.reference_max_abs_error_degc -gt 0.0001) {
        throw "UNO Q output disagrees with the Colab reference: $($runtime.reference_max_abs_error_degc) degC"
    }

    $evidenceDir = Join-Path (Split-Path -Parent $PSScriptRoot) "..\evidence\halo-safeshift\board-emergency-$stamp"
    $evidenceDir = [System.IO.Path]::GetFullPath($evidenceDir)
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null

    $evidence = [ordered]@{
        artifact = "board-runtime.json"
        measured_utc = (Get-Date).ToUniversalTime().ToString("o")
        device_scope = "Arduino UNO Q connected through ADB; serial intentionally omitted"
        remote_dir = $RemoteDir
        artifact_zip = [System.IO.Path]::GetFileName($ArtifactZip)
        artifact_zip_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArtifactZip).Hash.ToLowerInvariant()
        remote_hash_output = $remoteHashes.Trim()
        runtime = $runtime
        claim_boundary = @(
            "Latency is Python sparse-INT8/persistence inference on this UNO Q run only.",
            "No NPU, QNN, Hexagon or GPU acceleration is claimed.",
            "Model quality remains the retrospective chronological Colab result in metrics.json."
        )
    }
    $evidence | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $evidenceDir "board-runtime.json") -Encoding utf8

    Write-Host "UNO_Q_INFERENCE_VERIFIED"
    Write-Host "Evidence: $(Join-Path $evidenceDir 'board-runtime.json')"
    Write-Host "Model: $($runtime.model_type)"
    Write-Host "Median latency: $($runtime.latency_ms.median) ms"
    Write-Host "P95 latency: $($runtime.latency_ms.p95) ms"
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
