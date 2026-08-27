# Deploy the P1 full bundle to an Arduino UNO Q over verified SSH.
# Every environment-specific value is a required parameter: pass your own board
# ADB serial, your own SSH host fingerprint, and your own Wi-Fi SSID.
param(
    [Parameter(Mandatory = $true)][string]$ArtifactZip,
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$AdbSerial,
    [string]$BoardIp = "",
    [Parameter(Mandatory = $true)][string]$ExpectedFingerprint,
    [Parameter(Mandatory = $true)][string]$ExpectedSsid
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) { if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw "$Name is required" } }
Require-Command adb
Require-Command ssh
Require-Command scp

py -3 -m prototype.halo_safeshift.full_bundle $ArtifactZip 2>$null
if ($LASTEXITCODE -ne 0) { throw "Full ZIP verification failed locally; P0 is untouched." }

adb devices -l
$adbInfo = adb -s $AdbSerial shell "hostname; uname -r; python3 --version; nmcli -t -f active,ssid,ip4.address dev wifi 2>/dev/null || true; df -h /home/arduino; free -m" 
if ($LASTEXITCODE -ne 0) { throw "ADB preflight failed" }
if ($adbInfo -notmatch "qualcomm") { throw "Unexpected board hostname; expected qualcomm" }
if ($adbInfo -notmatch $ExpectedSsid) { throw "Board is not on $ExpectedSsid; do not deploy" }
if (-not $BoardIp) { $BoardIp = [regex]::Match($adbInfo, "(\d{1,3}(?:\.\d{1,3}){3})").Value }
if (-not $BoardIp) { throw "Could not obtain board IP through ADB" }

$fingerprint = ssh-keyscan -t ed25519 $BoardIp 2>$null | ssh-keygen -lf - -E sha256
if ($fingerprint -notmatch [regex]::Escape($ExpectedFingerprint)) { throw "SSH ED25519 fingerprint mismatch; refusing Wi-Fi deployment" }

$remoteBase = "/home/arduino/halo-safeshift-full"
$remoteNew = "$remoteBase/$RunId.new"
$remoteFinal = "$remoteBase/$RunId"
$remoteRollback = "/home/arduino/halo-safeshift-emergency"
ssh "arduino@$BoardIp" "test -d '$remoteRollback' && mkdir -p '$remoteBase' && rm -rf '$remoteNew' && mkdir -p '$remoteNew'"
scp $ArtifactZip "arduino@${BoardIp}:$remoteNew/full.zip"
ssh "arduino@$BoardIp" "cd '$remoteNew' && python3 -c 'from pathlib import Path; import zipfile; zipfile.ZipFile(\"full.zip\").extractall(\"bundle\")' && test -f bundle/full_runtime.py && test -f bundle/full_dashboard.py && find bundle -type f -print0 | sort -z | xargs -0 sha256sum > board-sha256.txt && test ! -e '$remoteFinal' && mv '$remoteNew' '$remoteFinal'"
if ($LASTEXITCODE -ne 0) { throw "Board extraction/hash/atomic rename failed; P0 remains untouched" }

Write-Output "DEPLOYED VERSIONED P1 DIRECTORY: $remoteFinal"
Write-Output "P0 ROLLBACK PRESERVED: $remoteRollback"
Write-Output "Run measure_full_board.py over both ADB and verified SSH before dashboard switching."
