$ErrorActionPreference = 'Stop'
$model = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'models/model.json') -Raw | ConvertFrom-Json
$target = Join-Path $PSScriptRoot ('models/' + $model.filename)
$partial = $target + '.part'
$receipt = Join-Path $PSScriptRoot 'models/download-status.json'
function Save-Status($state, $detail) {
    @{ status = $state; detail = $detail; updated_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding utf8
}
try {
    if (-not (Test-Path -LiteralPath $target)) {
        Save-Status 'downloading' 'Resumable download of the pinned model revision.'
        for ($attempt = 1; $attempt -le 10; $attempt++) {
            & curl.exe --fail --location --retry 5 --retry-delay 3 --continue-at - --output $partial $model.url
            if ($LASTEXITCODE -eq 0) { break }
            if ($attempt -lt 10) {
                Save-Status 'downloading' "Connection interrupted; resuming existing bytes (attempt $($attempt + 1)/10)."
                Start-Sleep -Seconds 5
            }
        }
        if ($LASTEXITCODE -ne 0) { throw "Download failed: curl exit $LASTEXITCODE. Run this script again to resume." }
        $candidate = $partial
    } else { $candidate = $target }
    Save-Status 'verifying' 'Checking exact byte length and SHA256.'
    if ((Get-Item -LiteralPath $candidate).Length -ne $model.size_bytes) { throw 'Model byte length differs from the pinned metadata.' }
    $digest = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $model.sha256) { throw 'Model checksum differs from the pinned metadata.' }
    if ($candidate -eq $partial) { Move-Item -LiteralPath $partial -Destination $target }
    Save-Status 'verified' $digest
    Write-Output "Verified model: $target"
} catch {
    Save-Status 'failed' $_.Exception.Message
    throw
}
