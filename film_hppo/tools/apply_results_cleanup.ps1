param([Parameter(Mandatory=$true)][string]$PlanPath)
$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath $PlanPath -Raw -Encoding utf8 | ConvertFrom-Json
$projectRoot = [IO.Path]::GetFullPath($plan.project_root).TrimEnd('\')
if ($projectRoot -ne 'E:\lithography_machine_lens\film_hppo') { throw 'Unexpected cleanup workspace' }
function CheckedPath([string]$relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    $insideRuns = $absolute.StartsWith($projectRoot+'\runs\',[StringComparison]::OrdinalIgnoreCase)
    $insideOutputs = $absolute.StartsWith($projectRoot+'\outputs\',[StringComparison]::OrdinalIgnoreCase)
    if (-not ($insideRuns -or $insideOutputs)) { throw "Unsafe target: $absolute" }
    return $absolute
}
function CheckFiles([string]$base, $records) {
    if (-not (Test-Path -LiteralPath $base)) { throw "Missing: $base" }
    $item = Get-Item -LiteralPath $base -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Unexpected link: $base" }
    if ($item.PSIsContainer) {
        $allItems = @(Get-ChildItem -LiteralPath $base -Recurse -Force)
        if (@($allItems | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw "Link in tree: $base" }
        if (@($allItems | Where-Object { -not $_.PSIsContainer }).Count -ne @($records).Count) { throw "Tree changed: $base" }
    }
    foreach ($record in $records) {
        $path = if ($record.path) { [IO.Path]::GetFullPath((Join-Path $base $record.path)) } else { $base }
        if ($path -ne $base -and -not $path.StartsWith($base+'\',[StringComparison]::OrdinalIgnoreCase)) { throw "Escaped record: $path" }
        if ((Get-Item -LiteralPath $path).Length -ne $record.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) { throw "Changed file: $path" }
    }
}
# Preflight all sources and final absolute destinations before the first move.
foreach ($m in $plan.moves) {
    $source = CheckedPath $m.source; $destination = CheckedPath $m.destination
    if (Test-Path -LiteralPath $destination) { throw "Destination exists: $destination" }
    CheckFiles $source $m.files
}
foreach ($d in $plan.deletes) { $null = CheckedPath $d.path; CheckFiles (CheckedPath $d.original) $d.files }
Write-Output "Preflight verified; $($plan.moves.Count) moves and $($plan.delete_files) file removals"
foreach ($m in $plan.moves) {
    $source = CheckedPath $m.source; $destination = CheckedPath $m.destination
    $parent = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $parent)) { $null = New-Item -ItemType Directory -Path $parent -Force }
    Move-Item -LiteralPath $source -Destination $destination
    CheckFiles $destination $m.files
}
foreach ($d in $plan.deletes) {
    $target = CheckedPath $d.path
    CheckFiles $target $d.files
    if ($d.is_directory) { Remove-Item -LiteralPath $target -Recurse -Force }
    else { Remove-Item -LiteralPath $target -Force }
}
$done = @{completed_at=(Get-Date -Format o);moved=$plan.moves.Count;deleted_files=$plan.delete_files;deleted_bytes=$plan.delete_bytes}
$done | ConvertTo-Json | Set-Content -LiteralPath (Join-Path (Split-Path -Parent $PlanPath) 'filesystem_completed.json') -Encoding utf8
Write-Output ($done | ConvertTo-Json -Compress)
