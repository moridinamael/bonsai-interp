# Prints "ok", or the reason a steering stage must give way: system commit above -Max of its limit, or a 50droid worker alive.
# 50droid is the maintainer's own live session on this machine and always has priority (scripts/moo_chain.sh).
param([double]$Max = 0.92)
$os = Get-CimInstance Win32_OperatingSystem
$f = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / $os.TotalVirtualMemorySize
$w = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { "$($_.ExecutablePath) $($_.CommandLine)" -match '50droid' }).Count
if ($w -gt 0) { "50droid workers alive: $w (commit {0:P0})" -f $f }
elseif ($f -gt $Max) { "commit {0:P0} of the limit, above {1:P0}" -f $f, $Max }
else { "ok {0:P0}" -f $f }
