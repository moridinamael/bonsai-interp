# Prints "ok", or the reason a steering stage must give way: system commit above -Max of its limit, or a 50droid or gmoo-inversebench python process alive.
# 50droid and gmoo-inversebench are the maintainer's own live work on this machine and always have priority (scripts/moo_chain.sh).
param([double]$Max = 0.92)
$os = Get-CimInstance Win32_OperatingSystem
$f = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / $os.TotalVirtualMemorySize
$w = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { "$($_.ExecutablePath) $($_.CommandLine)" -match '50droid|gmoo-inversebench' }).Count
if ($w -gt 0) { "priority (50droid or gmoo-inversebench) python alive: $w (commit {0:P0})" -f $f }
elseif ($f -gt $Max) { "commit {0:P0} of the limit, above {1:P0}" -f $f, $Max }
else { "ok {0:P0}" -f $f }
