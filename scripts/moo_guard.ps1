param([double]$Max = 0.92, [int]$MinFreeGpuMb = 0)
$os = Get-CimInstance Win32_OperatingSystem
$f = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / $os.TotalVirtualMemorySize
$w = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { "$($_.ExecutablePath) $($_.CommandLine)" -match '50droid|gmoo-inversebench' }).Count
$free = -1
if ($MinFreeGpuMb -gt 0) { try { $free = [int](((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1).Trim()) } catch { $free = -1 } }
if ($w -gt 0) { "priority (50droid or gmoo-inversebench) python alive: $w (commit {0:P0})" -f $f }
elseif ($f -gt $Max) { "commit {0:P0} of the limit, above {1:P0}" -f $f, $Max }
elseif ($MinFreeGpuMb -gt 0 -and $free -lt $MinFreeGpuMb) { "GPU has $free MB free, under $MinFreeGpuMb MB (commit {0:P0})" -f $f }
else { "ok {0:P0}" -f $f }
