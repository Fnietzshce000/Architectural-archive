# Get all PnP signed drivers with details
$drivers = Get-CimInstance Win32_PnPSignedDriver | 
    Where-Object { $_.DeviceName -ne $null } |
    Select-Object DeviceName, DriverVersion, @{N='DriverDate';E={if($_.DriverDate){$_.DriverDate.ToString('yyyy-MM-dd')}else{'N/A'}}}, Manufacturer |
    Sort-Object DeviceName

$drivers | Format-Table -AutoSize | Out-String -Width 300

Write-Host "`n=== DRIVER UPDATE CHECK ==="
Write-Host "Checking Windows Update for driver updates..."

# Check for available driver updates via Windows Update
$UpdateSession = New-Object -ComObject Microsoft.Update.Session
$UpdateSearcher = $UpdateSession.CreateUpdateSearcher()
try {
    $SearchResult = $UpdateSearcher.Search("IsInstalled=0 AND Type='Driver'")
    if ($SearchResult.Updates.Count -eq 0) {
        Write-Host "`nNo driver updates available through Windows Update."
    } else {
        Write-Host "`nAvailable driver updates:"
        foreach ($Update in $SearchResult.Updates) {
            Write-Host "  - $($Update.Title)"
        }
    }
} catch {
    Write-Host "Windows Update driver check failed: $_"
}

Write-Host "`n=== GPU INFO ==="
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, @{N='DriverDate';E={if($_.DriverDate){$_.DriverDate.ToString('yyyy-MM-dd')}else{'N/A'}}} | Format-Table -AutoSize

Write-Host "`n=== NETWORK ADAPTERS ==="
Get-CimInstance Win32_NetworkAdapter | Where-Object { $_.PhysicalAdapter -eq $true } | Select-Object Name, @{N='MACAddress';E={$_.MACAddress}} | Format-Table -AutoSize

Write-Host "`n=== SYSTEM INFO ==="
$os = Get-CimInstance Win32_OperatingSystem
Write-Host "OS: $($os.Caption) Build $($os.BuildNumber)"
Write-Host "Last Boot: $($os.LastBootUpTime)"
