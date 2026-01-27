# Kill process occupying specified port
param(
    [int]$Port = 0,
    [string]$Type = ""  # "backend" or "frontend"
)

# Auto-detect port based on type
if ($Type -eq "backend") {
    # Read backend port from .env
    $envFile = Join-Path $PSScriptRoot "..\.env"
    if (Test-Path $envFile) {
        $envContent = Get-Content $envFile
        $portLine = $envContent | Where-Object { $_ -match "^VITE_API_PORT\s*=" }
        if ($portLine) {
            $Port = [int]($portLine -replace "^VITE_API_PORT\s*=\s*", "")
        }
    }
    if ($Port -eq 0) { $Port = 8001 }
} elseif ($Type -eq "frontend") {
    $Port = 3100
}

if ($Port -eq 0) {
    Write-Host "Error: Please specify -Port or -Type parameter"
    exit 1
}

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object State -eq 'Listen'

if ($connections) {
    $procIds = $connections.OwningProcess | Select-Object -Unique
    foreach ($procId in $procIds) {
        Write-Host "Killing process on port $Port, PID: $procId"
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "No process on port $Port"
}
