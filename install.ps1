$packages = Get-Content requirements.txt | Where-Object { $_.Trim() -ne "" }

foreach ($pkg in $packages) {
    Write-Host "`n=================================================" -ForegroundColor Cyan
    Write-Host "Starting installation for: $pkg" -ForegroundColor Yellow
    Write-Host "=================================================" -ForegroundColor Cyan
    
    $success = $false
    $retry_count = 0
    $max_retries = 10
    
    while (-not $success -and $retry_count -lt $max_retries) {
        if ($retry_count -gt 0) {
            Write-Host "Retry $($retry_count)/$max_retries for $pkg..." -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
        
        # Run pip install
        .venv\Scripts\pip.exe install --default-timeout=1000 $pkg
        
        if ($LASTEXITCODE -eq 0) {
            $success = $true
            Write-Host "SUCCESS: $pkg" -ForegroundColor Green
        } else {
            $retry_count++
            Write-Host "Network error for $pkg. Preparing to retry..." -ForegroundColor Red
        }
    }
    
    if (-not $success) {
        Write-Host "FAILED to install $pkg after $max_retries retries." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nALL PACKAGES INSTALLED SUCCESSFULLY!" -ForegroundColor Green
