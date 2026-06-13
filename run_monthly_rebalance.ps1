<#
    run_monthly_rebalance.ps1
    --------------------------------------------------------------------------
    One double-click monthly rebalance for the SPY factor strategy:

      1. Checks that IBKR TWS is reachable (prompts you to start it if not).
      2. Runs  dashboard/update_data.py  (EDGAR + IBKR prices + rescore +
         refresh picks/cluster buys -> backtest_metrics.json).
      3. Commits the refreshed data files and pushes to GitHub, which triggers
         a Streamlit Cloud redeploy.

    For the YEARLY run (re-weight backtest + universe change) run instead, from
    the dashboard folder:   python update_data.py --full
    ...then commit & push (or just run this script after editing $FullRun=$true).
#>

[CmdletBinding()]
param(
    [switch]$FullRun  # pass -FullRun for the annual backtest re-weight
)

$ErrorActionPreference = 'Stop'

# Full path to the interpreter that has the pipeline deps (ib_insync, pandas...)
$Python   = 'C:\Users\Jia Wei\AppData\Local\Programs\Python\Python312\python.exe'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path          # ...\SPY
$Dash     = Join-Path $RepoRoot 'dashboard'
$IbHost   = '127.0.0.1'
$IbPort   = 7496                                                      # live TWS

function Test-IBKR {
    # Returns $true if something is listening on the TWS API port.
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async  = $client.BeginConnect($IbHost, $IbPort, $null, $null)
        $ok     = $async.AsyncWaitHandle.WaitOne(2000)               # 2s timeout
        $result = $ok -and $client.Connected
        $client.Close()
        return $result
    } catch {
        return $false
    }
}

try {
    # Green console theme
    $Host.UI.RawUI.BackgroundColor = 'Black'
    $Host.UI.RawUI.ForegroundColor = 'Green'
    Clear-Host

    Write-Host ''
    Write-Host '=== Midas Collective Update - Monthly Rebalance ===' -ForegroundColor Green
    Write-Host ''

    if (-not (Test-Path $Python)) {
        throw "Python not found at $Python. Edit `$Python at the top of this script."
    }

    # ---- 1. IBKR pre-flight (soft: prompt + retry, never hard-fail) ----------
    while (-not (Test-IBKR)) {
        Write-Host "IBKR TWS is not reachable on ${IbHost}:${IbPort}." -ForegroundColor Yellow
        Write-Host 'Please start IBKR Trader Workstation, log in to the LIVE account,' -ForegroundColor Yellow
        Write-Host 'and make sure the API is enabled, then come back here.'           -ForegroundColor Yellow
        $ans = Read-Host 'Press ENTER to retry  (or type Q then ENTER to quit)'
        if ($ans -match '^[Qq]') {
            Write-Host 'Aborted - IBKR not started. Nothing was changed.' -ForegroundColor Red
            return
        }
    }
    Write-Host 'IBKR connection detected.' -ForegroundColor Green
    Write-Host ''

    # ---- 2. Run the data update ---------------------------------------------
    Set-Location $Dash
    if ($FullRun) {
        Write-Host 'Running FULL update (annual backtest re-weight)...' -ForegroundColor Cyan
        & $Python 'update_data.py' '--full'
    } else {
        Write-Host 'Running monthly update...' -ForegroundColor Cyan
        & $Python 'update_data.py'
    }
    if ($LASTEXITCODE -ne 0) {
        throw "update_data.py exited with code $LASTEXITCODE. Nothing was committed or pushed."
    }

    # ---- 3. Commit & push ----------------------------------------------------
    Set-Location $RepoRoot
    git add -A
    $pending = git status --porcelain
    if ([string]::IsNullOrWhiteSpace($pending)) {
        Write-Host 'No changes to commit - data already up to date.' -ForegroundColor Yellow
    } else {
        $label = if ($FullRun) { 'Annual rebalance' } else { 'Monthly rebalance' }
        $stamp = Get-Date -Format 'yyyy-MM'
        git commit -m "$label $stamp"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed (exit $LASTEXITCODE)." }
        git push origin main
        if ($LASTEXITCODE -ne 0) { throw "git push failed (exit $LASTEXITCODE). Check your GitHub credentials." }
        Write-Host ''
        Write-Host 'Pushed to GitHub. Streamlit Cloud will redeploy shortly.' -ForegroundColor Green
    }

    Write-Host ''
    Write-Host 'Done.' -ForegroundColor Green
}
catch {
    Write-Host ''
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Write-Host ''
    Read-Host 'Press ENTER to close this window'
}
