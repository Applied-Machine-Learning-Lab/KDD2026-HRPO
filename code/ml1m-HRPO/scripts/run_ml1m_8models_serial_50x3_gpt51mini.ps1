$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }
$apiBase = if ($env:OPENAI_API_BASE) { $env:OPENAI_API_BASE } else { "" }
$apiKey = $env:OPENAI_API_KEY
if (-not $apiKey) {
    throw "OPENAI_API_KEY is required."
}

$env:ARENA_MAX_WORKERS = "1"
$seeds = @(101, 202, 303)
$dataset = "ml-1m"
$models = @("Random", "SASRec", "GRU4Rec", "TIGER", "TD", "A2C", "HAC", "DDPG")

foreach ($seed in $seeds) {
    $runName = "ML1M_gpt51codexmini_serial1_s${seed}_formal8_u50"

    Write-Host "=================================================="
    Write-Host "[START] dataset=$dataset seed=$seed run=$runName"
    Write-Host "models=$($models -join ',')"
    Write-Host "ARENA_MAX_WORKERS=$env:ARENA_MAX_WORKERS"
    Write-Host "execution_mode=serial"
    Write-Host "=================================================="

    & $pythonExe scripts/run_baseline_simulations.py `
        --dataset $dataset `
        --models $models `
        --model_path Saved `
        --model_path_override SASRec=Saved_agentalign `
        --model_path_override GRU4Rec=Saved_cf_loo50_20260308 `
        --simulation_name $runName `
        --n_avatars 50 `
        --max_pages 10 `
        --items_per_page 1 `
        --execution_mode serial `
        --seed $seed `
        --llm_model gpt-5.1-codex-mini `
        --llm_api_style responses `
        --openai_api_base $apiBase `
        --openai_api_key $apiKey

    if ($LASTEXITCODE -ne 0) {
        throw "run failed: dataset=$dataset seed=$seed run=$runName exit=$LASTEXITCODE"
    }

    Write-Host "[DONE] dataset=$dataset seed=$seed run=$runName"
}

Write-Host "All scheduled runs completed."
