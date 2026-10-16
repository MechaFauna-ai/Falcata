$ErrorActionPreference = 'Stop'

$settings = @{
    Severity = @(
        'Information',
        'Warning',
        'Error',
        # syntax errors; without this a .ps1 that does not parse passes
        'ParseError'
    )
    IncludeDefaultRules = $true
    ExcludeRules = @(
        'PSAvoidUsingInvokeExpression'
    )
    # Additional rules that are disabled by default.
    #
    # Some of the skips could be replaced with inline comments if PSScriptAnalyzer
    # supports that in the future (https://github.com/PowerShell/PSScriptAnalyzer/issues/849).
    Rules = @{
        PSAvoidExclaimOperator = @{
            Enable = $true
        }
        PSAvoidLongLines = @{
            Enable = $true
            MaximumLineLength = 120
        }
        PSAvoidSemicolonsAsLineTerminators = @{
            Enable = $true
        }
        PSPlaceCloseBrace = @{
            Enable = $true
            NoEmptyLineBefore = $true
            IgnoreOneLineBlock = $true
            NewLineAfter = $false
        }
        PSPlaceOpenBrace = @{
            Enable = $true
            OnSameLine = $true
            NewLineAfter = $true
            IgnoreOneLineBlock = $true
        }
        PSUseConsistentIndentation = @{
            Enable = $true
            IndentationSize = 4
            PipelineIndentation = 'IncreaseIndentationAfterEveryPipeline'
            Kind = 'space'
        }
        PSUseConsistentWhitespace = @{
            Enable = $true
            CheckInnerBrace = $true
            CheckOpenBrace = $true
            CheckOpenParen = $true
            CheckOperator = $true
            CheckSeparator = $true
            CheckPipe = $true
            CheckPipeForRedundantWhitespace = $true
            CheckParameter = $true
            IgnoreAssignmentOperatorInsideHashTable = $false
        }
        PSUseCorrectCasing = @{
            Enable = $true
        }
    }
}

# this pre-listing of files can be removed whenever PSScriptAnalyzer adds support for exclusions.
#
# see:
#
#  * https://github.com/PowerShell/PSScriptAnalyzer/issues/561
#  * https://github.com/PowerShell/vscode-powershell/issues/3048
#
# lint-powershell.ps1 itself is linted too. It used to be excluded because linting it
# sometimes failed (non-deterministically) with "Object reference not set to an instance
# of an object"; that analyzer crash is retried below.
#
$excludedDirs = '[/\\](bin|external_libs|\.pixi|venv)[/\\]'
$files = @(
    (Get-ChildItem -Path ./ -Recurse -Force -Filter '*.ps1').FullName |
        Where-Object { $_ -notmatch $excludedDirs }
)

# Invoke-ScriptAnalyzer itself sometimes crashes (non-deterministically) with
# "Object reference not set to an instance of an object", a NullReferenceException
# inside the analyzer rather than a lint finding. Retry only that crash.
#
# -EnableExit is deliberately not used: under 'pwsh -file' it does not stop the
# script, it only records an exit code, so the last linted file would decide the
# result and findings in every other file would be ignored. Instead, collect each
# file's findings, print them, and fail after all files have been linted.
$maxAttempts = 3
$failed = $false
foreach ($file in $files) {
    Write-Output "linting '$file'"
    $results = @()
    for ($attempt = 1; ; $attempt++) {
        try {
            $results = @(Invoke-ScriptAnalyzer -Path $file -Settings $settings)
            break
        } catch {
            $isAnalyzerCrash = (
                ($_.Exception.GetBaseException() -is [System.NullReferenceException]) -or
                ($_.Exception.Message -like 'Object reference not set to an instance of an object*')
            )
            if ((-not $isAnalyzerCrash) -or ($attempt -ge $maxAttempts)) {
                throw
            }
            $message = $_.Exception.Message
            Write-Output "Invoke-ScriptAnalyzer crashed on attempt $attempt/$maxAttempts, retrying: $message"
            Start-Sleep -Seconds 5
        }
    }
    if ($results.Count -gt 0) {
        $failed = $true
        $results | Format-Table -AutoSize -Wrap | Out-String -Width 4096 | Write-Output
    }
}

if ($failed) {
    Write-Output 'PSScriptAnalyzer reported findings, see above.'
    exit 1
}
