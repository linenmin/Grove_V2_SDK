# Windows host probe. No bytes are transmitted; opening a serial port may affect board reset lines.
[CmdletBinding()]
param(
    [ValidateSet('list', 'capture')][string]$Action = 'list',
    [string]$Port,
    [string]$SerialNumber,
    [ValidateRange(1200, 3000000)][int]$Baud = 921600,
    [ValidateRange(1, 60)][int]$Seconds = 8,
    [string]$OutputRoot = (Join-Path $PSScriptRoot '../logs/probe')
)
$ErrorActionPreference = 'Stop'
$devices = @(Get-CimInstance Win32_SerialPort | ForEach-Object {
    [pscustomobject]@{ port = $_.DeviceID; name = $_.Name; id = $_.PNPDeviceID }
})
if ($Action -eq 'list') {
    ConvertTo-Json -InputObject $devices -Depth 4
    exit 0
}
$candidates = @($devices | Where-Object {
    $_.id -like 'USB\VID_1A86&PID_55D3\*' -and
    (!$Port -or $_.port -eq $Port) -and
    (!$SerialNumber -or ($_.id -split '\\')[-1] -eq $SerialNumber)
})
if ($candidates.Count -ne 1) {
    throw 'Expected exactly one matching USB bridge. Run -Action list and specify -Port and/or -SerialNumber. Bridge ID alone does not prove board identity.'
}
$device = $candidates[0]
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$runDir = [IO.Path]::GetFullPath((Join-Path $OutputRoot $runId))
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$rawPath = Join-Path $runDir 'uart.bin'
$serial = [IO.Ports.SerialPort]::new($device.port,$Baud,[IO.Ports.Parity]::None,8,[IO.Ports.StopBits]::One)
$serial.DtrEnable = $false
$serial.RtsEnable = $false
$serial.ReadTimeout = 200
$stream = $null
$failure = $null
$count = 0
try {
    $serial.Open()
    $stream = [IO.File]::Create($rawPath)
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $buffer = [byte[]]::new(8192)
    while ($timer.Elapsed.TotalSeconds -lt $Seconds) {
        try {
            $n = $serial.Read($buffer,0,$buffer.Length)
            $stream.Write($buffer,0,$n)
            $count += $n
        } catch [TimeoutException] {}
    }
} catch {
    $failure = $_.Exception.Message
} finally {
    if ($stream) { $stream.Dispose() }
    $serial.Dispose()
}
$text = ''
if (Test-Path -LiteralPath $rawPath) {
    $text = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($rawPath))
    [IO.File]::WriteAllText((Join-Path $runDir 'uart.log'), $text, [Text.UTF8Encoding]::new($false))
}
$summary = [ordered]@{
    captured_at = (Get-Date -Format o); device = $device; baud = $Baud
    requested_seconds = $Seconds; bytes = $count; transmitted_bytes = 0
    camera_init_failed = $text.Contains('camera init fail')
    model_io = @([regex]::Matches($text,'model io:[^\r\n]*') | ForEach-Object { $_.Value })
    error = $failure; evidence_directory = $runDir
    note = 'A silent capture does not prove success or failure. Opening COM may affect reset lines. Model identity needs a firmware/model hash.'
}
$json = $summary | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText((Join-Path $runDir 'summary.json'), $json, [Text.UTF8Encoding]::new($false))
$json
if ($failure) { exit 2 }
if ($summary.camera_init_failed) { exit 3 }
if ($count -eq 0) { exit 4 }
exit 0
