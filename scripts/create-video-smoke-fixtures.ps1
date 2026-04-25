param([string]$Output = 'runs\smoke')
$ErrorActionPreference = 'Stop'
$taskRepository = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRepository
$taskOutput = [System.IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
Add-Type -AssemblyName System.Speech
$taskReviews = @{
    positive = 'I really love this camera. The image quality is excellent and the battery lasts all day. I would absolutely recommend it.'
    negative = 'I really hate this camera. The picture quality is terrible and the battery dies after an hour. I regret buying it and I would not recommend it.'
}
foreach ($taskLabel in @('positive','negative')) {
    $taskVoice = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $taskWav = Join-Path $taskOutput "$taskLabel.wav"
    $taskVideo = Join-Path $taskOutput "$taskLabel.mp4"
    try {
        $taskVoice.SetOutputToWaveFile($taskWav)
        $taskVoice.Speak($taskReviews[$taskLabel])
    } finally {
        $taskVoice.Dispose()
    }
    & ffmpeg -v error -f lavfi -i color=c=0x263626:s=640x360:r=24 -i $taskWav -c:v libx264 -c:a aac -shortest -y $taskVideo
    if ($LASTEXITCODE -ne 0) { throw 'FFmpeg failed to create the spoken review fixture.' }
}
& ffmpeg -v error -f lavfi -i color=c=black:s=320x240:r=24:d=2 -an -c:v libx264 -y (Join-Path $taskOutput 'silent.mp4')
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg failed to create the silent fixture.' }
Write-Output "Created synthetic spoken-review video fixtures in $taskOutput. These are smoke tests, not a sentiment benchmark."
