$ErrorActionPreference = 'Stop'
$reportDirectory = $PSScriptRoot
$inputPath = Join-Path $reportDirectory '薄膜强化学习研究思考与实验记录_20261016至20261021.docx'
$qaDirectory = Join-Path $reportDirectory 'qa'
New-Item -ItemType Directory -Path $qaDirectory -Force | Out-Null
$pdfPath = Join-Path $qaDirectory 'research_record.pdf'
$documentApplication = $null
$reportDocument = $null
$initialDocumentCount = -1
try {
    $documentApplication = New-Object -ComObject KWPS.Application
    $initialDocumentCount = $documentApplication.Documents.Count
    if ($initialDocumentCount -eq 0) {
        $documentApplication.Visible = $false
        $documentApplication.DisplayAlerts = 0
    }
    $reportDocument = $documentApplication.Documents.Open($inputPath, $false, $true, $false)
    $reportDocument.Repaginate()
    $pages = $reportDocument.ComputeStatistics(2)
    Write-Output "WPS document pages: $pages"
    $reportDocument.ExportAsFixedFormat($pdfPath, 17)
    Write-Output "Rendered PDF: $pdfPath"
} finally {
    if ($null -ne $reportDocument) {
        $reportDocument.Close(0)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($reportDocument)
    }
    if ($null -ne $documentApplication) {
        if ($initialDocumentCount -eq 0 -and $documentApplication.Documents.Count -eq 0) {
            $documentApplication.Quit()
        }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($documentApplication)
    }
}
