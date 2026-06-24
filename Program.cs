using DeepfakeArtifactDetection.Models;
using DeepfakeArtifactDetection.Services;
using Microsoft.AspNetCore.Http.Features;
using System.Globalization;

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddConsole();

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 250 * 1024 * 1024;
});

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSingleton<VideoPreprocessingService>();
builder.Services.AddSingleton<ResNet50ArtifactClassifier>();
builder.Services.AddSingleton<ExplainableReportService>();
builder.Services.AddSingleton<EqafScoringService>();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/api/health", (ResNet50ArtifactClassifier classifier) =>
{
    return Results.Ok(new
    {
        status = "ready",
        model = classifier.ModelStatus,
        utc = DateTimeOffset.UtcNow
    });
});

app.MapPost("/api/analyze", async (
    HttpRequest request,
    IWebHostEnvironment environment,
    VideoPreprocessingService preprocessing,
    ResNet50ArtifactClassifier classifier,
    ExplainableReportService reports,
    EqafScoringService eqaf,
    CancellationToken cancellationToken) =>
{
    if (!request.HasFormContentType)
    {
        return Results.BadRequest(new { error = "Expected multipart/form-data with a video file named 'video'." });
    }

    var form = await request.ReadFormAsync(cancellationToken);
    var video = form.Files["video"];

    if (video is null || video.Length == 0)
    {
        return Results.BadRequest(new { error = "Upload a non-empty .mp4 video file." });
    }

    var extension = Path.GetExtension(video.FileName);
    if (!string.Equals(extension, ".mp4", StringComparison.OrdinalIgnoreCase))
    {
        return Results.BadRequest(new { error = "Only .mp4 videos are accepted by this research demo." });
    }

    var analysisId = Guid.NewGuid();
    var uploadsDirectory = Path.Combine(environment.ContentRootPath, "App_Data", "uploads");
    Directory.CreateDirectory(uploadsDirectory);

    var safeFileName = $"{analysisId:N}.mp4";
    var uploadedVideoPath = Path.Combine(uploadsDirectory, safeFileName);

    await using (var stream = File.Create(uploadedVideoPath))
    {
        await video.CopyToAsync(stream, cancellationToken);
    }

    var steps = PipelineStepFactory.CreateInitial();

    steps.MarkProcessing("upload");
    steps.MarkCompleted("upload", $"Stored {video.FileName} as analysis asset {analysisId:N}.");

    steps.MarkProcessing("preprocess");
    var preprocessingResult = await preprocessing.ExtractFacesAsync(uploadedVideoPath, analysisId, cancellationToken);
    steps.MarkCompleted(
        "preprocess",
        $"Sampled {preprocessingResult.SampledFrameCount} frames and normalized {preprocessingResult.Faces.Count} facial regions.");

    steps.MarkProcessing("classify");
    var classification = await classifier.ClassifyAsync(preprocessingResult.Faces, cancellationToken);
    steps.MarkCompleted(
        "classify",
        $"ResNet-50 artifact head selected {classification.DominantClass} at {FormatPercent(classification.DominantProbability)}.");

    steps.MarkProcessing("report");
    var verdict = VerdictEvaluator.Evaluate(classification, threshold: 0.50);
    var report = reports.Generate(verdict, classification);
    steps.MarkCompleted("report", $"APE forensic report generated with {verdict.Status} diagnostic path.");

    steps.MarkProcessing("eqaf");
    var eqafScores = eqaf.Score(report, verdict, classification);
    steps.MarkCompleted("eqaf", "Explanation quality scored against the five EQAF dimensions.");

    var response = new AnalysisResponse(
        analysisId,
        DateTimeOffset.UtcNow,
        video.FileName,
        preprocessingResult.SampledFrameCount,
        preprocessingResult.Faces.Select(FaceRegionDto.FromExtractedFace).ToArray(),
        classification,
        verdict,
        report,
        eqafScores,
        steps);

    return Results.Ok(response);
});

app.MapFallbackToFile("index.html");

app.Run();

static string FormatPercent(double value, int decimals = 1)
{
    return string.Concat((value * 100).ToString($"F{decimals}", CultureInfo.InvariantCulture), "%");
}
