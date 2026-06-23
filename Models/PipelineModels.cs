namespace DeepfakeArtifactDetection.Models;

public sealed record CropBox(int X, int Y, int Width, int Height);

public sealed record ExtractedFace(
    string Id,
    int FrameIndex,
    double TimestampSeconds,
    string ImageUrl,
    string PhysicalPath,
    CropBox Crop,
    double Sharpness,
    double Illumination,
    double TextureEnergy);

public sealed record FaceRegionDto(
    string Id,
    int FrameIndex,
    double TimestampSeconds,
    string ImageUrl,
    CropBox Crop,
    double Sharpness,
    double Illumination,
    double TextureEnergy)
{
    public static FaceRegionDto FromExtractedFace(ExtractedFace face)
    {
        return new FaceRegionDto(
            face.Id,
            face.FrameIndex,
            Math.Round(face.TimestampSeconds, 2),
            face.ImageUrl,
            face.Crop,
            Math.Round(face.Sharpness, 3),
            Math.Round(face.Illumination, 3),
            Math.Round(face.TextureEnergy, 3));
    }
}

public sealed record PreprocessingResult(int SampledFrameCount, IReadOnlyList<ExtractedFace> Faces);

public sealed record ArtifactScore(string ClassName, double Probability, string Evidence);

public sealed record ClassificationSummary(
    string ModelName,
    string ModelStatus,
    double ReferenceAccuracy,
    double ReferenceMacroF1,
    double ReferenceAuc,
    string DominantClass,
    double DominantProbability,
    IReadOnlyList<ArtifactScore> Scores);

public sealed record Verdict(
    string Status,
    bool IsFake,
    double Confidence,
    string DominantSignal,
    double ArtifactThreshold,
    string Rule);

public sealed record ExplainableReport(
    string Template,
    string Status,
    IReadOnlyList<string> DiagnosticReasons,
    IReadOnlyList<string> PotentialGenerativeMechanisms,
    IReadOnlyList<string> ForensicVerificationGuidance);

public sealed record EqafDimension(string Name, double Score, string Rationale);

public sealed record EqafScores(double Overall, IReadOnlyList<EqafDimension> Dimensions);

public sealed record PipelineStep(string Id, string Label, string Status, string Detail);

public sealed record AnalysisResponse(
    Guid AnalysisId,
    DateTimeOffset CreatedAt,
    string OriginalFileName,
    int SampledFrameCount,
    IReadOnlyList<FaceRegionDto> Faces,
    ClassificationSummary Classification,
    Verdict Verdict,
    ExplainableReport Report,
    EqafScores Eqaf,
    IReadOnlyList<PipelineStep> Steps);

public static class PipelineStepFactory
{
    public static List<PipelineStep> CreateInitial()
    {
        return
        [
            new PipelineStep("upload", "Video Upload", "Pending", "Awaiting MP4 evidence."),
            new PipelineStep("preprocess", "Frame Sampling & Face Cropping", "Pending", "Awaiting video preprocessing."),
            new PipelineStep("classify", "ResNet-50 Artifact Classification", "Pending", "Awaiting normalized facial regions."),
            new PipelineStep("report", "XAI Report Generation", "Pending", "Awaiting diagnostic path."),
            new PipelineStep("eqaf", "EQAF Quality Assessment", "Pending", "Awaiting generated explanation.")
        ];
    }

    public static void MarkProcessing(this List<PipelineStep> steps, string id)
    {
        Replace(steps, id, "Processing", null);
    }

    public static void MarkCompleted(this List<PipelineStep> steps, string id, string detail)
    {
        Replace(steps, id, "Completed", detail);
    }

    private static void Replace(List<PipelineStep> steps, string id, string status, string? detail)
    {
        var index = steps.FindIndex(step => step.Id == id);
        if (index < 0)
        {
            return;
        }

        var current = steps[index];
        steps[index] = current with
        {
            Status = status,
            Detail = detail ?? current.Detail
        };
    }
}

public static class VerdictEvaluator
{
    public static Verdict Evaluate(ClassificationSummary classification, double threshold)
    {
        var normal = classification.Scores.First(score => score.ClassName == ArtifactCategories.NormalReal);
        var artifacts = classification.Scores.Where(score => score.ClassName != ArtifactCategories.NormalReal).ToArray();
        var strongestArtifact = artifacts.MaxBy(score => score.Probability)!;
        var allArtifactsBelowThreshold = artifacts.All(score => score.Probability < threshold);
        var isReal = classification.DominantClass == ArtifactCategories.NormalReal || allArtifactsBelowThreshold;

        if (isReal)
        {
            return new Verdict(
                "REAL",
                false,
                Math.Max(normal.Probability, 1 - strongestArtifact.Probability),
                ArtifactCategories.NormalReal,
                threshold,
                "Normal/Real is dominant or every artifact score is below the configured 50% forensic threshold.");
        }

        return new Verdict(
            "FAKE",
            true,
            strongestArtifact.Probability,
            strongestArtifact.ClassName,
            threshold,
            "A deepfake artifact class is the dominant diagnostic signal above the configured forensic threshold.");
    }
}
