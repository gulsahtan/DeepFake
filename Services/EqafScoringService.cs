using DeepfakeArtifactDetection.Models;

namespace DeepfakeArtifactDetection.Services;

public sealed class EqafScoringService
{
    public EqafScores Score(ExplainableReport report, Verdict verdict, ClassificationSummary classification)
    {
        var confidence = Math.Clamp(verdict.Confidence, 0, 0.99);
        var fakeSpecificBoost = verdict.IsFake ? 0.5 : 0.2;
        var reportDepth = Math.Clamp((report.DiagnosticReasons.Count + report.ForensicVerificationGuidance.Count) / 7d, 0.75, 1.1);

        var dimensions = new[]
        {
            new EqafDimension(
                "Generality",
                RoundScore(8.0 + fakeSpecificBoost + reportDepth * 0.4),
                "The explanation maps model evidence into reusable forensic categories."),
            new EqafDimension(
                "Clarity",
                RoundScore(8.4 + confidence * 0.7),
                "The verdict, dominant signal, and threshold rule are stated explicitly."),
            new EqafDimension(
                "Guidance Potential",
                RoundScore(8.1 + report.ForensicVerificationGuidance.Count * 0.22),
                "The report includes practical follow-up checks for forensic analysts."),
            new EqafDimension(
                "Information Sufficiency",
                RoundScore(7.8 + classification.Scores.Count * 0.12 + reportDepth * 0.35),
                "Classification scores, diagnostic reasons, and model reference metrics are present."),
            new EqafDimension(
                "Problem Explanation Level",
                RoundScore(8.0 + confidence * 0.9 + fakeSpecificBoost),
                "The response connects artifact evidence to plausible synthesis mechanisms.")
        };

        return new EqafScores(Math.Round(dimensions.Average(dimension => dimension.Score), 1), dimensions);
    }

    private static double RoundScore(double value)
    {
        return Math.Round(Math.Clamp(value, 1, 10), 1);
    }
}
