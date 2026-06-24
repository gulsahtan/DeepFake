using System.Globalization;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using DeepfakeArtifactDetection.Models;

namespace DeepfakeArtifactDetection.Services;

public sealed class OpenAiApeAssessmentService
{
    private const string DefaultModel = "gpt-4.1";
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private readonly HttpClient _httpClient;
    private readonly ExplainableReportService _fallbackReports;
    private readonly EqafScoringService _fallbackEqaf;
    private readonly ILogger<OpenAiApeAssessmentService> _logger;

    public OpenAiApeAssessmentService(
        HttpClient httpClient,
        ExplainableReportService fallbackReports,
        EqafScoringService fallbackEqaf,
        ILogger<OpenAiApeAssessmentService> logger)
    {
        _httpClient = httpClient;
        _fallbackReports = fallbackReports;
        _fallbackEqaf = fallbackEqaf;
        _logger = logger;
    }

    public string ProviderStatus
    {
        get
        {
            var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            var model = GetModel();
            return string.IsNullOrWhiteSpace(apiKey)
                ? $"Offline fallback: set OPENAI_API_KEY to enable GPT-4 APE assessment with {model}."
                : $"OpenAI GPT-4 APE assessment enabled with {model}.";
        }
    }

    public async Task<ApeAssessment> GenerateAsync(
        Verdict verdict,
        ClassificationSummary classification,
        CancellationToken cancellationToken)
    {
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            return CreateFallback(verdict, classification, "Offline APE fallback (OPENAI_API_KEY not configured)");
        }

        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, GetResponsesEndpoint());
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
            request.Content = new StringContent(
                JsonSerializer.Serialize(BuildRequest(verdict, classification), JsonOptions),
                Encoding.UTF8,
                "application/json");

            using var response = await _httpClient.SendAsync(request, cancellationToken);
            var body = await response.Content.ReadAsStringAsync(cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("OpenAI APE request failed with status {StatusCode}: {Body}", response.StatusCode, body);
                return CreateFallback(verdict, classification, $"Offline APE fallback (OpenAI returned {(int)response.StatusCode})");
            }

            var assessment = ParseAssessment(body, verdict);
            return assessment with
            {
                UsedOpenAi = true,
                Provider = $"OpenAI GPT-4 APE ({GetModel()})"
            };
        }
        catch (Exception exception) when (exception is HttpRequestException or JsonException or TaskCanceledException)
        {
            _logger.LogWarning(exception, "OpenAI APE assessment failed. Falling back to local report generation.");
            return CreateFallback(verdict, classification, "Offline APE fallback (OpenAI unavailable)");
        }
    }

    private static object BuildRequest(Verdict verdict, ClassificationSummary classification)
    {
        return new
        {
            model = GetModel(),
            input = new object[]
            {
                new
                {
                    role = "system",
                    content = """
                    You are GPT-4 operating as the Automatic Prompt Engineering (APE) explainer for an academic explainable deepfake artifact detection framework.
                    Produce a concise, publication-ready forensic report and assign EQAF comments and scores.
                    Ground every statement in the supplied M2TR scores, frame-level voting, peak anomaly data, and final verdict.
                    Never invent unavailable evidence. Use expert digital-forensics language, but keep each item screenshot-friendly.
                    """
                },
                new
                {
                    role = "user",
                    content = JsonSerializer.Serialize(BuildPromptPayload(verdict, classification), JsonOptions)
                }
            },
            text = new
            {
                format = new
                {
                    type = "json_schema",
                    name = "deepfake_ape_assessment",
                    strict = true,
                    schema = BuildSchema()
                }
            }
        };
    }

    private static object BuildPromptPayload(Verdict verdict, ClassificationSummary classification)
    {
        return new
        {
            apeObjective = "Generate the final explainable forensic report and EQAF score comments according to the article's GPT-4 APE strategy.",
            requiredStatus = verdict.Status,
            diagnosticRule = verdict.Rule,
            model = classification.ModelName,
            modelStatus = classification.ModelStatus,
            referenceMetrics = new
            {
                accuracy = classification.ReferenceAccuracy,
                macroF1 = classification.ReferenceMacroF1,
                auc = classification.ReferenceAuc
            },
            verdict,
            stability = classification.Stability,
            aggregateScores = classification.Scores,
            framePredictions = classification.FramePredictions.Select(frame => new
            {
                frame.FaceId,
                frame.FrameIndex,
                frame.DominantClass,
                frame.DominantProbability,
                frame.IsSuspicious,
                frame.SuspiciousArtifactClass,
                frame.SuspiciousArtifactProbability
            }),
            eqafDimensions = new[]
            {
                "Generality",
                "Clarity",
                "Guidance Potential",
                "Information Sufficiency",
                "Problem Explanation Level"
            },
            constraints = new[]
            {
                "Return status exactly as requiredStatus.",
                "For REAL, potentialGenerativeMechanisms must be an empty array.",
                "For FAKE caused by peak anomaly, mention the localized critical anomaly and 80% peak threshold.",
                "Scores must be numeric values from 1 to 10.",
                "Each EQAF rationale is the comment explaining why the score was assigned."
            }
        };
    }

    private static object BuildSchema()
    {
        var stringArray = new
        {
            type = "array",
            items = new { type = "string" }
        };

        return new
        {
            type = "object",
            additionalProperties = false,
            required = new[]
            {
                "template",
                "status",
                "diagnosticReasons",
                "potentialGenerativeMechanisms",
                "forensicVerificationGuidance",
                "eqaf"
            },
            properties = new
            {
                template = new { type = "string" },
                status = new { type = "string", @enum = new[] { "REAL", "FAKE" } },
                diagnosticReasons = stringArray,
                potentialGenerativeMechanisms = stringArray,
                forensicVerificationGuidance = stringArray,
                eqaf = new
                {
                    type = "object",
                    additionalProperties = false,
                    required = new[] { "overall", "dimensions" },
                    properties = new
                    {
                        overall = new { type = "number" },
                        dimensions = new
                        {
                            type = "array",
                            items = new
                            {
                                type = "object",
                                additionalProperties = false,
                                required = new[] { "name", "score", "rationale" },
                                properties = new
                                {
                                    name = new { type = "string" },
                                    score = new { type = "number" },
                                    rationale = new { type = "string" }
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    private ApeAssessment ParseAssessment(string responseBody, Verdict verdict)
    {
        using var responseDocument = JsonDocument.Parse(responseBody);
        var outputText = ExtractOutputText(responseDocument.RootElement);
        if (string.IsNullOrWhiteSpace(outputText))
        {
            throw new JsonException("OpenAI response did not include output text.");
        }

        using var assessmentDocument = JsonDocument.Parse(outputText);
        var root = assessmentDocument.RootElement;
        var eqaf = ParseEqaf(root.GetProperty("eqaf"));
        var status = GetString(root, "status", verdict.Status).Equals(verdict.Status, StringComparison.OrdinalIgnoreCase)
            ? verdict.Status
            : verdict.Status;

        var report = new ExplainableReport(
            GetString(root, "template", $"APE-GPT4-OpenAI-{GetModel()}"),
            status,
            GetStringArray(root, "diagnosticReasons"),
            verdict.IsFake ? GetStringArray(root, "potentialGenerativeMechanisms") : [],
            GetStringArray(root, "forensicVerificationGuidance"));

        return new ApeAssessment(report, eqaf, true, $"OpenAI GPT-4 APE ({GetModel()})");
    }

    private ApeAssessment CreateFallback(Verdict verdict, ClassificationSummary classification, string provider)
    {
        var report = _fallbackReports.Generate(verdict, classification) with
        {
            Template = "APE-GPT4-Offline-Fallback"
        };
        var eqaf = _fallbackEqaf.Score(report, verdict, classification);
        return new ApeAssessment(report, eqaf, false, provider);
    }

    private static EqafScores ParseEqaf(JsonElement eqafElement)
    {
        var dimensions = eqafElement.GetProperty("dimensions")
            .EnumerateArray()
            .Select(item => new EqafDimension(
                GetString(item, "name", "Unspecified"),
                ClampScore(GetDouble(item, "score", 7.0)),
                GetString(item, "rationale", "GPT-4 assigned this EQAF score from the supplied forensic evidence.")))
            .ToArray();

        var overall = ClampScore(GetDouble(eqafElement, "overall", dimensions.Average(dimension => dimension.Score)));
        return new EqafScores(overall, dimensions);
    }

    private static string ExtractOutputText(JsonElement root)
    {
        if (root.TryGetProperty("output_text", out var outputText) && outputText.ValueKind == JsonValueKind.String)
        {
            return outputText.GetString() ?? string.Empty;
        }

        if (!root.TryGetProperty("output", out var output) || output.ValueKind != JsonValueKind.Array)
        {
            return string.Empty;
        }

        foreach (var outputItem in output.EnumerateArray())
        {
            if (!outputItem.TryGetProperty("content", out var content) || content.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            foreach (var contentItem in content.EnumerateArray())
            {
                if (contentItem.TryGetProperty("text", out var text) && text.ValueKind == JsonValueKind.String)
                {
                    return text.GetString() ?? string.Empty;
                }
            }
        }

        return string.Empty;
    }

    private static IReadOnlyList<string> GetStringArray(JsonElement root, string propertyName)
    {
        if (!root.TryGetProperty(propertyName, out var array) || array.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        return array.EnumerateArray()
            .Where(item => item.ValueKind == JsonValueKind.String)
            .Select(item => item.GetString() ?? string.Empty)
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .ToArray();
    }

    private static string GetString(JsonElement root, string propertyName, string fallback)
    {
        return root.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? fallback
            : fallback;
    }

    private static double GetDouble(JsonElement root, string propertyName, double fallback)
    {
        return root.TryGetProperty(propertyName, out var value) && value.TryGetDouble(out var number)
            ? number
            : fallback;
    }

    private static double ClampScore(double value)
    {
        return Math.Round(Math.Clamp(value, 1, 10), 1);
    }

    private static string GetModel()
    {
        return Environment.GetEnvironmentVariable("OPENAI_MODEL")?.Trim() is { Length: > 0 } model
            ? model
            : DefaultModel;
    }

    private static string GetResponsesEndpoint()
    {
        var baseUrl = Environment.GetEnvironmentVariable("OPENAI_BASE_URL");
        if (string.IsNullOrWhiteSpace(baseUrl))
        {
            baseUrl = "https://api.openai.com/v1";
        }

        return string.Concat(baseUrl.TrimEnd('/'), "/responses");
    }
}
