namespace DeepfakeArtifactDetection.Models;

public static class ArtifactCategories
{
    public const string NormalReal = "Normal/Real";
    public const string FaceBoundary = "Face Boundary Artifacts";
    public const string EyeBlink = "Eye and Blink Anomalies";
    public const string SkinTexture = "Skin Texture Inconsistencies";
    public const string LightingShadow = "Lighting and Shadow Inconsistencies";
    public const string ExpressionMismatch = "Facial Expression Mismatches";
    public const string TemporalFlickering = "Temporal Flickering Artifacts";

    public static readonly string[] Ordered =
    [
        NormalReal,
        FaceBoundary,
        EyeBlink,
        SkinTexture,
        LightingShadow,
        ExpressionMismatch,
        TemporalFlickering
    ];

    public static readonly string[] ArtifactOnly =
    [
        FaceBoundary,
        EyeBlink,
        SkinTexture,
        LightingShadow,
        ExpressionMismatch,
        TemporalFlickering
    ];
}
