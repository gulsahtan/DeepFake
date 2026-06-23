# Explainable Deepfake Artifact Detection Framework

Full-stack ASP.NET Core 8 research demo for a sequential deepfake artifact detection methodology. The backend performs video intake, OpenCV-based frame sampling and face-crop normalization, ResNet-50-compatible artifact scoring, APE-style explainable reporting, and EQAF scoring. The frontend is a Tailwind and vanilla JavaScript dashboard designed for academic screenshots.

## Structure

```text
DeepFake/
├── Program.cs
├── DeepfakeArtifactDetection.csproj
├── Models/
│   ├── ArtifactCategory.cs
│   └── PipelineModels.cs
├── Services/
│   ├── EqafScoringService.cs
│   ├── ExplainableReportService.cs
│   ├── ResNet50ArtifactClassifier.cs
│   └── VideoPreprocessingService.cs
├── wwwroot/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── extracted_faces/
└── App_Data/
    └── uploads/
```

## Run

```bash
dotnet restore
dotnet run
```

Open the URL printed by ASP.NET Core and upload an `.mp4` file.

## Optional ONNX model

Place a trained seven-class ResNet-50 ONNX model at:

```text
Models/resnet50-artifact.onnx
```

The expected class order is:

1. Normal/Real
2. Face Boundary Artifacts
3. Eye and Blink Anomalies
4. Skin Texture Inconsistencies
5. Lighting and Shadow Inconsistencies
6. Facial Expression Mismatches
7. Temporal Flickering Artifacts

When no model is present, the classifier uses a deterministic calibrated simulator so the complete methodology remains demonstrable.
