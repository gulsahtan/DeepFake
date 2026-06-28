# Explainable Deepfake Artifact Detection Framework

Full-stack ASP.NET Core 8 research demo for a sequential deepfake artifact detection methodology. The backend performs video intake, OpenCV-based frame sampling and face-crop normalization, M2TR-compatible artifact scoring, APE-style explainable reporting, and EQAF scoring. The frontend is a Tailwind and vanilla JavaScript dashboard designed for academic screenshots.

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
│   ├── M2TRArtifactClassifier.cs
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

## OpenAI APE report generation

Set an OpenAI API key before running the backend to enable GPT-4-based Automatic Prompt Engineering (APE) report generation and EQAF scoring:

```bash
set OPENAI_API_KEY=your_api_key_here
set OPENAI_MODEL=gpt-4.1
dotnet run
```

`OPENAI_MODEL` is optional and defaults to `gpt-4.1`. `OPENAI_BASE_URL` is also optional and defaults to `https://api.openai.com/v1`.

When `OPENAI_API_KEY` is not configured or the API call fails, the app uses the local offline fallback so the dashboard remains demonstrable.

## Optional ONNX model

Place a trained seven-class M2TR ONNX model at:

```text
Models/m2tr-artifact.onnx
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


## License

All Rights Reserved © 2026 Fatma Gülşah Tan

This repository is publicly available solely to support research transparency and the peer-review process.

No permission is granted to copy, modify, redistribute, or use any part of this repository without prior written permission from the copyright holder.
