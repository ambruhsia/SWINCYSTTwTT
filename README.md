# Swin Transformer PCOS Classifier

## Description
This project implements a PCOS classification model using the Swin Transformer architecture. The model classifies images into two categories: "Infected" and "Not Infected."

## Project Structure
- **swincyst/**: Contains the main application files.
  - **app.py**: FastAPI application that serves the model and provides endpoints for predictions.
  - **best_swin_pcos_model(4).pth**: The trained model weights.
  - **Dockerfile**: Docker configuration for containerizing the application.
  - **requirements.txt**: Python dependencies required for the project.
  - **trained_model/**: Directory for storing trained models.
- **pcos_classifier.py**: Contains the implementation of the SwinPCOSClassifier class, which handles model training and evaluation.
- **model_metadata_20250223_110549.json**: Metadata about the model, including its type, architecture, and framework versions.
- **swincyst-ii-1.ipynb**: Jupyter notebook for experimentation and analysis.
- **test_image.jpg** and **test_image2.jpg**: Sample images for testing the model.
- **img_0_38.jpg** and **img_0_75.jpg**: Additional images used in the project.

## Model Details
- **Model Type**: Swin Transformer
- **Base Model**: swin_base_patch4_window7_224
- **Classes**: 
  - Not Infected
  - Infected
- **Framework Versions**:
  - PyTorch: 2.5.1+cu121
  - timm: 1.0.12
0.0.0 --port 8000
```

## License
This project is licensed under the MIT License.
# swincystTTwTT
# swincystTTwTT
