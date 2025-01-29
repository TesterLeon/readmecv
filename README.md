# CCTV Trailer Detection

A computer vision system for detecting and analyzing trailers in CCTV footage using advanced deep learning models and geometric analysis. The system performs ground plane detection, perspective transformation, and trailer dimension estimation while integrating with real-world GPS coordinates.

## Features

- **Ground Plane Detection**: Uses RANSAC algorithm to detect and model the ground surface
- **Trailer Detection & Segmentation**: Employs YOLOv8 and SAM2 for accurate trailer detection and segmentation
- **Perspective Transformation**: Maps between real-world GPS coordinates and image space
- **Parking Spot Integration**: Visualizes designated parking spots with real GPS coordinates
- **Dimension Estimation**: Calculates approximate trailer dimensions and orientation
- **Ground Footprint Projection**: Projects trailer footprints onto the detected ground plane
- **Real-time Visualization**: Renders detected trailers, parking spots, and ground plane in an interactive display

## Architecture

The system integrates several state-of-the-art computer vision models and algorithms:

- **Object Detection**: YOLOv8x for initial trailer detection
- **Segmentation**: SAM2 (Segment Anything Model 2) for precise trailer segmentation
- **Ground Plane Analysis**: RANSAC for robust ground plane detection
- **Geometric Analysis**: Custom algorithms for trailer orientation and dimension estimation

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- SAM2 and YOLOv8 models

### Required Packages

```bash
pip install opencv-python
pip install numpy
pip install torch
pip install ultralytics
pip install scipy
pip install scikit-learn
```

### Model Setup

1. Download required model weights:
   - SAM2 checkpoint: `sam2.1_hiera_large.pt`
   - YOLOv8 weights: `yolov8x.pt`

2. Place model files in the project directory:
```
project/
├── sam2.1_hiera_large.pt
├── yolov8x.pt
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_l.yaml
```

## Usage

### Input Requirements

1. Place input files in the project directory:
   - RGB image: `input.png`
   - Depth map: `depth_input.png` (can be generated using [Depth Anything V2](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2))

2. Configure GPS coordinates in the code:
```python
gps_coords = [
    (51.06346248, 13.71108476),
    (51.06348720, 13.71106312),
    (51.06354688, 13.71080708),
    (51.06335838, 13.71081740),
]

pixel_coords = [
    (410, 763),
    (628, 738),
    (1168, 393),
    (399, 366),
]
```

### Parking Lot Configuration (Optional)

Create a `parking-lots.json` file with the following structure:
```json
[
  {
    "name": "Stellplatz-A-34",
    "center_long": 13.708249451894732,
    "center_lat": 51.06338451377482,
    "rotation": 133,
    "allowedType": 1,
    "maxStack": 1,
    "color": "#6f7228",
    "long1": 13.708108772888155,
    "lat1": 51.06344505655194,
    "long2": 13.708188449210809,
    "lat2": 51.06344250708165,
    "long3": 13.708390113027562,
    "lat3": 51.063324235076124,
    "long4": 13.708309758120503,
    "lat4": 51.06332669136053
  }
]
```

### Running the System

```python
python main.py
```

The system will:
1. Load and initialize all required models
2. Process the input image for trailer detection
3. Perform ground plane detection and perspective transformation
4. Generate visualization with detected trailers and parking spots
5. Display and save the results as `combined_detection_ground.png`

## Performance Notes

- Processing time is approximately 60 seconds per frame, primarily due to the SAM2 segmentation
- For faster processing, consider:
  - Using edge detection instead of full segmentation
  - Implementing a lighter segmentation model
  - Batch processing multiple frames

## Future Improvements

- Implement real-time processing capabilities
- Add support for multiple camera viewpoints
- Integrate with parking management systems
- Improve processing speed through model optimization
- Add support for nighttime detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- SAM2 by Meta Research
- OpenCV community
- Depth Anything V2 team
