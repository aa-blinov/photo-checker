# Photo Validator

A module for validating photos against requirements: white background, proper face centering, and no head tilt.

## Features

- **White background check**: Analysis of white pixel ratio using HSV filtering
- **Face detection**: Using YuNet for accurate face detection and landmarks
- **Centering check**: Control of nose position relative to image center (±10% horizontal, ±25% vertical)
- **Head tilt check**: Control of roll (±15°) and pitch (±15°) angles
- **Automatic visualization**: On errors, creates debug images with landmarks overlaid on reference frame

## Requirements

- Python >= 3.10
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pydantic >= 2.0.0
- Pillow >= 10.0.0

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Basic usage

```bash
uv run photo.py <path_to_photo>
```

or

```bash
python photo.py <path_to_photo>
```

### Examples

```bash
# Check photo
uv run photo.py test_imgs/photo_good1.jpg

# Success result
{
  "success": true,
  "metadata": {
    "image_size": {
      "width_px": 640,
      "height_px": 640
    },
    "background_white_ratio_percent": 38.8,
    "faces_count": 1,
    "centering_horizontal_percent": 4.4,
    "centering_vertical_percent": 23.8,
    "eye_angle_degrees": -2.03,
    "pitch_angle_degrees": -3.68
  }
}

# Error result
{
  "success": false,
  "code": "background_not_white",
  "text": "Фон не белый",
  "metadata": {
    "image_size": {
      "width_px": 1152,
      "height_px": 560
    },
    "background_white_ratio_percent": 2.8
  },
  "debug_image_path": "outputs\\cn_photo_bad1.jpg"
}
```

## Error Codes

| Code                     | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| `load_error`           | File loading error                                   |
| `background_not_white` | Background is not white (less than 30% white pixels) |
| `face_not_recognized`  | No face detected                                     |
| `multiple_faces`       | Multiple faces detected                              |
| `centering_left`       | Face shifted to the left                             |
| `centering_right`      | Face shifted to the right                            |
| `centering_up`         | Face shifted upward                                  |
| `centering_down`       | Face shifted downward                                |
| `tilt_left`            | Head tilted left (roll > 15°)                       |
| `tilt_right`           | Head tilted right (roll < -15°)                     |
| `tilt_up`              | Head tilted up (pitch < -15°)                       |
| `tilt_down`            | Head tilted down (pitch > 15°)                      |

## Metadata

All results contain metadata with measured parameters:

- **image_size**: Image dimensions (width_px, height_px)
- **background_white_ratio_percent**: White background percentage (0-100%)
- **faces_count**: Number of detected faces
- **centering_horizontal_percent**: Horizontal nose offset from center (0% = centered, 100% = at edge)
- **centering_vertical_percent**: Vertical nose offset from center
- **eye_angle_degrees**: Head tilt angle left-right (roll, in degrees)
- **pitch_angle_degrees**: Head tilt angle up-down (pitch, in degrees)

## Debug Visualization

On validation error, a debug image is automatically created in the `outputs/` directory with the name `cn_<original_filename>`.

Debug image contains:

- Original photo (semi-transparent, 50%)
- Reference frame `border_black.png` with markup lines
- Red dots — all 5 face landmarks
- Green line — between eyes (for roll control)
- Cyan dot — nose (key point for centering)
- Yellow cross — image center

## Architecture

The project uses design patterns for scalability and ease of extension:

- **Chain of Responsibility**: Chain of independent validators
- **Strategy**: Pluggable face detectors (YuNet)
- **Facade**: Single entry point through `PhotoAnalyzer`

### Main Components

```
PhotoAnalyzer (Facade)
  ├─ ValidationPipeline (Chain Manager)
  │   ├─ BackgroundValidator
  │   ├─ FaceDetectionValidator
  │   ├─ CenteringValidator
  │   └─ TiltValidator
  ├─ YuNetFaceDetector (Strategy)
  └─ DebugVisualizer
```

### Adding a New Validator

```python
class CustomValidator(IValidator):
    def validate(self, context: PhotoContext) -> Optional[Result]:
        # Your validation logic
        if error_condition:
            return Result(
                success=False,
                code="custom_error",
                text="Error description",
                metadata=context.metadata
            )
        return None

# Usage
analyzer = PhotoAnalyzer(
    image_path="photo.jpg",
    validators=[
        BackgroundValidator(),
        FaceDetectionValidator(YuNetFaceDetector()),
        CenteringValidator(),
        TiltValidator(),
        CustomValidator(),  # Your validator
    ]
)
```

## Threshold Configuration

Validation thresholds can be configured when creating validators:

```python
analyzer = PhotoAnalyzer(
    image_path="photo.jpg",
    validators=[
        BackgroundValidator(min_white_ratio=0.25),  # 25% instead of 30%
        FaceDetectionValidator(YuNetFaceDetector()),
        CenteringValidator(
            horizontal_threshold=0.15,  # ±15% instead of ±10%
            vertical_threshold=0.30     # ±30% instead of ±25%
        ),
        TiltValidator(max_angle=20.0),  # ±20° instead of ±15°
    ]
)
```

## Project Structure

```
.
├── photo.py              # Main module
├── check_photo.py        # Simplified version (legacy)
├── pyproject.toml        # Dependencies and metadata
├── outputs/              # Debug images (generated)
├── faces/
│   └── img/
│       └── border_black.png  # Reference frame
└── test_imgs/            # Test images
```

## License

MIT
