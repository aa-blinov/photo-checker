# photo-checker

Check a portrait photograph against four rules and get the answer as JSON: the
background is white, exactly one face is found, the nose is near the middle,
and the head is not tilted. When a rule fails, the run also writes a picture
of the detected landmarks, so a rejection can be looked at instead of guessed
at.

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Quick start

```bash
uv sync
uv run photo.py path/to/photo.jpg
```

Python 3.10 or newer. In a pip environment, `pip install -e .` pulls in the
same four dependencies; pip refuses the install on an older interpreter
because the manifest sets the floor at 3.10.

The first run downloads the YuNet face detector, 232,589 bytes, into the
current directory and prints a line about the download. Every run after that
prints the result and nothing else.

The result is one JSON object on stdout. A photo that passes every rule:

```json
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
```

A photo that fails:

```json
{
  "success": false,
  "code": "background_not_white",
  "text": "Фон не белый",
  "metadata": {
    "image_size": {
      "width_px": 640,
      "height_px": 640
    },
    "background_white_ratio_percent": 0.0
  },
  "debug_image_path": "outputs/cn_dark.jpg"
}
```

The exit code is 0 in both cases. The `success` field is the verdict.

## What it does

- **Measures the white background.** Pixels with saturation below 30 and
  brightness above 200 count as white, and at least 30% of the image has to be
  white. A photo below that gets the code `background_not_white`.
- **Finds the face.** The YuNet detector returns the five landmarks of a face:
  the eyes, the nose and the corners of the mouth. No face is
  `face_not_recognized`, two faces are `multiple_faces`.
- **Checks that the face is centred.** The nose landmark has to sit within 10%
  of the half-width and 25% of the half-height of the image centre. The four
  failures are told apart: `centering_left`, `centering_right`,
  `centering_up`, `centering_down`.
- **Checks that the head is not tilted.** The angle of the line between the
  eyes gives the roll, and the angle of the nose against the midpoint of the
  eyes gives the pitch. Both have to stay within 15 degrees:
  `tilt_left`, `tilt_right`, `tilt_up`, `tilt_down`.
- **Draws the failure.** When a check fails, the photo is scaled into the
  reference frame and written to `outputs/cn_<original filename>`, with red dots
  on the five landmarks, a green line between the eyes, a cyan nose and a
  yellow cross at the centre of the frame. The frame is 1000 by 1000 when the
  frame file is missing, which is the case in a fresh checkout.
- **Reports the measurements.** Every result carries the numbers behind the
  verdict: the white ratio, the face count, the two centring offsets, the roll
  and the pitch. Only the measurements taken before the first failure are
  there.

## How it works

```
photo.py <path>
  PhotoAnalyzer.analyze()
    cv2.imread the file
    ValidationPipeline.execute(PhotoContext)
      BackgroundValidator      background_white_ratio_percent
      FaceDetectionValidator   faces_count, face_data
      CenteringValidator       centering_horizontal_percent, centering_vertical_percent
      TiltValidator            eye_angle_degrees, pitch_angle_degrees
    DebugVisualizer            only when a validator failed
```

A validator returns `None` to pass, or a `Result` to fail. The pipeline runs
them in the order above, stops at the first failure and returns it. A photo
with a dark background never reaches the face detector, which is why its
result carries no face count.

`PhotoContext` is what the validators share: the decoded image, its size, the
face that was found and the metadata being filled in as the run goes.

The centring check uses the nose landmark, index 2 of the five YuNet returns.
The tilt check uses the two eyes, index 0 and index 1, for the roll and the
nose against their midpoint for the pitch.

## Configuration

The thresholds are constructor arguments. The defaults are the rules the
module was written to, and a different set of rules is a different validator.

| Option | Default | Where it is set |
| --- | --- | --- |
| `min_white_ratio` | `0.30` | `BackgroundValidator(min_white_ratio=...)` |
| `horizontal_threshold` | `0.10` | `CenteringValidator(horizontal_threshold=...)` |
| `vertical_threshold` | `0.25` | `CenteringValidator(vertical_threshold=...)` |
| `max_angle` | `15.0` | `TiltValidator(max_angle=...)` |
| model path | `face_detection_yunet_2023mar.onnx` | `YuNetFaceDetector(model_path=...)` |
| frame path | `faces/img/border_black.png` | `DebugVisualizer(border_path=...)` |

Two paths are relative to the working directory. The model is downloaded there
when it is missing. The frame is the reference picture the debug image is drawn
over; when it is missing, a black 1000 by 1000 frame is used instead.

The pipeline order and the detector are arguments of `PhotoAnalyzer` as well, so
a caller that wants a fifth rule passes the whole list:

```python
analyzer = PhotoAnalyzer(
    image_path="photo.jpg",
    validators=[
        BackgroundValidator(),
        FaceDetectionValidator(YuNetFaceDetector()),
        CenteringValidator(horizontal_threshold=0.15),
        TiltValidator(max_angle=20.0),
    ],
)
```

## Development

```bash
uv sync                            # install the runtime dependencies
uv run photo.py <photo>            # check one photo
uv tool run ruff check photo.py    # lint, with the settings in pyproject.toml
```

ruff and black are configured in `pyproject.toml` with a line length of 100,
but neither is a declared dependency, so `uv sync` does not install them.
`uv tool run ruff check photo.py` runs the lint in an isolated environment.

There is no test suite and no build step. `photo.py` is a script that runs
from the repository root.

| Path | What it holds |
| --- | --- |
| `photo.py` | The whole module: the models, the detector, the four validators, the pipeline, the visualizer and the command line at the bottom of the file. |
| `pyproject.toml` | The dependencies, the Python floor, the ruff and black settings. |
| `uv.lock` | The resolved versions: numpy 1.26.4, opencv-python 4.11.0.86, pillow 12.0.0, pydantic 2.12.5. |
| `outputs/` | The debug images a failed run writes. Created on demand, not tracked. |

## License

MIT, © Alexander Blinov. See [LICENSE](LICENSE).
