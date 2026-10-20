# Architecture

How photo-checker is put together.

Everything is in one file, `photo.py`, 498 lines. The structure inside it is
a chain of responsibility over a shared context, with a strategy for the
face detector behind a one-method interface.

## The shape

| Module | What it is responsible for |
| --- | --- |
| `ImageSize`, `Metadata`, `Result`, `FaceData`, `PhotoContext` | The pydantic models. `PhotoContext` is what the validators share: the decoded image, its size, the face that was found and the metadata being filled in as the run goes. |
| `IFaceDetector`, `YuNetFaceDetector` | Finding one face and its five landmarks. The detector behind the interface. |
| `IValidator`, `BackgroundValidator`, `FaceDetectionValidator`, `CenteringValidator`, `TiltValidator` | The four rules. Each returns `None` to pass, or a `Result` to fail. |
| `ValidationPipeline` | Runs the validators in order and stops at the first failure. |
| `DebugVisualizer` | Draws the failure into a picture. |
| `PhotoAnalyzer` | The facade. Reads the file, builds the pipeline, runs it and writes the picture. |
| the `__main__` block | The command line: one argument, one JSON object on stdout. |

## The data flow

```
photo.py <path>
  cv2.imread -> PhotoContext
  ValidationPipeline.execute(context)
    BackgroundValidator     -> background_white_ratio_percent
    FaceDetectionValidator  -> faces_count, face_data
    CenteringValidator      -> centering_horizontal_percent, centering_vertical_percent
    TiltValidator           -> eye_angle_degrees, pitch_angle_degrees
  DebugVisualizer           -> outputs/cn_<name>
  model_dump_json(indent=2, exclude_unset=True)
```

The metadata is filled in as the run goes, and the chain stops at the first
failure. A photograph with a dark background never reaches the face
detector, so its result carries a white ratio and nothing else.

The centring check uses the nose landmark, index 2 of the five YuNet
returns. The tilt check uses the two eyes, index 0 and index 1, for the roll
and the nose against their midpoint for the pitch.

## Decisions worth knowing about

### A validator returns a Result or None

A validator returns `None` when the photograph passes and a `Result` when it
fails. The pipeline asks for the first one that is not `None` and returns it.

The cost: a validator with nothing to say also returns `None`.
`CenteringValidator` and `TiltValidator` return `None` when
`context.face_data` is `None`, so putting either of them ahead of the face
detector turns the check into a silent pass.

### The detector sits behind an interface

`IFaceDetector` has one method, `detect(image, image_size)`, and YuNet
implements it. A second detector is a class rather than a patch of the
pipeline.

The cost: `YuNetFaceDetector.detect` raises `ValueError` when it finds more
than one face, and `FaceDetectionValidator` catches that exception to turn
it into the `multiple_faces` code. The exception is the seam between the
two.

### The order of the chain is fixed

Background, face, centring, tilt. The first failure is the answer, and the
metadata carries only what was measured before it.

The cost: the order is a contract with whoever reads the result. A
photograph that breaks two rules reports the first one only.

### The thresholds are constructor arguments

Every default is an argument: `min_white_ratio`, `horizontal_threshold`,
`vertical_threshold`, `max_angle`, the model path and the frame path. The
defaults are the rules the module was written to.

The cost: `PhotoAnalyzer` builds a `YuNetFaceDetector` in `__init__` before
it looks at the validator list, so a custom list does not spare the
download of the model.

### The result is printed with exclude_unset

`result.model_dump_json(indent=2, exclude_unset=True)` leaves out every
field that was never set, so a failed run carries the measurements it took
and nothing more. Every metadata field is optional to a consumer.

### The exit code is 0

A photograph that fails a rule is still a successful run of the program.
The `success` field is the verdict.

The cost: a CI step that tests `$?` passes a photograph that failed.

## Boundaries

- `YuNetFaceDetector` knows about OpenCV and the model file. It knows
  nothing about the validators.
- The validators know about `PhotoContext` and about the numbers they
  measure. They know nothing about each other, and a validator never calls
  another one.
- `ValidationPipeline` knows the order. It knows nothing about what a
  validator measures.
- `DebugVisualizer` knows about the frame and the marks. It reads the
  context and does not change it.
- `PhotoAnalyzer` is the only class that knows all four.
- The `__main__` block is the only place that prints, with one exception:
  the detector prints the model download notice on the first run, and that
  notice lands in front of the JSON.
