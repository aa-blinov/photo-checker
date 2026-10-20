# AGENTS.md — agent context for `photo-checker`

A single Python module that checks a portrait photograph against four rules
and prints the verdict as JSON. There is no test suite, no CI on `master`, and
no interface beyond the stdout of one command and the debug images a failed run
writes into `outputs/`.

This file is what you need to work on the project without reading all of it.

## Run / build / test

```bash
uv sync                                 # install the runtime dependencies
uv run photo.py <photo>                 # check one photo, the JSON goes to stdout
uv tool run ruff check photo.py         # lint, with the settings in pyproject.toml
uv tool run ruff format --diff photo.py # formatting check
```

There is nothing to build and nothing to test. `photo.py` is a script that
runs from the repository root, and the manifest declares a setuptools build so
that the four dependencies can be installed, not so that a package can be
imported.

Python 3.10 or newer. The locked versions are numpy 1.26.4,
opencv-python 4.11.0.86, pillow 12.0.0 and pydantic 2.12.5.

## Where things live

| Path | What it holds |
| --- | --- |
| `photo.py` | Everything. The pydantic models, the detector behind its interface, the four validators, the pipeline, the visualizer, and the command line at the bottom of the file. 498 lines. |
| `pyproject.toml` | The dependency list, the Python floor, the ruff and black settings. |
| `uv.lock` | The resolved versions. |
| `outputs/` | The debug images. Created by a failed run, gitignored. |
| `faces/img/border_black.png` | The reference frame. Not in the repository, `faces/` is gitignored. |
| `face_detection_yunet_2023mar.onnx` | The detector model. Downloaded on the first run, gitignored. |

## Traps

- **The exit code is 0 for a failed check.** `analyze()` returns the result and
  the script prints it; nothing sets a non-zero status when a rule fails. A CI
  step that tests `$?` passes a photograph that failed a rule. The `success`
  field is the only verdict.
- **The first run prints the model download to stdout.** The constructor of
  `YuNetFaceDetector` prints "Скачивание модели YuNet..." before the JSON, so
  `uv run photo.py photo.jpg | jq .success` fails on a fresh clone: the first
  line of the stream is not JSON. Run the tool once before piping it, or read
  the JSON from the end of the output.
- **The model lands in the working directory.** The path in the constructor is
  relative, so running from another directory downloads a second copy there
  and leaves the first one behind. The file is 232,589 bytes.
- **The reference frame is not in the repository.** `faces/` is gitignored, so
  every debug image a fresh clone produces is the black 1000 by 1000
  fallback, and the real frame has to be put back by hand.
- **The photo is stretched into the frame, not letterboxed.**
  `cv2.resize(context.image, (border_w, border_h))` ignores the aspect ratio,
  so a 1152 by 560 photograph comes out square and distorted. The landmark
  markers are scaled with the same two factors and stay in place.
- **The metadata only carries what was measured before the failure.** The
  result is dumped with `exclude_unset=True` and the chain stops at the first
  failure, so a `background_not_white` result has the white ratio and nothing
  else. Every metadata field is optional to a consumer.
- **The chain order decides which error you get.** A validator that returns
  `None` passes. `CenteringValidator` and `TiltValidator` return `None` when
  `context.face_data` is `None`, so putting either of them ahead of the face
  detector turns the check into a silent pass.
- **`PhotoAnalyzer` builds a `YuNetFaceDetector` even when you pass your own
  validators.** The detector is constructed in `__init__` before the validator
  list is looked at, and its constructor downloads the model. A custom
  validator list does not avoid the download.
- **The install does not make `photo` importable.** The manifest declares
  `[tool.setuptools.packages.find] where = ["."]` and no `py-modules`, so
  setuptools finds no package. `import photo` raises `ModuleNotFoundError`
  after `uv sync`.
- **The `text` field is Russian; the `code` field is the contract.** So are the
  usage line and the "file not found" line. A caller that shows `text` to an
  English reader shows Russian.
- **The detector's score threshold is OpenCV's default.**
  `cv2.FaceDetectorYN.create` is called with the model path and the image size
  only. A small, blurred or turned-away face is reported as no face at all.

## Style / safety rails

- **The thresholds are the product.** 30% white, plus or minus 10% and 25% on
  the centring, plus or minus 15 degrees on the roll and the pitch. Changing a
  default changes which photographs pass, so a threshold change is the
  owner's decision and not part of a refactor.
- **The chain order is fixed**: background, face, centring, tilt.
- **The `code` strings are the public interface.** `background_not_white`,
  `face_not_recognized`, `multiple_faces`, the four `centering_` codes and the
  four `tilt_` codes, plus `load_error`. The `text` beside them is a message
  and may be translated.
- **stdout carries the JSON and nothing else.** The model download notice is
  the one exception, and it is a fault. Any new notice goes to stderr.
- **The identifiers are English, the comments and docstrings are Russian.**
  Keep one language per layer, the way the repository does it now.
- **Four dependencies, and the four imports.** `cv2`, `numpy`, `PIL` and
  `pydantic`. A new import needs a new dependency in `pyproject.toml` and a
  lock update.
- **`ruff check photo.py` has to pass.** It reports one finding today: an
  unused exception variable at `photo.py:160`. `ruff format --diff photo.py`
  is already clean.
- **The name is not settled.** The repository and the distribution are
  `photo-checker`, the module is `photo.py`, the first README called the tool
  Photo Validator, and the GitHub description is empty. Pick one before the
  README is published.

## Where the product truth lives

`PRODUCT.md` holds it. Read it before changing the behaviour of the module,
and write the change there when the behaviour changes.
