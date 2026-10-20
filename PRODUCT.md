# Product

<!-- impeccable:product-schema 1 -->

## Platform

cli

The product is a Python module and a command line script, version 0.1.0,
running on CPython 3.10 or newer. The interface is the JSON object that comes
out of stdout and the debug image that a failed run writes. There is no server
and no database.

## Users

The code does not name its user. The GitHub description is empty, the README
never mentions an audience, and the repository has no stars, no forks and no
issue other than the pull request that adopts the house standard. Who runs it
is an open question.

What the code does say is what the job is. The four rules are the ones a
photographer applies to a portrait of a person: a white background, one face,
the face in the middle, the head straight. The numbers attached to those
rules, 30% white, plus or minus 10% and 25% on the nose, plus or minus 15
degrees on the roll and the pitch, are the working numbers of a photo studio
or a document photo service. A batch that shoots thousands of portraits and
needs each one judged the same way before it goes out is the situation the
module is built for. That reading comes from the rules; the repository states
it nowhere.

The one confirmed user is the author, Alexander Blinov.

## Product Purpose

The module answers one question about a file on disk: does this photograph
meet the rules?

The answer is a JSON object with two halves. `success` says whether the
photograph passed. `code` names the first rule that failed, in a stable
English key, and `metadata` carries the numbers that were measured before
that failure. A second half exists for the person: when a rule fails, the
module draws the five face landmarks onto the photo and writes the picture to
`outputs/`.

Success for the user is a photograph that passes without a person looking at
it, and a rejection that can be shown to the person whose photograph it is.
Success for the product is that the result can be read by a program. The
`code` and the metadata are the contract; the Russian sentence next to the
code is a message for a person.

## Positioning

The nearest neighbour is a script that measures a photograph and prints a
number. Two things here are harder to copy.

The first is that a rejection is drawn. Most photo checks answer with a
threshold and leave the person to work out what the threshold meant. This one
writes the landmarks onto the photo and marks the centre of the frame, so the
reason for a rejection is a picture.

The second is that the four rules are not welded to the module. The detector
sits behind an interface with one method, the validators take their
thresholds as constructor arguments, and the pipeline takes a list. A second
rule set is a second `PhotoAnalyzer` built in the calling code, with no fork
of `photo.py`.

The repository holds no comparison with another tool and no measurement of
this one against any, so nothing here says how it stands next to a
neighbour.

## Operating Context

The photograph is a file on disk and the run is per file: one command, one
JSON object on stdout. There is no queue, no server and no state carried
between runs.

Three things happen around a run that a caller has to know about.

The first run downloads the YuNet detector, 232,589 bytes, from the
opencv_zoo repository into the working directory, and prints a line about the
download in front of the JSON.

The first run on a photograph that fails writes an image into `outputs/`. The
directory is created on demand and is gitignored, so it grows with no cleanup
step.

The working directory is part of the contract. The model path, the frame at
`faces/img/border_black.png` and the `outputs/` directory are all relative to
it.

## Capabilities and Constraints

Confirmed by the code:

- Four checks in a fixed order: background, face, centring, tilt.
- A pixel is white when its saturation is below 30 and its brightness is above
  200; at least 30% of the image has to be white.
- Exactly one face is required. Zero is `face_not_recognized`, two or more
  are `multiple_faces`.
- The nose landmark has to sit within 10% of the half-width and 25% of the
  half-height of the image centre.
- The roll and the pitch each have to stay within 15 degrees.
- The chain stops at the first failure, and the result carries only the
  measurements taken before it.
- A failed run writes a picture the size of the reference frame, with the
  landmarks, the eye line, the nose and the centre cross drawn onto it.
- The thresholds are constructor arguments, and the pipeline and the detector
  are arguments of the facade.
- The output is one JSON object on stdout, printed with `exclude_unset=True`.

Deliberately not true, and worth knowing before a change:

- The exit code is 0 for a failed check.
- The messages in `text`, the usage line and the file-not-found line are
  Russian. The `code` values are English.
- `photo.py` is not importable after an install; the manifest declares no
  package and no module.
- The frame and the sample photographs are not in the repository, because
  `faces/` and `test_imgs/` are gitignored.
- There is no test suite, and no CI on `master`.
- The detector's score threshold is OpenCV's default, and the pitch angle is
  a two-dimensional estimate taken from the angle of the nose against the
  midpoint of the eyes.

Not decided, and to be answered by the owner:

- Who the user is, and where a run happens in production: on a workstation, in
  a container, or in a batch job.
- Whether the four thresholds come from a published specification or from the
  author's own practice.
- Whether the JSON result is consumed by one service or by several.

## Brand Commitments

**Name:** the repository and the distribution are `photo-checker`, the module
is `photo.py`. The first README called the tool Photo Validator, and the
GitHub description is empty, so the name has never been settled in one place.
**Voice:** the code speaks in English identifiers and Russian sentences. The
message a user reads is Russian.

There is no logo, no palette and no site. The only picture the product makes
is the debug image, and its colours are fixed in the code: red landmarks, a
green line between the eyes, a cyan nose and a yellow cross at the centre, on
a black frame.

## Evidence on Hand

- `photo.py` holds every rule, every threshold, every error code and the
  command line. It is 498 lines.
- `pyproject.toml` holds the dependencies (opencv-python, numpy, pydantic,
  Pillow), the Python floor, and the ruff and black settings.
- `uv.lock` holds the resolved versions: numpy 1.26.4, opencv-python
  4.11.0.86, pillow 12.0.0, pydantic 2.12.5, pydantic-core 2.41.5.
- The first README published two real result objects, one successful and one
  rejected.
- The repository has two commits, both on 2025-12-01. The second one
  reorganised the imports, simplified the `CenteringValidator` constructor and
  streamlined the image conversion in `DebugVisualizer`.

Absent, and not to be invented: no usage numbers, no testimonials, no
measurement of how long a run takes, no specification behind the thresholds,
no release and no tag, no CI, no test suite. The GitHub description is empty,
and the repository has no stars and no forks, so there is nothing to count.

## Product Principles

1. **The verdict is a code, not a sentence.** `code` is stable and readable by
   a program; `text` is a message and may be translated. Changing a code is a
   change to the interface.
2. **The first failure is the answer.** The chain stops at the first rule that
   fails, and the metadata carries only what was measured before it, so a
   rejection names the thing to fix first.
3. **The thresholds are arguments.** Every default is a constructor argument,
   so a different rule set is another instance of the same module rather than
   a fork of it.
4. **A rejection is shown, not stated.** The debug image exists so that the
   reason can be looked at by the person the photograph belongs to.
5. **One detector, behind an interface.** YuNet is the default. The interface
   has a single method, so a different detector is a class and not a patch.

## Accessibility & Inclusion

The product judges photographs of people, and the rules it applies are the
ones a photo studio applies to a portrait. The repository does not say who
those rules are for, and the question matters: a photograph taken by someone
who cannot hold their head straight, or who is photographed from a wheelchair,
can fail the roll and the centring checks with nothing wrong with the
photograph. There is no accommodation in the code, and no statement that one
is needed.

Two things in the output are worth naming.

The debug image separates the landmarks from the eye line by colour alone:
red dots and a green line on a photograph, with a yellow cross for the
centre. Red against green is the pair that a reader with deuteranopia tells
apart least well. The marks differ in shape as well, so the line still reads
as a line, but the picture carries no legend.

The messages are Russian only, so a person reading `text` inside an English
interface gets a Russian sentence.

Nothing else has been decided. The module has no interface of its own to
make accessible, and no standard has been set for the photographs it accepts.
