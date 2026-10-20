---
name: photo-checker
description: The marks a rejected photograph carries. There is no interface beyond the JSON verdict and the debug image.
colors:
  landmark: "#FF0000"
  eye-line: "#00FF00"
  nose: "#00FFFF"
  centre-cross: "#FFFF00"
  frame: "#000000"
typography:
  verdict:
    fontFamily: "not set by the code"
    fontSize: "not set by the code"
    indent: "2 spaces"
rounded:
  landmark: "8px"
  nose: "10px"
spacing:
  cross-arm: "40px"
  eye-line-width: "3px"
  cross-width: "2px"
---

# Design system: photo-checker

## Overview

**Creative North Star: "The marked photograph"**

photo-checker has no interface of its own. There is no window, no page, no
menu and no type on a screen. What the product puts in front of a person is
a JSON object on stdout and, when a rule fails, one picture written into
`outputs/`.

That picture is the whole visual output, and the code draws it with a fixed
set of marks: five red dots for the landmarks, a green line between the eyes,
a cyan dot for the nose and a yellow cross at the centre of the frame. The
photograph sits under them at half opacity, over a black ground.

Every decision in this document concerns one diagram. They are written down so
that a change to the marks is a deliberate change rather than a drift.

Key characteristics:

- One picture, drawn in one place, `DebugVisualizer.create_debug_image`.
- Four colours, one for each mark, and no colour for anything else.
- The geometry lives in the code: radius 8 for a landmark, radius 10 for the
  nose, a 3px eye line, a 2px cross with 40px arms.
- No type, no spacing scale and no radius ladder. The output is a
  photograph with marks on it.

## Colors

Four colours, one for each mark, and the black of the frame underneath.

### Primary

- **Landmark** (`#FF0000`): the five points of the face, drawn as filled
  ellipses of radius 8. The right eye, the left eye, the nose tip and the
  two corners of the mouth.
- **Eye line** (`#00FF00`): the segment between the two eye landmarks, 3px
  wide. It is the roll, drawn, so a tilted head is visible at a glance.
- **Nose** (`#00FFFF`): the third landmark, drawn as a filled ellipse of
  radius 10, so it reads over the red dot underneath it.
- **Centre cross** (`#FFFF00`): two strokes of 2px and 40px each, one
  horizontal and one vertical, at the centre of the frame. It is the target
  the nose is measured against.

### Neutral

- **Frame** (`#000000`): the 1000 by 1000 fallback picture when
  `faces/img/border_black.png` is missing, and the ground the photograph is
  composited onto. The reference frame in a full checkout is a black
  picture with a border, so the photograph sits inside it.

### Named rules

**The One Photograph Rule.** The photograph is the picture. Every mark is
drawn over it at half opacity, never over an opaque panel, so the person
looks at their own face and not at a diagram of it.

**The Shape Carries Meaning Rule.** The marks differ in shape as well as in
colour: a dot for a point, a line for a line, a cross for a target. Colour
is the second signal.

## Typography

**Display Font:** none
**Body Font:** none

The product sets no font and no size. The only typographic decision in the
code is the two-space indent of the result, `model_dump_json(indent=2)`. The
JSON goes to stdout and whatever renders it decides the rest.

The `text` field of a failure is a Russian sentence inside that JSON. It
carries no weight, no size and no language of its own; it inherits the
terminal's.

## Layout

One picture, one size: the frame. When `faces/img/border_black.png` is
missing, the frame is 1000 by 1000 pixels of black. When the file is there,
the frame's own size decides everything and the photograph is resized to it.

The photograph is stretched to the frame, not letterboxed. `cv2.resize` takes
the width and the height and ignores the aspect ratio, so a 1152 by 560
photograph comes out square. The landmarks are scaled by the same two
factors, so they stay on top of the face they came from.

There is no margin and no padding around the photograph inside the frame.
The reference picture carries the border.

## Elevation & Depth

There is no depth in the product. The frame is a flat black ground, the
photograph sits on it at 50% opacity, and the marks are drawn on top at full
opacity.

The opacity is the only layering there is. It lets the border of the
reference frame show through the photograph, so the person can see where the
frame ends.

## Shapes

The marks are the shapes. Two ellipses and three lines:

| Mark | Shape | Size |
| --- | --- | --- |
| Landmark | filled ellipse | radius 8 |
| Nose | filled ellipse | radius 10 |
| Eye line | stroke | 3px |
| Centre cross | two strokes | 2px each, 40px arms |

Nothing has a corner and nothing is clipped. The frame is a rectangle
because a picture is a rectangle.

## Components

### The debug image

- **Shape:** the frame's rectangle, with the photograph stretched across it.
- **Ground:** black, or the reference picture when it is in the checkout.
- **Photograph:** at alpha 128, so the frame shows through.
- **Marks:** the five landmarks, the eye line, the nose and the cross, in
  the four colours above.
- **File:** a JPEG, written to `outputs/cn_<original name>` with the
  original extension. The result carries the path in `debug_image_path`.

### The result

- **Shape:** JSON, printed with `indent=2` and `exclude_unset=True`.
- **Fields:** `success`, then `code` and `text` when a rule failed, then
  `metadata` with the numbers, then `debug_image_path` when a picture was
  written.
- **Order:** the keys keep the order the model declares them in, so a reader
  of the raw output sees the verdict before the measurements.

## Do's and Don'ts

### Do:

- **Do** keep the four marks in the four colours: red for the landmarks,
  green for the eye line, cyan for the nose and yellow for the cross.
- **Do** keep the geometry as it is: radius 8, radius 10, 3px, 2px, 40px.
- **Do** write the debug image as a JPEG into `outputs/`, with the `cn_`
  prefix and the original extension.

### Don't:

- **Don't** add a colour, a font or a shape to the debug image. A fifth
  mark has to mean a fifth measurement, and there are four measurements.
- **Don't** put the marks on a panel or under a caption. The photograph is
  the picture.
- **Don't** draw the photograph opaque over the frame. The border is the
  only thing that shows where the picture ends.
- **Don't** rely on the colour alone. Red against green is the pair that a
  reader with deuteranopia tells apart least well, and the picture carries
  no legend.
