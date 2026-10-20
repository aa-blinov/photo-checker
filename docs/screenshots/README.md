# Screenshots

The repository holds no images.

photo-checker makes one picture, the debug image a failed run writes into
`outputs/cn_<original name>`. It is a product of the run rather than a
documented state of the interface, and it is not tracked: `outputs/` is
created on demand and `.gitignore` keeps it out of the repository.

Two more files sit outside the repository for the same reason. The YuNet
model, `face_detection_yunet_2023mar.onnx`, is downloaded into the working
directory on the first run. The reference frame,
`faces/img/border_black.png`, is the picture the debug image is drawn over,
and `faces/` is gitignored, so a fresh checkout draws over a black 1000 by
1000 fallback instead.

The folder holds this file and nothing else. When a picture is worth
keeping, put it here and point at it from the README with a relative path.
