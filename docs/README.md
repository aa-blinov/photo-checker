# Documentation

The documents of photo-checker. The module is one file, so the set is short.

| Document | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | The classes in `photo.py`, the order the four checks run in, and the boundaries between them. |
| [screenshots/README.md](screenshots/README.md) | The images the product makes. The repository holds none. |
| [../README.md](../README.md) | The front door: what it does and how to run it. |
| [../AGENTS.md](../AGENTS.md) | The commands, and the traps that cost a rebuild to learn. |
| [../PRODUCT.md](../PRODUCT.md) | The product truth: the users, the purpose, the constraints. |
| [../DESIGN.md](../DESIGN.md) | The colours and the geometry of the debug image. |

## Reading order

New here: [../README.md](../README.md), then
[architecture.md](architecture.md). The README says what the four rules are;
the architecture says how they are put together.

Changing the code: [architecture.md](architecture.md) for the shape, then
[../AGENTS.md](../AGENTS.md) for the traps. The thresholds live in both
places, and a threshold is the owner's decision.
