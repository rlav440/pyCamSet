# Recording stack fixtures

Three tiny recording stacks written by the panoramic acquisition software's own
writer (`StackWriter` in `utils/recording_stack.py` of `software-panoramic-refactored`,
commit `15fff89d9f91c0f0770d5de5c1d5e109f8c9b203`), for `tests/test_recording_stack.py`:

| Folder | Frames | Pixels | Usable pages |
|---|---|---|---|
| `cam1/` | 4, 6 x 4 | uint16, Mono12 shifted left by 4 (MSB-aligned) | 0-3 |
| `cam2/` | 3, 6 x 4 | uint16, Mono12 shifted left by 4; page 0 flagged as foreign | 1-2 |
| `cam3/` | 3, 6 x 4 | uint8, Mono8 | 0-2 |

`<stem>_expected.npy` beside each stack holds the exact frames written, stacked.
The pixels are random (seeded), not images of a target.

To regenerate them, with that commit's `utils/recording_stack.py`:

```bash
python make_fixtures.py <path to its recording_stack.py> <output folder>
```
