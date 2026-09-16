# Third-party code

## aruco2

`third_party/aruco2` is a git submodule holding the aruco2 library and its Python wrapper. Once
built and installed, the wrapper is the `aruco2` package pyCamSet imports when
`marker_backend='aruco2'` is chosen. The submodule is not part of pyCamSet's source distribution
or wheel: `pyproject.toml` excludes `third_party` from package discovery.

- **Submodule URL:** `https://github.com/ColDSnit/aruco2.git`, pinned at commit `1091fb9`
  ("Restore upstream's credit to ArUco Nano for the marker detector"). This fork's four commits
  on top of upstream commit `8755035` add the Python wrapper (`python/`, a root `pyproject.toml`,
  and changes to `CMakeLists.txt`, `README.md` and `.gitignore`), add input checks to
  `src/aruco2.cpp` and `src/aruco2_fractal.cpp`, fix an image-edge bounds check in the latter,
  and restore the README's credit to ArUco Nano as upstream words it. Upstream's `main` has moved
  on since `8755035` (it was at `93bfda4` on 2026-09-16).
- **Upstream:** `https://github.com/rmsalinas/aruco2.git`, the repository of R. Muñoz-Salinas
  (GitHub: rmsalinas). The submodule does not track it; it is named here for provenance.
- **Fetching it:** `git submodule update --init third_party/aruco2`, or clone pyCamSet with
  `git clone --recurse-submodules`.
- **Licences:** listed in the top-level `NOTICE`, including the one file under
  `third_party/aruco2` that is not Apache-2.0 (`opencl/aruco2detect.cl`, MIT).
- **Building it, what has been tested, and citations:** see the top-level `CITATION.md`.
