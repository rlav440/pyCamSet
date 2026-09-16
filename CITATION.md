# Citations and the aruco2 backend

pyCamSet's optional aruco2 marker backend (`marker_backend='aruco2'`) uses the aruco2 library in
the `third_party/aruco2` git submodule. This file reproduces the citations that library and its
upstream give, exactly as they give them, and says how to build the library for pyCamSet.
Licences are listed separately, in `NOTICE`.

## ArUco markers

From `third_party/aruco2/src/aruco2_dictionary.cpp`, lines 457-459, without the comment markers
and the leading "See":

> S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
> "Automatic generation and detection of highly reliable fiducial markers under occlusion".
> Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005

## ArUco Nano

Upstream aruco2's `README.md` (https://github.com/rmsalinas/aruco2, commit `93bfda4`, lines
453-457) states that aruco2's marker detector, which pyCamSet's aruco2 backend calls, is based on
[ArUco Nano](https://www.sciencedirect.com/science/article/pii/S2352711026001822), and cites it as:

> R. Muñoz-Salinas et al., *"ArUco Nano: a simpler, faster, and more reliable fiducial marker
> detector"*, SoftwareX, 2026.

The pinned submodule does not contain this passage: the fork's commit `e902da2` removed it from
its `README.md`. It is reproduced here from upstream.

## ChArUco2

`third_party/aruco2/opencv2/objdetect/aruco2.hpp` attributes its grid board and diamond designs to
the citation key `MunozSalinas2026ChArUco2`. The submodule contains no bibliography entry for that
key; its `README.md` links the paper as:

- [ChArUco2](https://www.sciencedirect.com/science/article/pii/S2352711026003249)

No authors, title or journal are given here, because the submodule gives none.

## Cited by key only

`aruco2.hpp` also gives these citation keys, for parts of aruco2 that pyCamSet's aruco2 backend
uses:

- `Aruco2014`, `romero2018speeded` and `GARRIDOJURADO2026102690`, for `detectFiducialMarkers`,
  the marker detector the backend calls;
- `garrido2016generation`, for the `DICT_ARUCO_MIP_36h12` dictionary, which the backend offers.

The submodule contains no bibliography file that resolves these keys, so they are listed by key
only.

## Not reproduced here

The submodule also cites work behind features pyCamSet does not use, for example fractal markers
(`third_party/aruco2/src/aruco2_fractal.cpp`, lines 47-50) and RArUco markers
(`third_party/aruco2/README.md`, lines 519-520). Those citations are not reproduced.

## Installing the aruco2 backend

aruco2 is not published on PyPI. `pyproject.toml` declares no `aruco2` extra: it is not on PyPI,
so there is nothing to pin. `pip install pyCamSet[aruco2]` fails with an unknown-extra error.
Build the wrapper from the submodule instead, into the environment pyCamSet runs in:

1. Fetch the source, if the checkout does not have it yet:
   `git submodule update --init third_party/aruco2`.
2. Follow the Build section of
   [`third_party/aruco2/python/README.md`](third_party/aruco2/python/README.md#build). Its
   commands run from the aruco2 repository root, which here is `third_party/aruco2`. It needs a
   native OpenCV development package that CMake can find (pass `OpenCV_DIR` if it is in a
   non-standard location).

### What has been tested

The "Platform status" section of `third_party/aruco2/python/README.md` states that the cp311
Windows wheel is tested, and that the CMake project is structured for Linux and macOS but
non-Windows builds are not run by its release check. The wrapper's `pyproject.toml` declares
`requires-python = ">=3.10"`; pyCamSet requires Python 3.11 or newer. Builds for any other Python
version or platform are untested.
