import sys, datetime, numpy, importlib.util, pathlib, shutil
spec = importlib.util.spec_from_file_location("recording_stack", sys.argv[1])
rs = importlib.util.module_from_spec(spec); sys.modules[spec.name] = rs; spec.loader.exec_module(rs)
out = pathlib.Path(sys.argv[2]); shutil.rmtree(out, ignore_errors=True)
t0 = datetime.datetime(2026, 10, 9, 14, 30, 15, tzinfo=datetime.timezone.utc)
def frames(n, shape, dtype, shift):
    rng = numpy.random.default_rng(7)
    hi = (1 << (16 - shift)) - 1 if dtype == "uint16" else 255
    return [(rng.integers(0, hi, size=shape, dtype=numpy.uint32) << shift).astype(dtype) for _ in range(n)]
def make(cam, stem, n, dtype, shift, flagged=False, close=True):
    d = out / cam
    w = rs.StackWriter.create(str(d), stem, (6, 4), dtype, camera_name=cam, serial="0",
                              bit_depth=12 if dtype == "uint16" else 8, shift=shift,
                              pixel_format="Mono12" if dtype == "uint16" else "Mono8",
                              started_at=t0, fsync_interval_s=None,
                              extra={"first_frame": {"status": "foreign" if flagged else "ok", "flagged": flagged,
                                                     "flagged_page": 0 if flagged else None}})
    fs = frames(n, (6, 4), dtype, shift)
    for i, f in enumerate(fs):
        w.write_frame(f, block_id=i, timestamp=1000 * (i + 1))
    if close:
        w.close()
    else:
        w._idx.close() if hasattr(w, "_idx") else None
    numpy.save(d / (stem + "_expected.npy"), numpy.stack(fs))
make("cam1", "cam1_20261009_143015", 4, "uint16", 4)
make("cam2", "cam2_20261009_143015", 3, "uint16", 4, flagged=True)
make("cam3", "cam3_20261009_143015", 3, "uint8", 0)
print(sorted(p.relative_to(out).as_posix() + f" {p.stat().st_size}" for p in out.rglob("*") if p.is_file()))
