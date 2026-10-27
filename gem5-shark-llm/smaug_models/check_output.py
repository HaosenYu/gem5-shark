"""Compare SMAUG's output.pb against the NumPy reference saved by the builder."""
import sys

import numpy as np
from smaug.core import tensor_pb2


def load_output(path):
    t = tensor_pb2.TensorProto()
    with open(path, "rb") as f:
        t.ParseFromString(f.read())
    if len(t.data.half_data):
        # Packed two fp16 values per int32, as in smaug/python/smaug_test.py.
        flat = np.array(t.data.half_data, dtype=np.int32) \
            .view(np.float16).astype(np.float32)
    else:
        flat = np.array(t.data.float_data, dtype=np.float32)
    dims = list(t.shape.dims)
    if flat.size != int(np.prod(dims)):          # strip alignment padding
        padded = dims[:-1] + [flat.size // int(np.prod(dims[:-1]))]
        flat = flat.reshape(padded)[..., :dims[-1]]
    return flat.reshape(dims)


def main(outdir):
    ref = np.load(outdir + "/ref_output.npy")
    out = load_output(outdir + "/output.pb")
    err = np.abs(out - ref)
    rel = err.max() / (np.abs(ref).max() + 1e-12)
    print("shape smaug=%s ref=%s" % (out.shape, ref.shape))
    print("max|ref|=%.4f  max abs err=%.5f  mean abs err=%.5f  rel=%.4f"
          % (np.abs(ref).max(), err.max(), err.mean(), rel))
    print("PASS" if rel < 2e-2 else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1])
