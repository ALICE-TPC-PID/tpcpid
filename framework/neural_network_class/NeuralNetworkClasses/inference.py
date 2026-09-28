"""Bounded ONNX inference for training QA."""
import numpy as np


def predict_onnx(session, values, batch_size=65536):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    output = None
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = np.ascontiguousarray(values[start:stop], dtype=np.float32)
        prediction = session.run(None, {'input': batch})[0]
        if output is None:
            output = np.empty((len(values), *prediction.shape[1:]), dtype=prediction.dtype)
        output[start:stop] = prediction
    if output is None:
        return session.run(None, {'input': np.asarray(values, dtype=np.float32)})[0]
    return output
