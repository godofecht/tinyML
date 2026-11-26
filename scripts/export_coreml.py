#!/usr/bin/env python3
"""Export ML::Model text weights to a fixed-shape Core ML inference program."""
import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=int, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.topology) < 2 or any(n <= 0 for n in args.topology):
        parser.error("topology must contain at least two positive widths")
    if args.output.suffix != ".mlpackage":
        parser.error("output must end in .mlpackage")
    weights = np.array([float(x) for x in args.weights.read_text().split()], dtype=np.float32)
    expected = sum((a + 1) * b for a, b in zip(args.topology, args.topology[1:]))
    if weights.size != expected or not np.isfinite(weights).all():
        parser.error(f"expected {expected} finite float32 weights")
    layers = []
    offset = 0
    for a, b in zip(args.topology, args.topology[1:]):
        block = weights[offset:offset + (a + 1) * b].reshape(a + 1, b)
        layers.append((block[:-1].T.copy(), block[-1].copy()))
        offset += (a + 1) * b

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, args.topology[0]))],
                opset_version=ct.target.iOS16)
    def program(input):
        value = input
        for index, (matrix, bias) in enumerate(layers):
            value = mb.linear(x=value, weight=matrix, bias=bias)
            value = mb.tanh(x=value, name="output" if index == len(layers) - 1 else f"tanh_{index}")
        return value

    program.functions["main"].set_output_types([ct.TensorType(name="output", dtype=np.float32)])
    model = ct.convert(program, source="milinternal", convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS13,
                       compute_precision=ct.precision.FLOAT16,
                       inputs=[ct.TensorType(name="input", shape=(1, args.topology[0]), dtype=np.float32)],
                       skip_model_load=True)
    spec = model.get_spec()
    for features, name, width in ((spec.description.input, "input", args.topology[0]),
                                  (spec.description.output, "output", args.topology[-1])):
        if len(features) != 1 or features[0].name != name:
            raise RuntimeError(f"Unexpected Core ML feature name: expected {name}")
        array = features[0].type.multiArrayType
        if list(array.shape) != [1, width] or array.dataType != ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32:
            raise RuntimeError(f"Unexpected Core ML feature shape or type: {name}")
    model.save(str(args.output))


if __name__ == "__main__":
    main()
