# Sliding Window Decoder

This repo contains my implementation of a sliding window decoder for the QLDPC codes from the [IBM paper](https://arxiv.org/pdf/2308.07915.pdf) on circuit-level noise.

- `IBM.ipynb` aims to reproduce the results in Figure 3 of the above paper.
- `Round Analysis.ipynb` takes a closer look at the parity check matrix, explains the basic concept of sliding window decoding and show how the windows are extracted.
- `Sliding Window.ipynb` contains the complete pipleline of code/circuit creation, window extraction and BP+OSD is used to decode each window.