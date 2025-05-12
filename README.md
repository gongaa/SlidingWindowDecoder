# Sliding Window with Guided Decimation Guessing (GDG) Decoding

This repo contains the source codes of the paper [Toward Low-latency Iterative Decoding of QLDPC Codes Under Circuit-Level Noise](https://arxiv.org/pdf/2403.18901.pdf).

Update [May-12-2025]
- [x] fix typos, update BP+OSD import/calls to ldpc_v2, add [CUDA-Q QEC](https://nvidia.github.io/cudaqx/components/qec/introduction.html) GPU decoder into `Sliding Window OSD.ipynb`
- [x] [SHYPS](https://arxiv.org/pdf/2502.07150) memory experiment
- [x] FAQ page
- [] make sliding window part more reusable
- [] GDG stim integration (update BP implementation to LDPC v2, improve data qubit noise multi-thread GDG implementation)
- [] let MAC users install from the directory
- [] CUDA implementation of GDG
- [] [logical Clifford synthesis](https://github.com/gongaa/RM127/blob/main/efficiency.pdf) compiler
- [] SHYPS logical experiment 
- [] QDistRnd to get code distance 


Please checkout the FAQ page for installation!

## Notebooks

- `IBM.ipynb` aims to reproduce the results in Figure 3 of the [IBM paper](https://arxiv.org/pdf/2308.07915v1.pdf) (this is arXiv v1, they updated their numerical results in v2). `N72circuit.svg` is the visualization of the noiseless, single syndrome cycle Stim circuit of their Figure 7 for the [[72,12,6]] code.
- `Round Analysis.ipynb` takes a closer look at the parity check matrix, explains the basic concept of sliding window decoding and show how the windows are extracted.
- `Sliding Window OSD.ipynb` contains the complete pipleline of code/circuit creation, window extraction and BP+OSD is used to decode each window.
- `Data noise.ipynb` introduces basic usage of GDG, and is for producing Figure 4 of our paper.
- `Sliding Window GDG.ipynb` uses GDG on each window for decoding, for Figure 3 and 7.
- `Syndrome code.ipynb` is related to Appendix B.
- `Misc.ipynb` is not related to our paper, but demonstrates my implementation of the following papers: [BP4+OSD](https://quantum-journal.org/papers/q-2021-11-22-585/pdf/), [2BGA codes](https://arxiv.org/pdf/2306.16400.pdf), [CAMEL](https://arxiv.org/pdf/2401.06874.pdf), [BPGD](https://arxiv.org/pdf/2312.10950.pdf).

## Directory Layout
    src
    ├── include
    │   └── bpgd.cpp              # base class for handling decimation, multi-thread GDG
    │
    ├── bp_guessing_decoder.pyx   # single-thread GDG, it shares interface with the multi-thread version
    ├── osd_window.pyx            # OSD on shortened window PCM
    ├── codes_q.py                # code constructions
    ├── build_circuit.py          # build Stim circuit for BB codes
    └── simulation.py             # simulation framework for data qubit noise