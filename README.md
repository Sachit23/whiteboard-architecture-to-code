# Whiteboard Architecture-to-Code Generator

A Python project that turns a structured architecture sketch into infrastructure configuration.

The first target is Docker Compose for an API, PostgreSQL, and Redis or Kafka. A later Terraform generator will use the same reviewed architecture graph and an explicit provider profile.

The pipeline will use OpenCV to detect shapes, a custom PyTorch CNN trained from scratch to classify them, OCR to read labels, and a parser to link nodes using arrows.

## Milestone 1

This starter contains the package layout and requires only Python 3 for its smoke check.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m app
```

Expected output: `Whiteboard Architecture-to-Code Generator is ready.`

Later milestones add image preprocessing, symbol classification and training, OCR, graph parsing, and configuration generation in that order.

## Symbol training

With the virtual environment activated, install the training dependencies and run these commands from the repository root:

```sh
python -m pip install numpy opencv-python torch
python -m app.generate_symbols
python -m app.generate_dataset
python -m app.train
```

The generators create four symbol prototypes, 400 training images, and 80 validation images. Training runs for 10 epochs and saves the final weights to `artifacts/symbol_cnn.pt`.

The generated folders `data/prototypes/`, `data/symbols/`, and `artifacts/` are ignored by Git. Run the commands above to recreate them after cloning the repository.
