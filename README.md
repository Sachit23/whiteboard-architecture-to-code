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
