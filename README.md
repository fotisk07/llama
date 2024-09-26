
# Llama Project

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This project explores the **Llama** architecture for machine learning, with implementations for positional encoding, model training, and testing. The repository provides a foundation for experiments on transformer models, focusing on their efficiency and performance in various tasks.

## Features

- **Model Implementation**: Detailed `models.py` for setting up and fine-tuning transformers.
- **Positional Encoding**: `pos_enc.py` for positional encoding strategies.
- **Training Pipeline**: `trainer.py` to manage model training and evaluations.
- **Utilities**: Various utilities to facilitate model experimentation.

## Installation

Clone the repository:

```bash
git clone https://github.com/fotisk07/llama.git
cd llama
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To train a model, run:

```bash
python trainer.py --config config.yaml
```

For testing:

```bash
python test.py --model model_path
```

## Files Structure

- `models.py`: Contains the architecture for the Llama model.
- `pos_enc.py`: Code for positional encoding.
- `trainer.py`: Handles training of the model.
- `test.py`: Testing and validation script.
- `utils.py`: Utility functions.

## License

This project is licensed under the MIT License.
