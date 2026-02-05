# GPTZip
 This is a repository for my final year project "Using Large Language Models for text compression". It uses the Rank Zip algorithm along with the Arithmetic Coding + LLM PMF model algorithm as outlined in this paper https://arxiv.org/abs/2306.04050 and in this repo https://github.com/erika-n/GPTzip


## Overview
`llmzip.py` is a CLI tool to **compress (zip)** and **decompress (unzip)** files using an LLM-backed compression scheme.

---

## Requirements
- **Python 3.10.13** (recommended, as the code was written with this version)
- An isolated environment is recommended (either `venv` or `conda`)

---

## Setup

### Option A: `venv` (recommended)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: `conda`
```bash
conda create -n llmzip python=3.10.13 -y
conda activate llmzip
pip install -r requirements.txt
```

After successfully installing the packages, the program is ready to run.

---

## Usage

### Compress (Zip)
```bash
python llmzip.py -z file_path -c compressed_path -model model -method method
```

**Parameters**
- `file_path`: path of the file to be compressed
- `compressed_path`: path where the compressed representation will be saved
- `model`: which model should be used  
  - possible values: `Mixtral`, `Yi`
- `method`: which compression scheme will be used  
  - possible values: `Rank`, `AC`  
  - **Note:** some examples use `Ranks`—use the value your code expects.

**Example**
```bash
python llmzip.py -z English.txt -c English.gpz -model Mixtral -method Ranks
```

---

### Decompress (Unzip)
```bash
python llmzip.py -u compressed_path -c output_path -model model -method method
```

**Parameters**
- `compressed_path`: path of the file to be decompressed
- `output_path`: path where the decompressed file will be saved
- `model`: which model should be used  
  - possible values: `Mixtral`, `Yi`
- `method`: which compression scheme will be used  
  - possible values: `Rank`, `AC`

**Example**
```bash
python llmzip.py -u Spanish.gpz -c SpanishUnzipped.txt -model Yi -method AC
```
