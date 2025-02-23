# Lossless compression challenge

**Prize: highest compression rate on 5,000 minutes of driving video (~915MB) - Challenge ends July, 1st 2024 11:59pm AOE**

Submit a single zip archive containing
- a compressed version of the first two splits (5,000 minutes) of the commaVQ dataset
- a python script named `decompress.py` to save the decompressed files into their original format in `OUTPUT_DIR`

Everything in this repository and in PyPI is assumed to be available (you can `pip install` in the decompression script), anything else should to be included in the archive.

To evalute your submission, we will run:
```bash
./compression/evaluate.sh path_to_submission.zip
```

| Implementation                                                                 | Compression rate |
| :----------------------------------------------------------------------------- | ---------------: |
| [pkourouklidis](https://github.com/pkourouklidis) (arithmetic coding with GPT) |              2.6 |
| anonymous (zpaq)                                                               |              2.3 |
| [rostislav](https://github.com/rostislav) (zpaq)                               |              2.3 |
| anonymous (zpaq)                                                               |              2.2 |
| anonymous (zpaq)                                                               |              2.2 |
| [0x41head](https://github.com/0x41head) (zpaq)                                 |              2.2 |
| [tillinf](https://github.com/tillinf) (zpaq)                                   |              2.2 |
| baseline (lzma)                                                                |              1.6 |


Have fun!

# Windows (Updated for 2025)

## Setup

Using Powershell from the `./compression/` directory:

```powershell
python -m venv venv
venv\Scripts\activate
pip install numpy
pip install datasets
```

## Environment Verification

### Compression

Run `python compress.py` and you should have a Compression Ratio of `1.6`

### Decompression

Manually unzip `compression_challenge_submission.zip` into a folder named `decompressed`. Make sure there are no sub-folders. You will need an environment variable here since it's not `bash`:

```powershell
$env:OUTPUT_DIR = "$(Resolve-Path .\decompressed)"
python decompress.py
```

It'll work with the 2 data splits, but no output message means success.

### Evaluation

Two more environment variables for this to run in Powershell, and if successful it will output the Compression Rate just like `compress.py`:

```powershell
$env:UNPACKED_ARCHIVE = "$(Resolve-Path .\decompressed)"
$env:PACKED_ARCHIVE = "$(Resolve-Path .\compression_challenge_submission.zip)"
python evaluate.py
```

## Issues

### `compress.py`

#### Original download of HuggingFace dataset might fail and need an extra Query Parameter set

Replace `ds = load_dataset...` with this:

```python
ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits, data_files=data_files, trust_remote_code=True)
```

#### If that needs to be set to True, that line then needs to be removed from the cache for the program to actually work

Find the cache directory (something like: `C:\Users\info\.cache\huggingface\modules\datasets_modules\datasets\commaai--commavq\`) and edit the `commavq.py` file in there to remove this line:

```python
dl_manager.download_config.ignore_url_params = True
```

### `decompress.py`

#### Tries to download entire HuggingFace dataset

Edit it like this:

```python
splits = ['0', '1']
data_files = {'0': 'data_0_to_2500.zip', '1': 'data_2500_to_5000.zip'}
ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits, data_files=data_files, trust_remote_code=True)
ds = DatasetDict(zip(splits, ds))
```

### `evaluate.py`

#### Tries to download entire HuggingFace dataset

Edit it like this:

```python
if __name__ == '__main__':
  num_proc = multiprocessing.cpu_count()
  # load split 0 and 1
  splits = ['0', '1']
  data_files = {'0': 'data_0_to_2500.zip', '1': 'data_2500_to_5000.zip'}
  ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits, data_files=data_files, trust_remote_code=True)
  ds = DatasetDict(zip(splits, ds))
  # compare
  ds.map(compare, desc="compare", num_proc=num_proc, load_from_cache_file=False)
  # print compression rate
  rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / archive_path.stat().st_size
  print(f"Compression rate: {rate:.1f}")
```