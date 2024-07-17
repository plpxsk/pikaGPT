# pikaGPT

`pikaGPT`: A tiny implementation of a GPT, accelerated for Apple Silicon

Built on `picoGPT`: a GPT in ~60 Lines of Numpy, using MLX: An array framework
for Apple Silicon

picoGPT: [jaymody/picoGPT](https://github.com/jaymody/picoGPT)  
Apple MLX: [ml-explore/mlx](https://github.com/ml-explore/mlx)

# Usage

	python pika.py "Alan Turing theorized that computers would one day become"

Returns something like:

	generating: 100%|█████████████████████████| 40/40 [00:00<00:00, 51.76it/s]
	 the most powerful machines on the planet.

	The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.

Note: Models will be downloaded to `/models` if needed.

Change model size, tokens to generate like, or model directory:

	python pika.py \
		"Alan Turing theorized that computers would one day become" \
		--n_tokens_to_generate 40 \
		--model_size "124M" \
		--models_dir "models"

To check against original Numpy implementation (non-MLX), add `--numpy`:

	python pika.py \
		"Alan Turing theorized that computers would one day become" \
		--numpy


# Install

_If Python>=3.12, first `pip install setuptools` to get `distutils`. See [docs](https://docs.python.org/3/whatsnew/3.12.html)_

`pip install -r requirements.txt`

Tested and benchmarked on `Python 3.12.4` and macOS Sonoma 14.5 (M1 Pro, 32GB)

# Benchmarks

MLX seems to provide >4x speedup, see iterations/second `it/s` etc:

```bash
(.venv) pikaGPT# python pika.py "Alan Turing theorized that computers would one day become"
generating: 100%|██████████████████████████| 40/40 [00:00<00:00, 53.03it/s]
 the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.


(.venv) pikaGPT# python pika.py "Alan Turing theorized that computers would one day become" --numpy
generating: 100%|██████████████████████████| 40/40 [00:04<00:00,  9.54it/s]
 the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.


(.venv) pikaGPT# python pika.py "Alan Turing theorized that computers would one day become" --model_size "1558M"
generating: 100%|██████████████████████████| 40/40 [00:06<00:00,  6.32it/s]
 so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T


(.venv) pikaGPT# python pika.py "Alan Turing theorized that computers would one day become" --model_size "1558M" --numpy
generating: 100%|██████████████████████████| 40/40 [00:43<00:00,  1.10s/it]
 so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T
```

# Tests

`pip install -r requirements_dev.txt`

Run some tests with `make test`
