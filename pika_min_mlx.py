import mlx.core as mx
import mlx.nn as nn


def gelu(x):
    return nn.activations.gelu(x)


def softmax(x):
    return nn.activations.softmax(x)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.var(x, axis=-1, keepdims=True)
    # normalize x to have mean=0 and var=1 over last axis
    x = (x - mean) / mx.sqrt(variance + eps)
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    # [n_seq, n_embd] -> [n_seq, 4*n_embd]
    a = gelu(linear(x, **c_fc))

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


# [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
def attention(q, k, v, mask):
    return softmax(q @ k.T / mx.sqrt(mx.array(q.shape[-1])) + mask) @ v


# [n_seq, n_embd] -> [n_seq, n_embd]
def mha(x, c_attn, c_proj, n_head):
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]
    qkv = mx.split(x, 3, axis=-1)
    # split into heads
    # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]
    qkv_heads = list(map(lambda x: mx.split(x, n_head, axis=-1), qkv))
    # causal mask to hide future inputs from being attended to
    # [n_seq, n_seq]
    causal_mask = (1 - mx.tri(x.shape[0], dtype=x.dtype)) * -1e10

    # perform attention over each head
    # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]
    out_heads = [attention(q, k, v, causal_mask)
                 for q, k, v in zip(*qkv_heads)]

    # merge heads
    # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
    x = mx.concatenate(out_heads, axis=1)

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


# [n_seq, n_embd] -> [n_seq, n_embd]
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    # multi-head causal self attention
    # [n_seq, n_embd] -> [n_seq, n_embd]
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)

    # position-wise feed forward network
    # [n_seq, n_embd] -> [n_seq, n_embd]
    x = x + ffn(layer_norm(x, **ln_2), **mlp)

    return x


# [n_seq] -> [n_seq, n_vocab]
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    # token + positional embeddings

    inputs = mx.array(inputs)

    # x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    x = wte[inputs] + wpe[:len(inputs)]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        # [n_seq, n_embd] -> [n_seq, n_embd]
        x = transformer_block(x, **block, n_head=n_head)

    # projection to vocab
    # [n_seq, n_embd] -> [n_seq, n_embd]
    x = layer_norm(x, **ln_f)
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = mx.argmax(logits[-1])  # greedy sampling
        # extract the value
        next_id = next_id.item()
        inputs.append(int(next_id))  # append prediction to input

    # only return generated ids
    return inputs[len(inputs) - n_tokens_to_generate:]


def main(prompt: str, numpy: bool = False, n_tokens_to_generate: int = 40,
         model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params, param_dict_to_mxarray

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(
        model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # change input_ids to mx.array in gpt2() but not for generate()
    input_ids = input_ids
    hparams = hparams
    params = param_dict_to_mxarray(params)

    # generate output ids
    output_ids = generate(
        input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
