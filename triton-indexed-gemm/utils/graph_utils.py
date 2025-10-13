from transformers import StaticCache
import torch

def capture_graph(model, batch_size, past_key_values_buf=None, token_dtype=torch.int, N_warmup=10):
    assert past_key_values_buf is None or isinstance(past_key_values_buf, StaticCache) # must use static cache
    assert model.device.type == "cuda" # must use cuda
    device = model.device
    token_buf = torch.full((batch_size, 1), 0, dtype=token_dtype, device=device)
    cache_position_buf = torch.full((1,), 1, dtype=torch.int, device=device)
    
    # if past_key_values_buf is None, create a new one
    if past_key_values_buf is None:
        max_cache_length = 2048
        past_key_values_buf = StaticCache(model.config, 1, max_cache_length, device, dtype=model.dtype)
    # warmup before capturing the graph
    # this only warmup the inference stage, prefill stage has different batch size and requires separate warmup
    with torch.no_grad():
        for i in range(N_warmup):
            logits = model(token_buf, cache_position=cache_position_buf, past_key_values=past_key_values_buf, return_dict=False, use_cache=True)[0]

    # start capturing the graph
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            logits = model(token_buf, cache_position=cache_position_buf, past_key_values=past_key_values_buf, return_dict=False, use_cache=True)[0]
    
    def run_graph(new_token_id, cache_position):
        token_buf.copy_(new_token_id)
        cache_position_buf.copy_(cache_position)
        g.replay()
        return logits.clone()
    
    return run_graph, past_key_values_buf