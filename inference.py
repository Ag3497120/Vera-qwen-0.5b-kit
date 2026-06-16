import struct
import torch
import torch.nn as nn
from transformers import AutoTokenizer

def load_jgen(filepath):
    print(f"Loading Generative Matrix from {filepath}")
    tensors = {}
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"JGEN":
            raise ValueError("Invalid JGEN file")
            
        version = struct.unpack("<I", f.read(4))[0]
        num_layers, rank = struct.unpack("<I I", f.read(8))
        print(f"Version: {version}, Layers: {num_layers}, Rank: {rank}")
        
        while True:
            header = f.read(1)
            if not header:
                break
                
            block_type = struct.unpack("<B", header)[0]
            
            if block_type == 0:
                rows, cols = struct.unpack("<I I", f.read(8))
                data = f.read(rows * cols * 2)
                tensors["embed_tokens"] = torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(rows, cols).clone()
            elif block_type == 1:
                rows, cols = struct.unpack("<I I", f.read(8))
                data = f.read(rows * cols * 2)
                tensors["lm_head"] = torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(rows, cols).clone()
            elif block_type == 2:
                rows, cols = struct.unpack("<I I", f.read(8))
                data = f.read(rows * cols * 2)
                tensors["norm"] = torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(rows).clone()
            elif block_type == 3:
                z, rows, cols = struct.unpack("<B I I", f.read(9))
                data = f.read(rows * cols * 2)
                tensors[f"{z}_attn_norm"] = torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(rows).clone()
            elif block_type == 4:
                z, rows, cols = struct.unpack("<B I I", f.read(9))
                data = f.read(rows * cols * 2)
                tensors[f"{z}_mlp_norm"] = torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(rows).clone()
            elif block_type == 5:
                z, mtype, rows, cols, r = struct.unpack("<B B I I I", f.read(14))
                
                U = torch.frombuffer(bytearray(f.read(rows * r * 2)), dtype=torch.float16).reshape(rows, r).clone()
                S = torch.frombuffer(bytearray(f.read(r * 2)), dtype=torch.float16).reshape(r).clone()
                V = torch.frombuffer(bytearray(f.read(r * cols * 2)), dtype=torch.float16).reshape(cols, r).T.clone()
                mod_x = torch.frombuffer(bytearray(f.read(cols * 2)), dtype=torch.float16).reshape(cols).clone()
                mod_y = torch.frombuffer(bytearray(f.read(rows * 2)), dtype=torch.float16).reshape(rows).clone()
                
                tensors[f"{z}_{mtype}"] = {
                    "U": U, "S": S, "V": V, "mod_x": mod_x, "mod_y": mod_y
                }
                
    return tensors, num_layers, rank

def rms_norm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight

def apply_rope(q, k, pos, head_dim=64):
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    
    freqs = pos * inv_freq
    freqs = torch.cat((freqs, freqs), dim=-1)
    
    sin = torch.sin(freqs)
    cos = torch.cos(freqs)
    
    def rotate(t):
        t1, t2 = t[..., :head_dim//2], t[..., head_dim//2:]
        return torch.cat((-t2, t1), dim=-1)
        
    q_out = q.view(-1, head_dim)
    q_out = (q_out * cos) + (rotate(q_out) * sin)
    
    k_out = k.view(-1, head_dim)
    k_out = (k_out * cos) + (rotate(k_out) * sin)
    
    return q_out.flatten(), k_out.flatten()

def generative_matmul(x, params):
    # h = V^T * (x * modX)
    # y = modY * (U * S * h)
    
    x = x.float()
    mod_x = params["mod_x"].float()
    V = params["V"].float()
    S = params["S"].float()
    U = params["U"].float()
    mod_y = params["mod_y"].float()
    
    h = torch.matmul(x * mod_x, V)
    y = torch.matmul(h * S, U.T)
    return y * mod_y

def generate(prompt, jgen_path, max_tokens=30):
    tensors, layers, rank = load_jgen(jgen_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
    
    input_ids = tokenizer.encode(prompt)
    print(f"Input: {prompt}")
    
    head_dim = 64
    num_q_heads = 1024 // 64
    num_kv_heads = 1024 // 64
    
    kv_cache = {z: {'k': [], 'v': []} for z in range(layers)}
    
    for t in range(max_tokens):
        token = input_ids[-1]
        x = tensors["embed_tokens"][token].float()
        pos = len(input_ids) - 1
        
        for z in range(layers):
            residual = x
            
            x = rms_norm(x, tensors[f"{z}_attn_norm"].float())
            
            q = generative_matmul(x, tensors[f"{z}_7"])
            k = generative_matmul(x, tensors[f"{z}_8"])
            v = generative_matmul(x, tensors[f"{z}_9"])
            
            q, k = apply_rope(q, k, pos, head_dim)
            
            kv_cache[z]['k'].append(k)
            kv_cache[z]['v'].append(v)
            
            K = torch.stack(kv_cache[z]['k'])
            V = torch.stack(kv_cache[z]['v'])
            
            q = q.view(num_q_heads, head_dim)
            K = K.view(-1, num_kv_heads, head_dim)
            V = V.view(-1, num_kv_heads, head_dim)
            
            attn_out = torch.zeros_like(q)
            for h in range(num_q_heads):
                q_h = q[h]
                K_h = K[:, h, :]
                V_h = V[:, h, :]
                
                scores = torch.matmul(K_h, q_h) / (head_dim ** 0.5)
                probs = torch.softmax(scores, dim=0)
                attn_out[h] = torch.matmul(probs, V_h)
                
            attn_out = attn_out.flatten()
            
            x = generative_matmul(attn_out, tensors[f"{z}_20"])
            
            x = x + residual
            residual = x
            
            x = rms_norm(x, tensors[f"{z}_mlp_norm"].float())
            
            gate = generative_matmul(x, tensors[f"{z}_10"])
            up = generative_matmul(x, tensors[f"{z}_11"])
            
            swiglu = torch.nn.functional.silu(gate) * up
            
            x = generative_matmul(swiglu, tensors[f"{z}_12"])
            
            x = x + residual
            
        x = rms_norm(x, tensors["norm"].float())
        logits = torch.matmul(tensors["lm_head"].float(), x)
        
        next_token = torch.argmax(logits).item()
        input_ids.append(next_token)
        
        print(tokenizer.decode([next_token]), end='', flush=True)
    print()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .jgen file")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt to generate text for")
    parser.add_argument("--tokens", type=int, default=30, help="Number of tokens to generate")
    args = parser.parse_args()
    
    generate(args.prompt, args.model, args.tokens)
