"""CPU reproduction of the arithmetic performed by needle-webgpu.

This is deliberately a debugging backend: GEMM uses the same tensor layout as
engine-core.js (the stored weight is [out, in], so the operation is A @ W.T).
It lets the timeline compare the browser algorithm without requiring a browser
or a WebGPU adapter.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np

TAG = 0x05E12A83
HEADER = 120
RECORD = 44
FP16, FP32, CQ, RAW = 1, 2, 3, 4


def _half(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def _fwht(a: np.ndarray) -> np.ndarray:
    """FWHT over the last axis, matching the JS in-place transform."""
    out = np.asarray(a, dtype=np.float32).copy()
    n = 1
    while n < out.shape[-1]:
        shape = out.shape[:-1] + (out.shape[-1] // (2 * n), 2, n)
        blocks = out.reshape(shape)
        x = blocks[:, :, 0, :].copy() if out.ndim > 1 else None
        left = blocks[..., 0, :].copy()
        right = blocks[..., 1, :].copy()
        blocks[..., 0, :] = left + right
        blocks[..., 1, :] = left - right
        n *= 2
    return out * np.float32(1.0 / math.sqrt(out.shape[-1]))


def _unpack(data: bytes, bits: int, row: int, width: int, stride: int) -> np.ndarray:
    out = np.empty(width, dtype=np.uint8)
    if bits == 5:
        start = row * stride
        for i in range(width):
            c = (data[start + (i >> 2)] >> ((i & 3) * 2)) & 3
            out[i] = 0 if c == 3 else c + 1
        return out
    start = row * stride
    mask = (1 << bits) - 1
    for i in range(width):
        bit = i * bits
        value = data[start + (bit >> 3)] >> (bit & 7)
        if (bit & 7) + bits > 8:
            value |= data[start + (bit >> 3) + 1] << (8 - (bit & 7))
        out[i] = value & mask
    return out


def _cq(t: dict[str, Any], codebook: np.ndarray) -> np.ndarray:
    rows, dim, group, bits = t['shape'][0], t['shape'][1], t['group'] or 128, t['bits']
    padded = math.ceil(dim / group) * group
    packed = padded // 4 if bits == 5 else padded * bits // 8
    groups = padded // group
    stride = packed + groups * 2
    if bits == 2: code = codebook[:4]
    elif bits == 3: code = codebook[4:12]
    elif bits == 4: code = codebook[12:28]
    else: code = np.asarray([-1.2240064, 0, 1.2240064], dtype=np.float32) / math.sqrt(group)
    raw = np.frombuffer(t['data'], dtype=np.uint8).reshape(rows, stride)
    packed_data = raw[:, :packed]
    if bits == 5:
        ids = ((packed_data[:, :, None] >> (np.arange(4, dtype=np.uint8) * 2)) & 3).reshape(rows, -1)
        ids = np.where(ids == 3, 0, ids + 1)
    else:
        bits_in_byte = np.arange(8, dtype=np.uint8)
        expanded = np.unpackbits(packed_data, axis=1, bitorder='little')
        ids = np.stack([np.packbits(expanded[:, i*bits:i*bits+bits], axis=1, bitorder='little')[:, 0] for i in range(padded)], axis=1) & ((1 << bits) - 1)
    scales = raw[:, packed:packed + groups * 2].view('<u2').astype(np.uint16).view(np.float16).astype(np.float32)
    scales = np.where(np.isfinite(scales), scales, 0).astype(np.float32)
    values = code[ids].reshape(rows, groups, group) * scales[:, :, None]
    transformed = _fwht(values.reshape(rows * groups, group)).reshape(rows, padded)
    return transformed[:, :dim].reshape(-1).astype(np.float32)


class CactModel:
    def __init__(self, path: str | Path):
        raw = Path(path).read_bytes()
        if len(raw) < HEADER or struct.unpack_from('<I', raw)[0] != TAG:
            raise ValueError(f'{path} is not a needle-webgpu .cact file')
        nt, cn = struct.unpack_from('<II', raw, 4)
        self.codebook = np.frombuffer(raw, dtype='<f4', count=cn, offset=HEADER).copy()
        keys = ['vocab_size','d_model','num_heads','num_kv_heads','num_layers','head_dim','max_seq_len','hada_n','mhc_lanes','engram_slots','engram_sub_dim','num_engram_tables','engram_conv_taps','engram_conv_dilation']
        self.g = {key: struct.unpack_from('<I', raw, 20 + i * 4)[0] for i, key in enumerate(keys)}
        self.g['kv_window'], self.g['kv_bits'] = struct.unpack_from('<II', raw, 12)
        no = struct.unpack_from('<I', raw, 76)[0]
        self.g['engram_orders'] = [struct.unpack_from('<I', raw, 80 + i * 4)[0] for i in range(no)]
        ns = struct.unpack_from('<I', raw, 96)[0]
        self.g['engram_layers'] = [struct.unpack_from('<I', raw, 100 + i * 4)[0] for i in range(ns)]
        self.g['rope_theta'] = struct.unpack_from('<f', raw, 116)[0]
        self.weights: list[np.ndarray] = []
        self.tensors: list[dict[str, Any]] = []
        offset = HEADER + cn * 4
        for _ in range(nt):
            dtype, nd = raw[offset], raw[offset + 1]
            shape = [struct.unpack_from('<I', raw, offset + 4 + j * 4)[0] for j in range(nd)]
            data_offset, nbytes = struct.unpack_from('<QQ', raw, offset + 20)
            tensor = {'dtype': dtype, 'shape': shape, 'group': struct.unpack_from('<I', raw, offset + 36)[0], 'bits': struct.unpack_from('<I', raw, offset + 40)[0], 'data': raw[data_offset:data_offset+nbytes]}
            self.tensors.append(tensor)
            offset += RECORD
            if dtype == RAW: continue
            if dtype == FP16: value = np.frombuffer(tensor['data'], dtype='<f2').astype(np.float32)
            elif dtype == FP32: value = np.frombuffer(tensor['data'], dtype='<f4').copy()
            elif dtype == CQ: value = _cq(tensor, self.codebook)
            else: raise ValueError(f'unsupported cact dtype {dtype}')
            self.weights.append(value)
        if not any(t['dtype'] == RAW for t in self.tensors): raise ValueError('cact has no tokenizer tensor')

    def tokenizer_bytes(self) -> bytes:
        return next(t['data'] for t in self.tensors if t['dtype'] == RAW)


def _matrix(w: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """View a WebGPU weight as [rows, cols], zero-padding shader OOB reads."""
    out = np.zeros((rows, cols), dtype=np.float32)
    flat = np.asarray(w, dtype=np.float32).reshape(-1)
    out.reshape(-1)[:min(flat.size, out.size)] = flat[:out.size]
    return out


def _norm(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    d = len(scale)
    rows = x.reshape(-1, d)
    return (rows * (1 + scale)) / np.sqrt(np.sum(rows * rows, axis=1, keepdims=True) / d + 1e-6)


def _at(w: np.ndarray, index: int, default: float = 0.0) -> float:
    return float(w[index]) if 0 <= index < w.size else default


def _sigmoid(x: Any) -> Any:
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _sink(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    n = a.shape[0]
    for _ in range(20):
        a = np.exp(a - np.max(a, axis=1, keepdims=True)); a /= np.maximum(np.sum(a, axis=1, keepdims=True), 1e-12)
        a /= np.maximum(np.sum(a, axis=0, keepdims=True), 1e-12)
    return a


def _att(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    T, heads, d = q.shape; kh = k.shape[1]; out = np.zeros_like(q); rep = heads // kh
    for t in range(T):
        for h in range(heads):
            scores = np.sum(q[t, h] * k[:t+1, h // rep], axis=1) / math.sqrt(d)
            p = np.exp(scores - np.max(scores)); p /= np.sum(p)
            out[t, h] = np.sum(p[:, None] * v[:t+1, h // rep], axis=0)
    return out


def run_webgpu(model: CactModel, tokens: list[int], tracer: Any, *, trace_level: str = 'op') -> np.ndarray:
    g, w, emit = model.g, model.weights, tracer.emit
    D, Hd, lanes, N = g['d_model'], g['head_dim'], g['mhc_lanes'], len(tokens)
    wi = 1; layers = []
    for _ in range(g['num_layers']):
        layers.append(dict(zip(('norm','q','k','v','qn','kn','gate','out','post','ag','pre','d1','d2','d3'), range(wi, wi + 14)))); wi += 14
    mh = dict(zip(('ap','apost','ar','bp','bpost','br','pp','ppo','pr'), range(wi, wi + 9))); wi += 9
    engrams = list(range(wi, wi + 4 * len(g['engram_layers']))); wi += 4 * len(g['engram_layers']); final = wi
    embedding = w[0].reshape(g['vocab_size'], D)
    x = np.tile(embedding[np.asarray(tokens, dtype=np.int64)].reshape(N, 1, D), (1, lanes, 1)) * np.float32(math.sqrt(D))
    emit('embedding_output', name='webgpu.embedding', tensors={'hidden': x}, metadata={'backend':'webgpu','source':'needle-webgpu'})
    for l, lc in enumerate(layers):
        z = x.copy(); nx = z / np.sqrt(np.sum(z*z, axis=(1,2), keepdims=True)/(lanes*D)+1e-6)
        pp = _matrix(w[mh['pp']], lanes, lanes * D)
        hp = nx.reshape(N, lanes*D) @ pp.T
        if w[mh['pp']].size < lanes * lanes * D:
            emit('webgpu.layout_warning', layer=l, name=f'webgpu.layer.{l}.mhc.pp.layout', metadata={'weight_elements': int(w[mh['pp']].size), 'shader_elements': lanes * lanes * D, 'behavior': 'zero-padded OOB read'})
        u = np.zeros((N,D), dtype=np.float32)
        for t in range(N):
            for lane in range(lanes):
                gate = float(_sigmoid(hp[t,lane]*w[mh['ap']][l] + (w[mh['bp']][l*lanes+lane] if l*lanes+lane < len(w[mh['bp']]) else 0) + 8*(lane == l%lanes)-4))
                u[t] += gate*z[t,lane]
        for si, site_layer in enumerate(g['engram_layers']):
            if site_layer != l: continue
            base, orders = engrams[si], g['engram_orders']; heads = g['num_engram_tables']//len(orders); sub = g['engram_sub_dim']; flat=np.zeros((N, heads*len(orders)*sub), np.float32)
            for t in range(N):
                for oi, order in enumerate(orders):
                    if t < order-1: continue
                    for h in range(heads):
                        a=(0x9e3779b9*(oi*heads+h+1))&0xffffffff
                        for j in range(order): a=((a ^ (tokens[t-j] if t-j>=0 else 0))*0x01000193)&0xffffffff
                        idx=((a^(a>>15))&0xffffffff)%g['engram_slots']; row=(oi*heads+h)*g['engram_slots']+idx
                        table = w[base].reshape(-1, sub)
                        if row >= table.shape[0]:
                            emit('webgpu.layout_warning', layer=l, name=f'webgpu.layer.{l}.engram.layout', metadata={'weight_index': base, 'row': row, 'available_rows': int(table.shape[0]), 'behavior': 'zero-filled OOB read'})
                            continue
                        flat[t,(oi*heads+h)*sub:(oi*heads+h+1)*sub]=table[row]
            ek=flat@_matrix(w[base+1], D, flat.shape[1]).T; raw_ev=flat@_matrix(w[base+2], D, flat.shape[1]).T; taps=_matrix(w[base+3], g['engram_conv_taps'] or 4, D); ev=np.zeros((N,D),np.float32); max_order=max(orders)
            for t in range(N):
                for j in range(g['engram_conv_taps'] or 4):
                    src=t-j*(g['engram_conv_dilation'] or 1)
                    if src>=0 and t>=j*max_order: ev[t]+=taps[j] * raw_ev[src]
            for t in range(N):
                alpha=float(_sigmoid(float(u[t]@ek[t])/math.sqrt(max(1,float(u[t]@u[t])*float(ek[t]@ek[t]))))); u[t]+=alpha*ev[t]
        un=_norm(u,w[lc['norm']].reshape(-1)); q0=un@w[lc['q']].reshape(-1,D).T; k0=un@w[lc['k']].reshape(-1,D).T; v0=un@w[lc['v']].reshape(-1,D).T
        q=q0.reshape(N,g['num_heads'],Hd); k=k0.reshape(N,g['num_kv_heads'],Hd)
        q=q*(1+w[lc['qn']].reshape(1,1,Hd))/np.sqrt(np.sum(q*q,axis=2,keepdims=True)/Hd+1e-6); k=k*(1+w[lc['kn']].reshape(1,1,Hd))/np.sqrt(np.sum(k*k,axis=2,keepdims=True)/Hd+1e-6)
        for arr in (q,k):
            half=Hd//2
            for t in range(N):
                for j in range(half):
                    a=t*g['rope_theta']**(-2*j/Hd); c,s=math.cos(a),math.sin(a); X,Y=arr[t,:,j].copy(),arr[t,:,j+half].copy(); arr[t,:,j]=X*c-Y*s; arr[t,:,j+half]=Y*c+X*s
        a=_att(q,k,v0.reshape(N,g['num_kv_heads'],Hd)); gate=un@w[lc['gate']].reshape(-1,D).T; a*=_sigmoid(gate.reshape(N,g['num_heads'],Hd)); ao=_norm(a.reshape(N,-1)@w[lc['out']].reshape(-1,g['num_heads']*Hd).T,w[lc['post']].reshape(-1)); block=un+float(_sigmoid(_at(w[lc['ag']], l)))*ao
        h=_norm(block,w[lc['pre']].reshape(-1)); tmp=np.zeros((N,g['hada_n']),np.float32); tmp[:,:D]=h
        tmp=_fwht(tmp.reshape(-1,g['hada_n'])).reshape(N,-1)*w[lc['d1']][:g['hada_n']]; tmp=_silu(tmp*w[lc['d2']][:g['hada_n']]); tmp=_fwht(tmp.reshape(-1,g['hada_n'])).reshape(N,-1); block += tmp[:,:D]*w[lc['d3']][:D]
        ppo = _matrix(w[mh['ppo']], lanes, lanes * D); pr = _matrix(w[mh['pr']], lanes * lanes, lanes * D)
        hp2=nx.reshape(N,-1)@ppo.T; res=nx.reshape(N,-1)@pr.T
        if w[mh['pr']].size < lanes * lanes * D:
            emit('webgpu.layout_warning', layer=l, name=f'webgpu.layer.{l}.mhc.pr.layout', metadata={'weight_elements': int(w[mh['pr']].size), 'shader_elements': lanes * lanes * D, 'behavior': 'zero-padded OOB read'})
        for t in range(N):
            sm=_sink(res[t].reshape(lanes,lanes)); post=2*_sigmoid(hp2[t]*w[mh['apost']][l]+w[mh['bpost']][l*lanes:(l+1)*lanes]-4*(1-(np.arange(lanes)==l%lanes)))
            x[t]=sm@z[t]+post[:,None]*(block[t]-un[t])
        emit('layer_output', layer=l, name=f'webgpu.layer.{l}.output', tensors={'hidden':x}, metadata={'backend':'webgpu','layer':l,'source':'needle-webgpu'})
    last=np.mean(x[-1],axis=0); fn=_norm(last[None,:],w[final].reshape(-1))[0]; logits=fn@embedding.T
    emit('model_output', name='webgpu.model.output', tensors={'logits':logits}, metadata={'backend':'webgpu','source':'webgpu GEMM'})
    return logits
