import fs from 'node:fs';
import { init, defaultDevice, numpy as np } from '@jax-js/jax';

const MAGIC = 'NEEDLEJS1';
const HEADER_BYTES = 4;
const D = np.float32;

function normalizeConfig(c) {
  const numeric = ['vocab_size','d_model','attn_dim','num_heads','num_kv_heads','num_layers','max_seq_len','pad_token_id','contrastive_dim','rope_theta','engram_heads','engram_slots','mhc_lanes','kv_window','kv_bits','act_bits','scan_unroll'];
  const out = { ...c };
  for (const k of numeric) if (k in out) out[k] = Number(out[k]);
  out.engram_orders = (out.engram_orders ?? [2,3]).map(Number);
  out.engram_layers = (out.engram_layers ?? [2,15]).map(Number);
  out.flash = out.flash === true || out.flash === 'True' || out.flash === 'true';
  out.remat = out.remat === true || out.remat === 'True' || out.remat === 'true';
  return out;
}

function readWeights(path = 'weights.bin') {
  const buf = fs.readFileSync(path);
  const magic = Buffer.from(buf.subarray(0, MAGIC.length)).toString('ascii');
  if (magic !== MAGIC) throw new Error(`bad weights.bin magic: ${magic}`);
  const headerLen = buf.readUInt32LE(MAGIC.length);
  const headerStart = MAGIC.length + HEADER_BYTES;
  const header = JSON.parse(Buffer.from(buf.subarray(headerStart, headerStart + headerLen)).toString('utf8'));
  const dataStart = headerStart + headerLen;
  const weights = {};
  for (const e of header.tensors) {
    const bytes = buf.subarray(dataStart + e.offset, dataStart + e.offset + e.nbytes);
    const raw = new Float32Array(bytes.length / 4);
    raw.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).buffer ? new Float32Array(Uint8Array.from(bytes).buffer) : []);
    weights[e.name] = np.array(raw, { dtype: D }).reshape(e.shape);
  }
  let reference = null;
  if (header.reference) {
    const r = buf.subarray(dataStart + header.reference.offset, dataStart + header.reference.offset + header.reference.nbytes);
    reference = new Float32Array(Uint8Array.from(r).buffer);
  }
  return { header: { ...header, reference }, weights };
}

function sl(x, starts, ends) {
  return x.ref.slice(...starts.map((start, i) => [start, ends[i]]));
}
function scalar(x) { return np.array(new Float32Array([x]), { dtype: D }).reshape([]); }
function sigmoid(x) { return np.divide(1, np.add(1, np.exp(np.negative(x)))); }
function silu(x) { return x.mul(sigmoid(x.ref)); }
function rmsUnit(x, eps = 1e-6) {
  const xf = x.astype(D);
  const sq = xf.ref.mul(xf.ref);
  const mean = np.mean(sq, -1).reshape([...sq.shape.slice(0, -1), 1]);
  const rmsInv = np.reciprocal(np.sqrt(np.add(mean, scalar(eps))));
  return xf.mul(rmsInv);
}
function zcrmsNorm(x, scale, eps = 1e-6) {
  const r = rmsUnit(x, eps);
  return np.multiply(np.add(1, scale), r);
}
function softmax(x, axis = -1) {
  const a = axis < 0 ? x.shape.length + axis : axis;
  const outShape = [...x.shape];
  outShape[a] = 1;
  const m = np.max(x.ref, axis).reshape(outShape);
  const e = np.exp(x.sub(m));
  const z = np.sum(e.ref, axis).reshape(outShape);
  return e.div(z);
}
function linear(x, kernel) { return np.matmul(x, kernel); }
function shiftRight(x, offset, axis = 1) {
  if (offset === 0) return x;
  const shape = [...x.shape];
  const zshape = [...shape]; zshape[axis] = offset;
  const z = np.zeros(zshape, { dtype: x.dtype });
  const parts = [];
  if (axis !== 1) throw new Error('shiftRight currently expects sequence axis=1');
  return np.concatenate([z, sl(x, [0, 0], [shape[0], shape[1] - offset])], 1);
}
function rope(x, cos, sin) {
  const h = x.shape.at(-1) / 2;
  const x1 = sl(x.ref, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], h]);
  const x2 = sl(x.ref, [0, 0, 0, h], [x.shape[0], x.shape[1], x.shape[2], x.shape[3]]);
  const c = sl(cos.ref, [0, 0], [x.shape[2], h]).reshape([1, 1, x.shape[2], h]);
  const s = sl(sin.ref, [0, 0], [x.shape[2], h]).reshape([1, 1, x.shape[2], h]);
  const a = x1.ref.mul(c.ref).sub(x2.ref.mul(s.ref));
  const b = x2.mul(c).add(x1.mul(s));
  return np.concatenate([a, b], -1);
}
function ropeFreqs(headDim, seqLen, theta) {
  const half = Math.floor(headDim / 2);
  const c = new Float32Array(seqLen * half);
  const s = new Float32Array(seqLen * half);
  for (let t = 0; t < seqLen; ++t) for (let j = 0; j < half; ++j) {
    const freq = 1 / Math.pow(theta, (2 * j) / headDim);
    const a = t * freq;
    c[t * half + j] = Math.cos(a);
    s[t * half + j] = Math.sin(a);
  }
  return [np.array(c, { dtype: D }).reshape([seqLen, half]), np.array(s, { dtype: D }).reshape([seqLen, half])];
}
function walsh(n) {
  // n is 512 for this checkpoint; construct on the host to avoid a large JS op graph.
  let h = [[1]];
  while (h.length < n) {
    const m = h.length;
    const next = Array.from({ length: m * 2 }, () => new Array(m * 2));
    for (let r = 0; r < m; ++r) for (let c = 0; c < m; ++c) {
      const v = h[r][c];
      next[r][c] = v; next[r][c + m] = v;
      next[r + m][c] = v; next[r + m][c + m] = -v;
    }
    h = next;
  }
  const a = new Float32Array(n * n), s = Math.sqrt(n);
  for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) a[i * n + j] = h[i][j] / s;
  return np.array(a, { dtype: D }).reshape([n, n]);
}

function engramGeometry(cfg) {
  const orders = cfg.engram_orders ?? [2, 3];
  const heads = cfg.engram_heads || Math.max(1, Math.floor(cfg.d_model / (orders.length * 128)));
  const subDim = Math.floor(cfg.d_model / (orders.length * heads));
  return { orders, heads, subDim };
}

function engramIndices(tokens, orders, heads, slots) {
  const B = tokens.shape[0], T = tokens.shape[1];
  const host = tokens.dataSync();
  const out = [];
  const SEED = 0x9E3779B9 >>> 0, PRIME = 0x010001F3 >>> 0;
  for (let oi = 0; oi < orders.length; ++oi) for (let h = 0; h < heads; ++h) {
    const seed = Math.imul(SEED, oi * heads + h + 1) >>> 0;
    const a = new Int32Array(B * T);
    for (let b = 0; b < B; ++b) for (let t = 0; t < T; ++t) {
      let acc = seed;
      for (let j = 0; j < orders[oi]; ++j) {
        const u = j <= t ? (host[b * T + t - j] >>> 0) : 0;
        acc = Math.imul((acc ^ u) >>> 0, PRIME) >>> 0;
      }
      acc = (acc ^ (acc >>> 15)) >>> 0;
      a[b * T + t] = acc % slots;
    }
    out.push(np.array(a, { dtype: np.int32 }).reshape([B, T]));
  }
  return out;
}

function makeEngramKV(tokens, maskKeep, cfg, w) {
  if (!cfg.engram_layers?.length) return null;
  const { orders, heads, subDim } = engramGeometry(cfg);
  const idx = engramIndices(tokens, orders, heads, cfg.engram_slots);
  const tables = [];
  for (let s = 0; s < idx.length; ++s) {
    const table = w[`engrams_${0}/embedding`];
    void table;
  }
  const numTables = orders.length * heads;
  const embeddings = [];
  for (let s = 0; s < cfg.engram_layers.length; ++s) {
    const table = w[`engrams_${s}/embedding`];
    const fetched = [];
    for (let j = 0; j < numTables; ++j) {
      const one = sl(table.ref, [j, 0, 0], [j + 1, table.shape[1], table.shape[2]]).reshape([table.shape[1], table.shape[2]]);
      const gathered = np.take(one, idx[j].ref, 0);
      const order = orders[Math.floor(j / heads)];
      const ok = np.array(new Float32Array(Array.from({ length: tokens.shape[1] }, (_, t) => t >= order - 1 ? 1 : 0)), { dtype: D }).reshape([1, tokens.shape[1], 1]);
      fetched.push(gathered.mul(ok));
    }
    let e = np.stack(fetched, 2).reshape([tokens.shape[0], tokens.shape[1], numTables * subDim]);
    e = e.mul(maskKeep.ref.reshape([1, tokens.shape[1], 1]));
    embeddings.push(e);
  }
  const ks = [], vs = [];
  const maxOrder = Math.max(...orders);
  for (let s = 0; s < cfg.engram_layers.length; ++s) {
    const e = embeddings[s];
    let k = linear(e.ref, w[`engrams_${s}/key_proj/kernel`]);
    let v = linear(e, w[`engrams_${s}/value_proj/kernel`]);
    const taps = w[`engrams_${s}/taps`];
    let vv = np.zeros(v.shape, { dtype: D });
    for (let j = 0; j < 4; ++j) {
      const shifted = shiftRight(v.ref, j * Math.max(...orders));
      const tap = sl(taps.ref, [j, 0], [j + 1, cfg.d_model]).reshape([1, 1, cfg.d_model]);
      const ok = np.array(new Float32Array(Array.from({ length: tokens.shape[1] }, (_, t) => t >= j * maxOrder ? 1 : 0)), { dtype: D }).reshape([1, tokens.shape[1], 1]);
      vv = vv.add(shifted.mul(tap).mul(ok));
    }
    ks.push(k); vs.push(vv);
  }
  return { k: np.stack(ks, 0), v: np.stack(vs, 0) };
}

function attention(x, layer, cfg, w, cos, sin, causal) {
  const prefix = `stack/layers/block/`;
  const qProj = sl(w[`${prefix}self_attn/q_proj/kernel`], [layer, 0, 0], [layer + 1, cfg.d_model, cfg.attn_dim]).reshape([cfg.d_model, cfg.attn_dim]);
  const kv = cfg.num_kv_heads * (cfg.attn_dim / cfg.num_heads);
  const kProj = sl(w[`${prefix}self_attn/k_proj/kernel`], [layer, 0, 0], [layer + 1, cfg.d_model, kv]).reshape([cfg.d_model, kv]);
  const vProj = sl(w[`${prefix}self_attn/v_proj/kernel`], [layer, 0, 0], [layer + 1, cfg.d_model, kv]).reshape([cfg.d_model, kv]);
  const hd = cfg.attn_dim / cfg.num_heads;
  const qNorm = sl(w[`${prefix}self_attn/q_norm/scale`], [layer, 0], [layer + 1, hd]).reshape([hd]);
  const kNorm = sl(w[`${prefix}self_attn/k_norm/scale`], [layer, 0], [layer + 1, hd]).reshape([hd]);
  const q0 = linear(x.ref, qProj).reshape([x.shape[0], x.shape[1], cfg.num_heads, hd]).transpose([0, 2, 1, 3]);
  const k0 = linear(x.ref, kProj).reshape([x.shape[0], x.shape[1], cfg.num_kv_heads, hd]).transpose([0, 2, 1, 3]);
  const v0 = linear(x.ref, vProj).reshape([x.shape[0], x.shape[1], cfg.num_kv_heads, hd]).transpose([0, 2, 1, 3]);
  const q = rope(zcrmsNorm(q0, qNorm), cos, sin);
  let k = rope(zcrmsNorm(k0, kNorm), cos, sin);
  let v = v0;
  const repeat = cfg.num_heads / cfg.num_kv_heads;
  if (repeat > 1) { k = np.repeat(k, repeat, 1); v = np.repeat(v, repeat, 1); }
  let scores = np.matmul(q, k.transpose([0, 1, 3, 2])).div(Math.sqrt(hd));
  scores = np.where(causal.ref, scores, -1e30);
  const p = softmax(scores, -1);
  let out = np.matmul(p, v).transpose([0, 2, 1, 3]).reshape([x.shape[0], x.shape[1], cfg.attn_dim]);
  const gateKernel = sl(w[`${prefix}self_attn/gate_proj/kernel`], [layer, 0, 0], [layer + 1, cfg.d_model, cfg.attn_dim]).reshape([cfg.d_model, cfg.attn_dim]);
  out = out.mul(sigmoid(linear(x.ref, gateKernel)));
  const outKernel = sl(w[`${prefix}self_attn/out_proj/kernel`], [layer, 0, 0], [layer + 1, cfg.attn_dim, cfg.d_model]).reshape([cfg.attn_dim, cfg.d_model]);
  return linear(out, outKernel);
}

function hadamardMLP(x, layer, cfg, w, H) {
  const p = `stack/layers/block/hadamard_mlp/`;
  const d1 = sl(w[`${p}d1`], [layer, 0], [layer + 1, cfg.d_model]).reshape([cfg.d_model]);
  const d2 = sl(w[`${p}d2`], [layer, 0], [layer + 1, cfg.d_model]).reshape([cfg.d_model]);
  const d3 = sl(w[`${p}d3`], [layer, 0], [layer + 1, cfg.d_model]).reshape([cfg.d_model]);
  let z = np.matmul(x.mul(d1), H.ref);
  z = np.matmul(silu(z.mul(d2)), H.ref);
  return z.mul(d3);
}

function logsumexpAxis(x, axis) {
  const a = axis < 0 ? x.shape.length + axis : axis;
  const outShape = [...x.shape]; outShape[a] = 1;
  const m = np.max(x.ref, axis).reshape(outShape);
  const e = np.exp(x.sub(m.ref));
  const z = np.sum(e, axis).reshape(outShape);
  return m.add(np.log(z));
}
function sinkhorn(logits, iters = 20) {
  let x = logits;
  for (let i = 0; i < iters; ++i) {
    x = x.sub(logsumexpAxis(x.ref, -1));
    x = x.sub(logsumexpAxis(x.ref, -2));
  }
  return np.exp(x);
}

function forward(tokens, cfg, w) {
  const B = tokens.shape[0], T = tokens.shape[1], C = cfg.d_model, n = cfg.mhc_lanes;
  const embed = np.take(w['embedding/embedding'].ref, tokens.ref, 0).mul(Math.sqrt(C));
  const [cos, sin] = ropeFreqs(cfg.attn_dim / cfg.num_heads, T, cfg.rope_theta ?? 100000);
  const causal2 = np.equal(np.tril(np.ones([T, T])), 1).reshape([1, 1, T, T]);
  const maskKeep = np.ones([T], { dtype: D });
  const engram = makeEngramKV(tokens, maskKeep, cfg, w);
  let x = embed.reshape([B, T, 1, C]);
  x = np.tile(x, [1, 1, n, 1]);
  const H = walsh(512);
  const laneHost = Array.from({ length: n }, (_, i) => i);
  for (let layer = 0; layer < cfg.num_layers; ++layer) {
    const nx = rmsUnit(x.ref.reshape([B, T, n * C]));
    const phiPre = sl(w['stack/mhc_phi_pre'], [layer, 0, 0], [layer + 1, n * C, n]).reshape([n * C, n]);
    const phiPost = sl(w['stack/mhc_phi_post'], [layer, 0, 0], [layer + 1, n * C, n]).reshape([n * C, n]);
    const phiRes = sl(w['stack/mhc_phi_res'], [layer, 0, 0], [layer + 1, n * C, n * n]).reshape([n * C, n * n]);
    const aPre = sl(w['stack/mhc_a_pre'], [layer], [layer + 1]).reshape([]);
    const aPost = sl(w['stack/mhc_a_post'], [layer], [layer + 1]).reshape([]);
    const aRes = sl(w['stack/mhc_a_res'], [layer], [layer + 1]).reshape([]);
    const bPre = sl(w['stack/mhc_b_pre'], [layer, 0], [layer + 1, n]).reshape([n]);
    const bPost = sl(w['stack/mhc_b_post'], [layer, 0], [layer + 1, n]).reshape([n]);
    const bRes = sl(w['stack/mhc_b_res'], [layer, 0, 0], [layer + 1, n, n]).reshape([n, n]);
    const activeLane = layer % n;
    const preOff = np.array(Array.from({ length: n }, (_, i) => i === activeLane ? 4 : -4), { dtype: D });
    const postOff = np.array(Array.from({ length: n }, (_, i) => i === activeLane ? 0 : -4), { dtype: D });
    const hpre = sigmoid(np.add(np.multiply(aPre, np.einsum('btc,cn->btn', nx.ref, phiPre)), bPre).add(preOff));
    const u = np.einsum('btn,btnc->btc', hpre, x.ref.astype(D)).astype(D);

    let blockInput = u;
    if (engram) {
      const ek = engram.k;
      const ev = engram.v;
      const ux = rmsUnit(u.ref);
      const ex = rmsUnit(ek.ref);
      const alpha = sigmoid(np.einsum('btd,sbtd->sbt', ux, ex).div(Math.sqrt(C)));
      const flags = np.array(Array.from({ length: cfg.engram_layers.length }, (_, s) => cfg.engram_layers[s] === layer ? 1 : 0), { dtype: D });
      blockInput = u.ref.add(np.einsum('s,sbt,sbtd->btd', flags, alpha, ev.ref));
    }

    const preNorm = zcrmsNorm(blockInput.ref, sl(w['stack/layers/block/ZCRMSNorm_0/scale'], [layer, 0], [layer + 1, C]).reshape([C]));
    const attn = attention(preNorm, layer, cfg, w, cos, sin, causal2);
    const postNorm = zcrmsNorm(attn, sl(w['stack/layers/block/post_attn_norm/scale'], [layer, 0], [layer + 1, C]).reshape([C]));
    const attnGate = sigmoid(sl(w['stack/layers/block/attn_gate'], [layer], [layer + 1]).reshape([]));
    const afterAttn = blockInput.add(postNorm.mul(attnGate));
    const preH = zcrmsNorm(afterAttn.ref, sl(w['stack/layers/block/pre_hada_norm/scale'], [layer, 0], [layer + 1, C]).reshape([C]));
    const blockOutput = hadamardMLP(preH, layer, cfg, w, H).add(afterAttn);
    const y = blockOutput.sub(u.ref);

    const hpost = sigmoid(np.add(np.multiply(aPost, np.einsum('btc,cn->btn', nx.ref, phiPost)), bPost).add(postOff)).mul(2);
    const res = np.einsum('btc,cn->btn', nx, phiRes);
    const hres = sinkhorn(res.mul(aRes).reshape([B, T, n, n]).add(bRes));
    const xf = x.astype(D);
    const mixed = np.einsum('btij,btjc->btic', hres, xf);
    x = mixed.add(np.einsum('btn,btc->btnc', hpost, y)).astype(D);
    void laneHost;
  }
  x = np.mean(x, 2);
  x = zcrmsNorm(x, w['stack/final_norm/scale']);
  return linear(x, np.transpose(w['embedding/embedding'], [1, 0]));
}

async function main() {
  const { header, weights } = readWeights(process.argv[2] || 'weights.bin');
  const cfg = normalizeConfig(header.config);
  const tokensArg = process.argv.find(x => x.startsWith('--tokens='));
  const tokens = tokensArg
    ? tokensArg.slice('--tokens='.length).split(',').filter(Boolean).map(Number)
    : (header.input_tokens || [1, 2, 3, 4]);
  const backend = (await init('wasm')).includes('wasm') ? 'wasm' : 'cpu';
  defaultDevice(backend);
  const tokenArray = np.array(Int32Array.from(tokens), { dtype: np.int32 }).reshape([1, tokens.length]);
  const logits = forward(tokenArray, cfg, weights);
  const out = logits.dataSync();
  let maxAbs = null, maxRel = null, rmse = null, cosine = null;
  if (header.reference) {
    const ref = header.reference;
    maxAbs = 0; maxRel = 0;
    let se = 0, dot = 0, na = 0, nb = 0;
    for (let i = 0; i < out.length; ++i) {
      const a = out[i], b = ref[i];
      const d = Math.abs(a - b);
      const r = d / Math.max(1e-6, Math.abs(b));
      if (d > maxAbs) maxAbs = d;
      if (r > maxRel) maxRel = r;
      se += (a - b) * (a - b); dot += a * b; na += a * a; nb += b * b;
    }
    rmse = Math.sqrt(se / out.length);
    cosine = dot / Math.sqrt(na * nb);
  }
  const final = Array.from(out.slice((tokens.length - 1) * cfg.vocab_size, tokens.length * cfg.vocab_size));
  let top = final.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
  console.log(`backend:    ${backend}`);
  console.log(`tokens:     ${tokens.length}`);
  console.log(`logits:     ${JSON.stringify(logits.shape)}`);
  console.log(`top-5:      ${JSON.stringify(top)}`);
  if (maxAbs !== null) console.log(`max_abs:    ${maxAbs}`);
  if (rmse !== null) console.log(`rmse:       ${rmse}`);
  if (cosine !== null) console.log(`cosine:     ${cosine}`);
  if (maxRel !== null) console.log(`max_rel:    ${maxRel}`);
}

main().catch(err => { console.error(err); process.exit(1); });














