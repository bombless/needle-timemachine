import { init, defaultDevice, numpy as np } from '@jax-js/jax';

const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node);
const fs = isNode ? (await import('node:fs')).default : null;

const MAGIC = 'NEEDLEJS1';
const HEADER_BYTES = 4;
const CACT_TAG = 0x05E12A83;
const CACT_HDR_BYTES = 120;
const CACT_REC_BYTES = 44;
const D = np.float32;

const HELP = `Usage: node forward.js [weights.bin] [options]

Run the Needle forward/verification path and print the final-token logits.

Options:
  --tokens=<ids>          Input token ids, as JSON or whitespace/comma-separated ids
  --prefill-file=<path>   Read input token ids from a JSON/text file
  --cact=<path>           Load quantized weights from a .cact export instead of weights.bin
  --cact <path>           Same as --cact=<path>
  --help                  Show this help message

If neither --tokens nor --prefill-file is provided, input_tokens from the
weights header are used, falling back to [1, 2, 3, 4].

The optional positional argument selects the weights file and defaults to
weights.bin. --cact switches the source to the Cactus quantized export.
`;

function normalizeConfig(c) {
  const numeric = ['vocab_size','d_model','attn_dim','num_heads','num_kv_heads','num_layers','max_seq_len','pad_token_id','contrastive_dim','rope_theta','engram_heads','engram_slots','kv_window','kv_bits','act_bits','scan_unroll'];
  const out = { ...c };
  for (const k of numeric) if (k in out) out[k] = Number(out[k]);
  out.engram_orders = (out.engram_orders ?? [2,3]).map(Number);
  out.engram_layers = (out.engram_layers ?? [2,15]).map(Number);
  out.flash = out.flash === true || out.flash === 'True' || out.flash === 'true';
  out.remat = out.remat === true || out.remat === 'True' || out.remat === 'true';
  return out;
}

function parseTokenList(value, source = 'tokens') {
  let parsed = value;
  if (!Array.isArray(value)) {
    const text = String(value).trim();
    if (text.startsWith('[')) {
      try { parsed = JSON.parse(text); }
      catch (err) { throw new Error(`invalid ${source} JSON: ${err.message}`); }
    } else {
      parsed = text.split(/[\s,]+/).filter(Boolean);
    }
  }
  if (!Array.isArray(parsed)) throw new Error(`${source} must be an array of token ids`);
  const tokens = parsed.map((item, i) => {
    const id = item && typeof item === 'object' ? item.token_id : item;
    const n = Number(id);
    if (!Number.isInteger(n) || n < 0) throw new Error(`invalid token id at ${source}[${i}]: ${id}`);
    return n;
  });
  if (!tokens.length) throw new Error(`${source} must contain at least one token`);
  return tokens;
}

function readPrefill(path) {
  let value;
  try { value = JSON.parse(fs.readFileSync(path, 'utf8')); }
  catch (err) { throw new Error(`unable to read prefill file ${path}: ${err.message}`); }
  return parseTokenList(value, path);
}

function loadTokenMetadata(header) {
  if (Array.isArray(header.token_metadata)) return header.token_metadata;
  const vocabPath = process.env.NEEDLE_TOKENIZER_VOCAB || 'needle/needle/model/tokenizer.vocab';
  try {
    const lines = fs.readFileSync(vocabPath, 'utf8').split(/\r?\n/);
    return lines.filter(line => line.length > 0).map((line, tokenId) => {
      const piece = line.split('\t', 1)[0];
      const byteMatch = piece.match(/^<0x([0-9A-Fa-f]{2})>$/);
      const tokenText = byteMatch
        ? String.fromCodePoint(parseInt(byteMatch[1], 16))
        : piece.replaceAll(String.fromCodePoint(0x2581), ' ');
      return {
        token_id: tokenId,
        token_text: tokenText,
        token_bytes_hex: Buffer.from(tokenText, 'utf8').toString('hex').match(/../g)?.join(' ') || '',
      };
    });
  } catch (_) {
    return [];
  }
}

function readWeights(path = 'weights.bin') {
  if (!fs) throw new Error('readWeights is only available in Node.js');
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

function halfToFloat(h) {
  const s = (h >>> 15) & 1, e = (h >>> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return (s ? -1 : 1) * (f ? Math.pow(2, -14) * (f / 1024) : 0);
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function cactCodebook(codebook, bits, group) {
  if (bits === 5) {
    const c = 1.2240064 / Math.sqrt(group);
    return new Float32Array([-c, 0, c]);
  }
  const start = bits === 2 ? 0 : bits === 3 ? 4 : bits === 4 ? 12 : -1;
  const len = 1 << bits;
  if (start < 0 || start + len > codebook.length) throw new Error(`unsupported .cact CQ bits=${bits}`);
  return codebook.subarray(start, start + len);
}

function unpackCactIndex(bytes, byteBase, bits, k) {
  if (bits === 5) {
    const crumb = (bytes[byteBase + (k >> 2)] >>> ((k & 3) * 2)) & 3;
    return crumb === 3 ? 0 : crumb + 1;
  }
  const chunk = Math.floor(k / 8), inChunk = k & 7;
  let word = 0;
  const p = byteBase + chunk * bits;
  for (let b = 0; b < bits; ++b) word |= bytes[p + b] << (8 * b);
  return (word >>> (inChunk * bits)) & ((1 << bits) - 1);
}

function fwhtInPlace(a) {
  for (let h = 1; h < a.length; h <<= 1) {
    for (let i = 0; i < a.length; i += h << 1) {
      for (let j = 0; j < h; ++j) {
        const x = a[i + j], y = a[i + j + h];
        a[i + j] = x + y;
        a[i + j + h] = x - y;
      }
    }
  }
  const scale = 1 / Math.sqrt(a.length);
  for (let i = 0; i < a.length; ++i) a[i] *= scale;
}

function decodeCactCQ(bytes, shape, offset, nbytes, group, bits, codebook) {
  const [out, inDim] = shape;
  const inPad = Math.ceil(inDim / group) * group;
  const packedPerRow = bits === 5 ? inPad / 4 : inPad * bits / 8;
  const packedBytes = out * packedPerRow;
  if (!Number.isInteger(packedBytes) || packedBytes > nbytes) throw new Error(`invalid .cact CQ payload for ${out}x${inDim}`);
  const normBase = offset + packedBytes;
  const result = new Float32Array(out * inDim);
  const cb = cactCodebook(codebook, bits, group);
  const tmp = new Float32Array(group);
  for (let r = 0; r < out; ++r) {
    const rowPacked = offset + r * packedPerRow;
    const rowNorm = normBase + r * (inPad / group) * 2;
    for (let g = 0; g < inPad / group; ++g) {
      const norm = halfToFloat(bytes[rowNorm + g * 2] | (bytes[rowNorm + g * 2 + 1] << 8));
      const baseK = g * group;
      for (let j = 0; j < group; ++j) tmp[j] = cb[unpackCactIndex(bytes, rowPacked, bits, baseK + j)] * norm;
      fwhtInPlace(tmp);
      const keep = Math.min(group, inDim - baseK);
      if (keep > 0) result.set(tmp.subarray(0, keep), r * inDim + baseK);
    }
  }
  return result;
}

function readCact(path) {
  if (!fs) throw new Error('readCact is only available in Node.js');
  const buf = fs.readFileSync(path);
  if (buf.length < CACT_HDR_BYTES) throw new Error(`.cact file is too small: ${path}`);
  const tag = buf.readUInt32LE(0);
  if (tag !== CACT_TAG) throw new Error(`bad .cact tag: 0x${tag.toString(16)}`);
  const numTensors = buf.readUInt32LE(4);
  const codebookLen = buf.readUInt32LE(8);
  const header = {
    kv_window: buf.readUInt32LE(12), kv_bits: buf.readUInt32LE(16),
    vocab_size: buf.readUInt32LE(20), d_model: buf.readUInt32LE(24),
    num_heads: buf.readUInt32LE(28), num_kv_heads: buf.readUInt32LE(32),
    num_layers: buf.readUInt32LE(36), head_dim: buf.readUInt32LE(40),
    max_seq_len: buf.readUInt32LE(44), hada_n: buf.readUInt32LE(48),
    mhc_lanes: buf.readUInt32LE(52), engram_slots: buf.readUInt32LE(56),
    engram_sub_dim: buf.readUInt32LE(60), num_engram_tables: buf.readUInt32LE(64),
    engram_conv_taps: buf.readUInt32LE(68), engram_conv_dilation: buf.readUInt32LE(72),
  };
  const numOrders = buf.readUInt32LE(76);
  header.engram_orders = Array.from({ length: numOrders }, (_, i) => buf.readUInt32LE(80 + i * 4));
  const numSites = buf.readUInt32LE(96);
  header.engram_layers = Array.from({ length: numSites }, (_, i) => buf.readUInt32LE(100 + i * 4));
  header.rope_theta = buf.readFloatLE(116);
  if (codebookLen !== 28) throw new Error(`unsupported .cact codebook length: ${codebookLen}`);
  const codebook = new Float32Array(buf.buffer, buf.byteOffset + CACT_HDR_BYTES, codebookLen);
  const recordsStart = CACT_HDR_BYTES + codebookLen * 4;
  if (recordsStart + numTensors * CACT_REC_BYTES > buf.length) throw new Error('truncated .cact tensor directory');
  const tensors = [];
  for (let i = 0; i < numTensors; ++i) {
    const p = recordsStart + i * CACT_REC_BYTES;
    const dtype = buf.readUInt8(p), ndim = buf.readUInt8(p + 1);
    if (ndim > 4) throw new Error(`invalid .cact tensor ndim=${ndim} at index ${i}`);
    const shape = [];
    for (let d = 0; d < ndim; ++d) shape.push(buf.readUInt32LE(p + 4 + d * 4));
    const offset = Number(buf.readBigUInt64LE(p + 20));
    const nbytes = Number(buf.readBigUInt64LE(p + 28));
    const group = buf.readUInt32LE(p + 36), bits = buf.readUInt32LE(p + 40);
    if (offset + nbytes > buf.length) throw new Error(`truncated .cact tensor ${i}`);
    let value;
    if (dtype === 3) value = decodeCactCQ(buf, shape, offset, nbytes, group, bits, codebook);
    else if (dtype === 1) {
      const n = shape.reduce((a, b) => a * b, 1), out = new Float32Array(n);
      for (let j = 0; j < n; ++j) out[j] = halfToFloat(buf.readUInt16LE(offset + j * 2));
      value = out;
    } else if (dtype === 2) {
      value = new Float32Array(buf.buffer, buf.byteOffset + offset, nbytes / 4).slice();
    } else if (dtype === 4) value = buf.subarray(offset, offset + nbytes);
    else throw new Error(`unsupported .cact dtype=${dtype} at index ${i}`);
    tensors.push({ dtype, shape, value });
  }
  if (tensors.length !== 405) throw new Error(`unexpected .cact tensor count ${tensors.length}`);

  const names = ['embedding'];
  for (let l = 0; l < header.num_layers; ++l) names.push(
    `layer${l}.norm_in`, `layer${l}.q_proj`, `layer${l}.k_proj`, `layer${l}.v_proj`,
    `layer${l}.q_norm`, `layer${l}.k_norm`, `layer${l}.gate_proj`, `layer${l}.out_proj`,
    `layer${l}.post_norm`, `layer${l}.attn_gate`, `layer${l}.pre_hada`,
    `layer${l}.d1`, `layer${l}.d2`, `layer${l}.d3`,
  );
  names.push('mhc_a_pre','mhc_a_post','mhc_a_res','mhc_b_pre','mhc_b_post','mhc_b_res','mhc_phi_pre','mhc_phi_post','mhc_phi_res');
  for (let s = 0; s < header.engram_layers.length; ++s) names.push(`engram${s}.tables`,`engram${s}.key_proj`,`engram${s}.value_proj`,`engram${s}.taps`);
  names.push('final_norm','heads.manifest','contrastive_head.probes','contrastive_head.proj','contrastive_head.bias','confidence_head.probes','confidence_head.proj','confidence_head.bias','tokenizer');
  if (names.length !== tensors.length) throw new Error(`unsupported .cact layout: expected ${names.length} records, got ${tensors.length}`);

  const weights = {};
  const add = (name, t, shape = t.shape) => { if (t.dtype === 4) return; weights[name] = np.array(t.value, { dtype: D }).reshape(shape); };
  add('embedding/embedding', tensors[0], tensors[0].shape);
  let k = 1;
  for (let l = 0; l < header.num_layers; ++l) {
    const p = `stack/layers/block/`;
    add(`${p}ZCRMSNorm_0/scale`, tensors[k++], [1, header.d_model]);
    for (const q of ['q_proj','k_proj','v_proj']) { const t=tensors[k++]; add(`${p}self_attn/${q}/kernel`, t, [1, t.shape[1], t.shape[0]]); }
    add(`${p}self_attn/q_norm/scale`, tensors[k++], [1, header.head_dim]);
    add(`${p}self_attn/k_norm/scale`, tensors[k++], [1, header.head_dim]);
    for (const q of ['gate_proj','out_proj']) { const t=tensors[k++]; add(`${p}self_attn/${q}/kernel`, t, [1, t.shape[1], t.shape[0]]); }
    add(`${p}post_attn_norm/scale`, tensors[k++], [1, header.d_model]);
    add(`${p}attn_gate`, tensors[k++], [1]);
    add(`${p}pre_hada_norm/scale`, tensors[k++], [1, header.d_model]);
    for (const q of ['d1','d2','d3']) add(`${p}hadamard_mlp/${q}`, tensors[k++], [1, header.d_model]);
  }
  for (const q of ['mhc_a_pre','mhc_a_post','mhc_a_res','mhc_b_pre','mhc_b_post','mhc_b_res']) {
    const t=tensors[k++]; add(`stack/${q}`, t, t.shape);
  }
  for (const q of ['mhc_phi_pre','mhc_phi_post','mhc_phi_res']) {
    const t=tensors[k++]; const [rows,nC]=t.shape; const lanes=q==='mhc_phi_res'?header.mhc_lanes*header.mhc_lanes:header.mhc_lanes;
    add(`stack/${q}`, t, [header.num_layers,nC,lanes]);
  }
  for (let s = 0; s < header.engram_layers.length; ++s) {
    const t=tensors[k++]; add(`engrams_${s}/embedding`, t, [header.num_engram_tables,header.engram_slots,header.engram_sub_dim]);
    for (const q of ['key_proj','value_proj']) { const u=tensors[k++]; add(`engrams_${s}/${q}/kernel`, u, [u.shape[1],u.shape[0]]); }
    add(`engrams_${s}/taps`, tensors[k++], tensors[k-1].shape);
  }
  add('stack/final_norm/scale', tensors[k++], tensors[k-1].shape);
  return { header, weights };
}

function decodeBase64F32(data) {
  if (typeof atob === 'function') {
    const binary = atob(data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; ++i) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  return new Float32Array(Uint8Array.from(Buffer.from(data, 'base64')).buffer);
}

function payloadTensorName(name) {
  return String(name).replace(/^engrams\.(\d+)(?=\.)/, 'engrams_$1').replaceAll('.', '/');
}

function weightsFromPayload(payload) {
  if (!payload || payload.format !== 'needle-timemachine.weights/v1') throw new Error('unsupported weights payload');
  const weights = {};
  for (const tensor of payload.tensors || []) {
    if (tensor.encoding !== 'base64-f32-le') throw new Error(`unsupported tensor encoding: ${tensor.encoding}`);
    const raw = decodeBase64F32(tensor.data);
    weights[payloadTensorName(tensor.name)] = np.array(raw, { dtype: D }).reshape(tensor.shape);
  }
  return weights;
}

function sl(x, starts, ends) { return x.ref.slice(...starts.map((start, i) => [start, ends[i]])); }
function scalar(x) { return np.array(new Float32Array([x]), { dtype: D }).reshape([]); }
function sigmoid(x) { return np.divide(1, np.add(1, np.exp(np.negative(x)))); }
function silu(x) { return x.mul(sigmoid(x.ref)); }
function rmsUnit(x, eps = 1e-6) {
  const xf = x.astype(D), sq = xf.ref.mul(xf.ref), mean = np.mean(sq, -1).reshape([...sq.shape.slice(0, -1), 1]);
  return xf.mul(np.reciprocal(np.sqrt(np.add(mean, scalar(eps)))));
}
function zcrmsNorm(x, scale, eps = 1e-6) { return np.multiply(np.add(1, scale), rmsUnit(x, eps)); }
function softmax(x, axis = -1) {
  const a = axis < 0 ? x.shape.length + axis : axis, outShape = [...x.shape]; outShape[a] = 1;
  const m = np.max(x.ref, axis).reshape(outShape), e = np.exp(x.sub(m)), z = np.sum(e.ref, axis).reshape(outShape);
  return e.div(z);
}
function linear(x, kernel) { return np.matmul(x, kernel); }
function shiftRight(x, offset, axis = 1) {
  if (offset === 0) return x;
  const shape = [...x.shape]; if (axis !== 1) throw new Error('shiftRight currently expects sequence axis=1');
  if (offset >= shape[axis]) return np.zeros(shape, { dtype: x.dtype });
  const zshape = [...shape]; zshape[axis] = offset;
  return np.concatenate([np.zeros(zshape, { dtype: x.dtype }), sl(x, [0, 0], [shape[0], shape[1] - offset])], 1);
}
function rope(x, cos, sin) {
  const h = x.shape.at(-1) / 2;
  const x1 = sl(x.ref, [0,0,0,0], [x.shape[0],x.shape[1],x.shape[2],h]), x2 = sl(x.ref, [0,0,0,h], [x.shape[0],x.shape[1],x.shape[2],x.shape[3]]);
  const c = sl(cos.ref,[0,0],[x.shape[2],h]).reshape([1,1,x.shape[2],h]), s = sl(sin.ref,[0,0],[x.shape[2],h]).reshape([1,1,x.shape[2],h]);
  return np.concatenate([x1.ref.mul(c.ref).sub(x2.ref.mul(s.ref)), x2.mul(c).add(x1.mul(s))], -1);
}
function ropeFreqs(headDim, seqLen, theta) {
  const half = Math.floor(headDim / 2), c = new Float32Array(seqLen*half), s = new Float32Array(seqLen*half);
  for (let t=0;t<seqLen;++t) for (let j=0;j<half;++j) { const a=t/Math.pow(theta,(2*j)/headDim); c[t*half+j]=Math.cos(a); s[t*half+j]=Math.sin(a); }
  return [np.array(c,{dtype:D}).reshape([seqLen,half]),np.array(s,{dtype:D}).reshape([seqLen,half])];
}
function walsh(n) {
  let h=[[1]];
  while(h.length<n){const m=h.length,next=Array.from({length:m*2},()=>new Array(m*2));for(let r=0;r<m;++r)for(let c=0;c<m;++c){const v=h[r][c];next[r][c]=v;next[r][c+m]=v;next[r+m][c]=v;next[r+m][c+m]=-v;}h=next;}
  const a=new Float32Array(n*n),s=Math.sqrt(n);for(let i=0;i<n;++i)for(let j=0;j<n;++j)a[i*n+j]=h[i][j]/s;return np.array(a,{dtype:D}).reshape([n,n]);
}
function engramGeometry(cfg){const orders=cfg.engram_orders??[2,3],heads=cfg.engram_heads||Math.max(1,Math.floor(cfg.d_model/(orders.length*128))),subDim=Math.floor(cfg.d_model/(orders.length*heads));return{orders,heads,subDim};}
function engramIndices(tokens,orders,heads,slots){const B=tokens.shape[0],T=tokens.shape[1],host=tokens.dataSync(),out=[],SEED=0x9E3779B9>>>0,PRIME=0x01000193>>>0;for(let oi=0;oi<orders.length;++oi)for(let h=0;h<heads;++h){const seed=Math.imul(SEED,oi*heads+h+1)>>>0,a=new Int32Array(B*T);for(let b=0;b<B;++b)for(let t=0;t<T;++t){let acc=seed;for(let j=0;j<orders[oi];++j){const u=j<=t?(host[b*T+t-j]>>>0):0;acc=Math.imul((acc^u)>>>0,PRIME)>>>0;}acc=(acc^(acc>>>15))>>>0;a[b*T+t]=acc%slots;}out.push(np.array(a,{dtype:np.int32}).reshape([B,T]));}return out;}
function makeEngramKV(tokens,maskKeep,cfg,w){if(!cfg.engram_layers?.length)return null;const{orders,heads,subDim}=engramGeometry(cfg),idx=engramIndices(tokens,orders,heads,cfg.engram_slots),numTables=orders.length*heads,embeddings=[];for(let s=0;s<cfg.engram_layers.length;++s){const table=w[`engrams_${s}/embedding`],fetched=[];for(let j=0;j<numTables;++j){const one=sl(table.ref,[j,0,0],[j+1,table.shape[1],table.shape[2]]).reshape([table.shape[1],table.shape[2]]),gathered=np.take(one,idx[j].ref,0),order=orders[Math.floor(j/heads)],ok=np.array(new Float32Array(Array.from({length:tokens.shape[1]},(_,t)=>t>=order-1?1:0)),{dtype:D}).reshape([1,tokens.shape[1],1]);fetched.push(gathered.mul(ok));}let e=np.stack(fetched,2).reshape([tokens.shape[0],tokens.shape[1],numTables*subDim]);e=e.mul(maskKeep.ref.reshape([1,tokens.shape[1],1]));embeddings.push(e);}const ks=[],vs=[],maxOrder=Math.max(...orders);for(let s=0;s<cfg.engram_layers.length;++s){const e=embeddings[s];let k=linear(e.ref,w[`engrams_${s}/key_proj/kernel`]),v=linear(e,w[`engrams_${s}/value_proj/kernel`]);const taps=w[`engrams_${s}/taps`];let vv=np.zeros(v.shape,{dtype:D});for(let j=0;j<4;++j){const shifted=shiftRight(v.ref,j*maxOrder),tap=sl(taps.ref,[j,0],[j+1,cfg.d_model]).reshape([1,1,cfg.d_model]),ok=np.array(new Float32Array(Array.from({length:tokens.shape[1]},(_,t)=>t>=j*maxOrder?1:0)),{dtype:D}).reshape([1,tokens.shape[1],1]);vv=vv.add(shifted.mul(tap).mul(ok));}ks.push(k);vs.push(vv);}return{k:np.stack(ks,0),v:np.stack(vs,0)};}
function attention(x,layer,cfg,w,cos,sin,causal){const prefix='stack/layers/block/',qProj=sl(w[`${prefix}self_attn/q_proj/kernel`],[layer,0,0],[layer+1,cfg.d_model,cfg.attn_dim]).reshape([cfg.d_model,cfg.attn_dim]),kv=cfg.num_kv_heads*(cfg.attn_dim/cfg.num_heads),kProj=sl(w[`${prefix}self_attn/k_proj/kernel`],[layer,0,0],[layer+1,cfg.d_model,kv]).reshape([cfg.d_model,kv]),vProj=sl(w[`${prefix}self_attn/v_proj/kernel`],[layer,0,0],[layer+1,cfg.d_model,kv]).reshape([cfg.d_model,kv]),hd=cfg.attn_dim/cfg.num_heads,qNorm=sl(w[`${prefix}self_attn/q_norm/scale`],[layer,0],[layer+1,hd]).reshape([hd]),kNorm=sl(w[`${prefix}self_attn/k_norm/scale`],[layer,0],[layer+1,hd]).reshape([hd]),q0=linear(x.ref,qProj).reshape([x.shape[0],x.shape[1],cfg.num_heads,hd]).transpose([0,2,1,3]),k0=linear(x.ref,kProj).reshape([x.shape[0],x.shape[1],cfg.num_kv_heads,hd]).transpose([0,2,1,3]),v0=linear(x.ref,vProj).reshape([x.shape[0],x.shape[1],cfg.num_kv_heads,hd]).transpose([0,2,1,3]),q=rope(zcrmsNorm(q0,qNorm),cos,sin);let kk=rope(zcrmsNorm(k0,kNorm),cos,sin),v=v0;const repeat=cfg.num_heads/cfg.num_kv_heads;if(repeat>1){kk=np.repeat(kk,repeat,1);v=np.repeat(v,repeat,1);}let scores=np.matmul(q,kk.transpose([0,1,3,2])).div(Math.sqrt(hd));scores=np.where(causal.ref,scores,-1e30);const p=softmax(scores,-1);let out=np.matmul(p,v).transpose([0,2,1,3]).reshape([x.shape[0],x.shape[1],cfg.attn_dim]);const gateKernel=sl(w[`${prefix}self_attn/gate_proj/kernel`],[layer,0,0],[layer+1,cfg.d_model,cfg.attn_dim]).reshape([cfg.d_model,cfg.attn_dim]);out=out.mul(sigmoid(linear(x.ref,gateKernel)));const outKernel=sl(w[`${prefix}self_attn/out_proj/kernel`],[layer,0,0],[layer+1,cfg.attn_dim,cfg.d_model]).reshape([cfg.attn_dim,cfg.d_model]);return linear(out,outKernel);}
function hadamardMLP(x,layer,cfg,w,H){const p='stack/layers/block/hadamard_mlp/',d1=sl(w[`${p}d1`],[layer,0],[layer+1,cfg.d_model]).reshape([cfg.d_model]),d2=sl(w[`${p}d2`],[layer,0],[layer+1,cfg.d_model]).reshape([cfg.d_model]),d3=sl(w[`${p}d3`],[layer,0],[layer+1,cfg.d_model]).reshape([cfg.d_model]);let z=np.matmul(x.mul(d1),H.ref);z=np.matmul(silu(z.mul(d2)),H.ref);return z.mul(d3);}
function logsumexpAxis(x,axis){const a=axis<0?x.shape.length+axis:axis,outShape=[...x.shape];outShape[a]=1;const m=np.max(x,axis).reshape(outShape),e=np.exp(x.sub(m.ref)),z=np.sum(e,axis).reshape(outShape);return m.add(np.log(z));}
function sinkhorn(logits,iters=20){let x=logits;for(let i=0;i<iters;++i){x=x.sub(logsumexpAxis(x.ref,-1));x=x.sub(logsumexpAxis(x.ref,-2));}return np.exp(x);}
function forward(tokens,cfg,w){const B=tokens.shape[0],T=tokens.shape[1],C=cfg.d_model,n=cfg.mhc_lanes,embed=np.take(w['embedding/embedding'].ref,tokens.ref,0).mul(Math.sqrt(C)),[cos,sin]=ropeFreqs(cfg.attn_dim/cfg.num_heads,T,cfg.rope_theta??100000),causal2=np.equal(np.tril(np.ones([T,T])),1).reshape([1,1,T,T]),maskKeep=np.ones([T],{dtype:D}),engram=makeEngramKV(tokens,maskKeep,cfg,w);let x=embed.reshape([B,T,1,C]);x=np.tile(x,[1,1,n,1]);const H=walsh(512);for(let layer=0;layer<cfg.num_layers;++layer){const nx=rmsUnit(x.ref.reshape([B,T,n*C])),phiPre=sl(w['stack/mhc_phi_pre'],[layer,0,0],[layer+1,n*C,n]).reshape([n*C,n]),phiPost=sl(w['stack/mhc_phi_post'],[layer,0,0],[layer+1,n*C,n]).reshape([n*C,n]),phiRes=sl(w['stack/mhc_phi_res'],[layer,0,0],[layer+1,n*C,n*n]).reshape([n*C,n*n]),aPre=sl(w['stack/mhc_a_pre'],[layer],[layer+1]).reshape([]),aPost=sl(w['stack/mhc_a_post'],[layer],[layer+1]).reshape([]),aRes=sl(w['stack/mhc_a_res'],[layer],[layer+1]).reshape([]),bPre=sl(w['stack/mhc_b_pre'],[layer,0],[layer+1,n]).reshape([n]),bPost=sl(w['stack/mhc_b_post'],[layer,0],[layer+1,n]).reshape([n]),bRes=sl(w['stack/mhc_b_res'],[layer,0,0],[layer+1,n,n]).reshape([n,n]),activeLane=layer%n,preOff=np.array(Array.from({length:n},(_,i)=>i===activeLane?4:-4),{dtype:D}),postOff=np.array(Array.from({length:n},(_,i)=>i===activeLane?0:-4),{dtype:D}),hpre=sigmoid(np.add(np.multiply(aPre,np.einsum('btc,cn->btn',nx.ref,phiPre)),bPre).add(preOff)),u=np.einsum('btn,btnc->btc',hpre,x.ref.astype(D)).astype(D);let blockInput=u;if(engram){const ux=rmsUnit(u.ref),ex=rmsUnit(engram.k.ref),alpha=sigmoid(np.einsum('btd,sbtd->sbt',ux,ex).div(Math.sqrt(C))),flags=np.array(Array.from({length:cfg.engram_layers.length},(_,s)=>cfg.engram_layers[s]===layer?1:0),{dtype:D});blockInput=u.ref.add(np.einsum('s,sbt,sbtd->btd',flags,alpha,engram.v.ref));}const preNorm=zcrmsNorm(blockInput.ref,sl(w['stack/layers/block/ZCRMSNorm_0/scale'],[layer,0],[layer+1,C]).reshape([C])),attn=attention(preNorm,layer,cfg,w,cos,sin,causal2),postNorm=zcrmsNorm(attn,sl(w['stack/layers/block/post_attn_norm/scale'],[layer,0],[layer+1,C]).reshape([C])),attnGate=sigmoid(sl(w['stack/layers/block/attn_gate'],[layer],[layer+1]).reshape([])),afterAttn=blockInput.add(postNorm.mul(attnGate)),preH=zcrmsNorm(afterAttn.ref,sl(w['stack/layers/block/pre_hada_norm/scale'],[layer,0],[layer+1,C]).reshape([C])),blockOutput=hadamardMLP(preH,layer,cfg,w,H).add(afterAttn),y=blockOutput.sub(u.ref),hpost=sigmoid(np.add(np.multiply(aPost,np.einsum('btc,cn->btn',nx.ref,phiPost)),bPost).add(postOff)).mul(2),res=np.einsum('btc,cn->btn',nx,phiRes),hres=sinkhorn(res.mul(aRes).reshape([B,T,n,n]).add(bRes)),xf=x.astype(D),mixed=np.einsum('btij,btjc->btic',hres,xf);x=mixed.add(np.einsum('btn,btc->btnc',hpost,y)).astype(D);}x=np.mean(x,2);x=zcrmsNorm(x,w['stack/final_norm/scale']);return linear(x,np.transpose(w['embedding/embedding'],[1,0]));}

export { forward, normalizeConfig, weightsFromPayload, readCact, init, defaultDevice, np };

async function main(){const startTime=+new Date,args=process.argv.slice(2);if(args.includes('--help')||args.includes('-h')){console.log(HELP.trimEnd());return;}console.log(HELP.trimEnd());const cactAt=args.indexOf('--cact'),cactEq=args.find(x=>x.startsWith('--cact='));if(cactAt>=0&&cactEq)throw new Error('use either --cact=<path> or --cact <path>');const cactPath=cactEq?cactEq.slice('--cact='.length):cactAt>=0?args[cactAt+1]:null;if(cactAt>=0&&!cactPath)throw new Error('--cact requires a path');if(cactAt>=0&&cactAt+1<args.length&&args[cactAt+1].startsWith('-'))throw new Error('--cact requires a path');const positional=args.find((x,i)=>!x.startsWith('-')&&!(i>0&&args[i-1]==='--cact'));const loaded=cactPath?readCact(cactPath):readWeights(positional||'weights.bin'),header=loaded.header,weights=loaded.weights,cfg=normalizeConfig(header.config??header),tokensArg=args.find(x=>x.startsWith('--tokens=')),prefillArg=args.find(x=>x.startsWith('--prefill-file='));if(tokensArg&&prefillArg)throw new Error('use either --tokens or --prefill-file, not both');const tokens=prefillArg?readPrefill(prefillArg.slice('--prefill-file='.length)):tokensArg?parseTokenList(tokensArg.slice('--tokens='.length),'--tokens'):(header.input_tokens||[1,2,3,4]);const backend=(await init('wasm')).includes('wasm')?'wasm':'cpu';defaultDevice(backend);const tokenArray=np.array(Int32Array.from(tokens),{dtype:np.int32}).reshape([1,tokens.length]),logits=forward(tokenArray,cfg,weights),out=logits.dataSync();let maxAbs=null,maxRel=null,rmse=null,cosine=null;if(header.reference){const ref=header.reference;maxAbs=0;maxRel=0;let se=0,dot=0,na=0,nb=0;for(let i=0;i<out.length;++i){const a=out[i],b=ref[i],d=Math.abs(a-b),r=d/Math.max(1e-6,Math.abs(b));if(d>maxAbs)maxAbs=d;if(r>maxRel)maxRel=r;se+=(a-b)*(a-b);dot+=a*b;na+=a*a;nb+=b*b;}rmse=Math.sqrt(se/out.length);cosine=dot/Math.sqrt(na*nb);}const final=Array.from(out.slice((tokens.length-1)*cfg.vocab_size,tokens.length*cfg.vocab_size));const tokenMetadata=loadTokenMetadata(header);const top=final.map((v,i)=>{const metadata=tokenMetadata[i]||{};return{token_id:i,token_text:metadata.token_text||'',token_bytes_hex:metadata.token_bytes_hex||'',logit:v};}).sort((a,b)=>b.logit-a.logit).slice(0,5);console.log(`backend:    ${backend}`);console.log(`weights:    ${cactPath?'cact':'weights.bin'}`);console.log(`tokens:     ${tokens.length}`);console.log(`logits:     ${JSON.stringify(logits.shape)}`);console.log(`top-5:      ${JSON.stringify(top)}`);if(maxAbs!==null)console.log(`max_abs:    ${maxAbs}`);if(rmse!==null)console.log(`rmse:       ${rmse}`);if(cosine!==null)console.log(`cosine:     ${cosine}`);if(maxRel!==null)console.log(`max_rel:    ${maxRel}`);console.log('time',+new Date-startTime,'ms')}

const invokedAsCli=isNode&&process.argv[1]&&process.argv[1].replaceAll('\\','/').endsWith('/forward.js');if(invokedAsCli)main().catch(err=>{console.error(err);process.exit(1);});