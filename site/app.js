import { createKVCache, forwardWithKVCache, normalizeConfig, init, defaultDevice, np } from '../forward.js'

const $ = id => document.getElementById(id)
const status = message => { $('status').textContent = message }

function parseWeights (buffer) {
  const bytes = new Uint8Array(buffer)
  if (new TextDecoder().decode(bytes.subarray(0, 9)) !== 'NEEDLEJS1') throw new Error('bad weights magic')
  const view = new DataView(buffer)
  const headerLength = view.getUint32(9, true)
  const start = 13
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(start, start + headerLength)))
  const dataStart = start + headerLength
  const product = shape => shape.reduce((a, b) => a * b, 1)
  const weights = {}
  for (const entry of header.tensors) {
    const shape = entry.shape
    if (entry.encoding !== 'w4') {
      const raw = new Float32Array(buffer.slice(dataStart + entry.offset, dataStart + entry.offset + entry.nbytes))
      weights[entry.name] = np.array(raw, { dtype: np.float32 }).reshape(shape)
      continue
    }
    const packed = bytes.subarray(dataStart + entry.offset, dataStart + entry.offset + entry.packed_bytes)
    const scales = new Float32Array(bytes.slice(dataStart + entry.scales_offset, dataStart + entry.scales_offset + entry.scales_nbytes).buffer)
    const secondLast = entry.name.includes('/kernel') || entry.name.includes('/mhc_phi_')
    const quantDim = secondLast ? shape.at(-2) : shape.at(-1)
    const columns = secondLast ? shape.at(-1) : 1
    const rows = product(shape) / (quantDim * columns)
    const out = new Float32Array(product(shape))
    let scaleIndex = 0
    for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
      const base = row * quantDim * columns + col
      for (let offset = 0; offset < quantDim; offset += entry.group_size) {
        const scale = scales[scaleIndex++]
        for (let i = offset; i < Math.min(quantDim, offset + entry.group_size); i++) {
          const index = base + i * columns
          out[index] = (((packed[index >> 1] >> ((index & 1) * 4)) & 15) - 8) * scale
        }
      }
    }
    weights[entry.name] = np.array(out, { dtype: np.float32 }).reshape(shape)
  }
  return { header, weights }
}

function parseTokens (value) { return (Array.isArray(value) ? value : JSON.parse(value)).map(x => Number(typeof x === 'object' ? x.token_id : x)) }
function tokenText (id, metadata) { return metadata[id]?.token_text ?? `<${id}>` }
function greedy (values, vocab) {
  const start = values.length - vocab
  let best = start
  for (let i = start + 1; i < values.length; i++) if (values[i] > values[best]) best = i
  return best - start
}
function decode (tokens, metadata) { return tokens.map(id => tokenText(id, metadata)).join('') }
async function load (url) { const response = await fetch(url, { cache: 'force-cache' }); if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`); return response.arrayBuffer() }

$('run').onclick = async () => {
  $('run').disabled = true; $('progress').value = 0; $('tokens').textContent = ''; $('output').textContent = ''
  try {
    status('下载 W4 权重和 prompt…')
    const [binary, prefillValue] = await Promise.all([load('./w4-packed.bin'), fetch('./prefill.json', { cache: 'no-store' }).then(r => r.json())])
    const { header, weights } = parseWeights(binary)
    const prompt = parseTokens(prefillValue)
    const metadata = header.token_metadata || []
    const cfg = normalizeConfig(header.config ?? header)
    const maxNew = Math.max(1, Math.min(1024, Number($('maxTokens').value) || 64))
    const devices = await init()
    const backend = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : devices[0]
    if (!backend) throw new Error('jax-js 没有可用 backend')
    defaultDevice(backend)
    const cache = createKVCache(cfg.max_seq_len)
    const tokens = [...prompt]
    const eos = new Set([5, cfg.eos_token_id, header.eos_token_id].filter(Number.isInteger))
    let generated = 0
    for (let step = 0; step < maxNew; step++) {
      const input = np.array(Int32Array.from(tokens), { dtype: np.int32 }).reshape([1, tokens.length])
      status(`${backend} · ${cache.position ? 'decode' : 'prefill'} · ${tokens.length} tokens · KV position ${cache.position}`)
      $('tokens').textContent = decode(tokens, metadata)
      $('progress').value = step / maxNew
      await new Promise(requestAnimationFrame)
      const logits = forwardWithKVCache(input, cfg, weights, cache)
      const next = greedy(logits.dataSync(), cfg.vocab_size)
      if (eos.has(next)) { status(`遇到 EOS ${next}，轨迹结束；生成 ${tokens.length - prompt.length} tokens。`); break }
      tokens.push(next)
      generated++
      $('output').textContent = decode(tokens.slice(prompt.length), metadata)
    }
    $('progress').value = 1
    $('tokens').textContent = decode(tokens, metadata)
    if (!$('output').textContent) $('output').textContent = decode(tokens.slice(prompt.length), metadata)
    if (generated === maxNew) status(`完成：${generated} 个新 token，KV cache position=${cache.position}。`)
  } catch (error) { console.error(error); status(`错误：${error?.stack || error}`) } finally { $('run').disabled = false }
}
