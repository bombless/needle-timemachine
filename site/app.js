import { forward, normalizeConfig, init, defaultDevice, np } from './forward.js'

const $ = id => document.getElementById(id)
const status = msg => { $('status').textContent = msg }

function parseWeights(buffer) {
  const bytes = new Uint8Array(buffer)
  const magic = new TextDecoder().decode(bytes.subarray(0, 9))
  if (magic !== 'NEEDLEJS1') throw new Error(`bad weights magic: ${magic}`)
  const dv = new DataView(buffer)
  const headerLen = dv.getUint32(9, true)
  const headerStart = 13
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(headerStart, headerStart + headerLen)))
  const dataStart = headerStart + headerLen
  const weights = {}

  const product = shape => shape.reduce((a, b) => a * b, 1)
  for (const e of header.tensors) {
    if (e.encoding !== 'w4') {
      const raw = new Float32Array(buffer.slice(dataStart + e.offset, dataStart + e.offset + e.nbytes))
      weights[e.name] = np.array(raw, { dtype: np.float32 }).reshape(e.shape)
      continue
    }
    const packed = bytes.subarray(dataStart + e.offset, dataStart + e.offset + e.packed_bytes)
    const scalesBytes = bytes.subarray(dataStart + e.scales_offset, dataStart + e.scales_offset + e.scales_nbytes)
    const scales = new Float32Array(scalesBytes.slice().buffer)
    const shape = e.shape
    const reduceSecondLast = e.name.includes('/kernel') || e.name.includes('/mhc_phi_')
    const quantDim = reduceSecondLast ? shape.at(-2) : shape.at(-1)
    const columns = reduceSecondLast ? shape.at(-1) : 1
    const rows = product(shape) / (quantDim * columns)
    const out = new Float32Array(product(shape))
    let si = 0
    for (let row = 0; row < rows; ++row) for (let col = 0; col < columns; ++col) {
      const base = row * quantDim * columns + col
      for (let start = 0; start < quantDim; start += e.group_size) {
        const scale = scales[si++]
        const end = Math.min(quantDim, start + e.group_size)
        for (let i = start; i < end; ++i) {
          const index = base + i * columns
          const q = ((packed[index >> 1] >> ((index & 1) * 4)) & 15) - 8
          out[index] = q * scale
        }
      }
    }
    weights[e.name] = np.array(out, { dtype: np.float32 }).reshape(shape)
  }
  return { header, weights }
}

function parsePrefill(value) {
  const parsed = typeof value === 'string' ? JSON.parse(value) : value
  return parsed.map(x => Number(typeof x === 'object' ? x.token_id : x))
}

function metadataMap(header, prefill) {
  if (Array.isArray(header.token_metadata)) return header.token_metadata
  return prefill
}

function tokenText(id, metadata) {
  const m = metadata[id]
  return m?.token_text ?? `<${id}>`
}

function greedyToken(logits, vocabSize) {
  const start = logits.length - vocabSize
  let best = start
  for (let i = start + 1; i < logits.length; ++i) if (logits[i] > logits[best]) best = i
  return best - start
}

function decode(tokens, metadata) {
  return tokens.map(id => tokenText(id, metadata)).join('')
}

async function loadBinary(url) {
  const r = await fetch(url, { cache: 'force-cache' })
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`)
  return r.arrayBuffer()
}

$('run').addEventListener('click', async () => {
  $('run').disabled = true
  $('progress').value = 0
  $('output').textContent = ''
  $('tokens').textContent = ''
  try {
    status('下载预设 prompt 和 W4 权重…')
    const [weightsBuffer, prefillJson] = await Promise.all([
      loadBinary('./w4-packed.bin'),
      fetch('./prefill.json', { cache: 'no-store' }).then(r => r.json())
    ])
    const { header, weights } = parseWeights(weightsBuffer)
    const prefill = parsePrefill(prefillJson)
    const metadata = metadataMap(header, prefill)
    const cfg = normalizeConfig(header.config ?? header)
    const maxNew = Math.max(1, Math.min(1024, Number($('maxTokens').value) || 128))

    status('初始化 jax-js（优先 WebGPU，否则 Wasm）…')
    const devices = await init()
    const backend = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : devices[0]
    if (!backend) throw new Error('jax-js 没有可用 backend')
    defaultDevice(backend)
    status(`backend=${backend}，prefill ${prefill.length} tokens；开始 forward…`)

    const tokens = [...prefill]
    const eos = new Set([5, cfg.eos_token_id, header.eos_token_id].filter(Number.isInteger))
    let lastText = ''

    for (let step = 0; step <= maxNew; ++step) {
      $('progress').value = step / maxNew
      $('tokens').textContent = tokens.map(id => tokenText(id, metadata)).join('')
      status(`${backend}: forward ${tokens.length} tokens（decode step ${step}/${maxNew}）`)
      await new Promise(requestAnimationFrame)

      const input = np.array(Int32Array.from(tokens), { dtype: np.int32 }).reshape([1, tokens.length])
      const logits = forward(input, cfg, weights)
      const data = logits.dataSync()
      const next = greedyToken(data, cfg.vocab_size)
      if (eos.has(next)) {
        status(`遇到 EOS token ${next}，推理结束。共生成 ${tokens.length - prefill.length} tokens。`)
        break
      }
      tokens.push(next)
      const text = decode(tokens, metadata)
      if (text !== lastText) {
        $('output').textContent = text
        lastText = text
      }
      if (step === maxNew) status(`达到 max new tokens=${maxNew}，停止。`)
    }
    $('progress').value = 1
    $('tokens').textContent = tokens.map(id => tokenText(id, metadata)).join('')
    $('output').textContent = decode(tokens, metadata)
  } catch (err) {
    console.error(err)
    status(`错误：${err?.stack || err}`)
  } finally {
    $('run').disabled = false
  }
})
