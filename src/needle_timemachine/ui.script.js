import {
  init,
  defaultDevice,
  np,
  forward,
  normalizeConfig,
  weightsFromPayload
} from './forward.js'
let trace = { events: [] },
  pos = 0,
  timer = null
const $ = id => document.getElementById(id)
const esc = v =>
  String(v ?? '').replace(
    /[&<>"']/g,
    c =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[
        c
      ])
  )
async function load () {
  trace = await (await fetch('/trace.json')).json()
  $('summary').textContent = `${trace.events.length} events · ${
    trace.checkpoint || 'unknown checkpoint'
  }`
  $('slider').max = Math.max(0, trace.events.length - 1)
  drawDots()
  renderPrompt()
  render(0)
}
function drawDots () {
  const t = $('track')
  t.innerHTML = ''
  trace.events.forEach((e, i) => {
    const d = document.createElement('button')
    d.className = 'dot' + (e.op === 'layer_output' ? ' layer' : '')
    d.title = `${e.step}: ${e.op}`
    d.style.left =
      (trace.events.length < 2 ? 50 : (i * 100) / (trace.events.length - 1)) +
      '%'
    d.onclick = () => render(i)
    t.appendChild(d)
  })
}
function decode (p) {
  if (!p || p.encoding !== 'base64-f32-le')
    throw new Error('unsupported tensor payload')
  const raw = Uint8Array.from(atob(p.data), c => c.charCodeAt(0))
  return np
    .array(new Float32Array(raw.buffer), { dtype: np.float32 })
    .reshape(p.shape)
}
async function fingerprint (a) {
  const sum = await a.ref.sum().jsAsync()
  const sq = await a.mul(a.ref).sum().jsAsync()
  return { sum: Number(sum), sumSquares: Number(sq) }
}
function closeEnough (a, b, scale = 1) {
  const tol = 1e-4 * Math.max(1, Math.abs(b), scale)
  return Math.abs(a - b) <= tol
}
async function verifyLayer () {
  const e = trace.events[pos]
  if (e.op !== 'layer_output' || !e.values?.input || !e.values?.output) {
    $('verifyStatus').className = 'status warn'
    $('verifyStatus').textContent =
      '该事件没有可重放的 layer input/output payload。'
    return
  }
  const btn = $('verify')
  btn.disabled = true
  btn.textContent = '验算中…'
  $('verifyStatus').className = 'status'
  try {
    const input = decode(e.values.input),
      output = decode(e.values.output)
    const fi = await fingerprint(input),
      fo = await fingerprint(output)
    const ri = e.values.input,
      ro = e.values.output
    const inputOk =
      closeEnough(fi.sum, ri.sum, Math.sqrt(Math.abs(fi.sumSquares))) &&
      closeEnough(fi.sumSquares, ri.sum_squares, Math.abs(ri.sum_squares))
    const outputOk =
      closeEnough(fo.sum, ro.sum, Math.sqrt(Math.abs(fo.sumSquares))) &&
      closeEnough(fo.sumSquares, ro.sum_squares, Math.abs(ro.sum_squares))
    let continuity = true,
      continuityText = ''
    const prev = trace.events.find(
      x => x.op === 'layer_output' && Number(x.layer) === Number(e.layer) - 1
    )
    if (prev?.values?.output) {
      const a = decode(prev.values.output),
        b = decode(e.values.input)
      const delta = await np.abs(a.sub(b)).max().jsAsync()
      continuity = Number(delta) <= 1e-5
      continuityText = `；与上一层输出最大差 ${Number(delta).toExponential(3)}`
    } else if (Number(e.layer) === 0) {
      const emb = trace.events.find(x => x.op === 'embedding_output')
      if (emb?.values?.output) {
        const a = decode(emb.values.output),
          b = decode(e.values.input)
        const delta = await np.abs(a.sub(b)).max().jsAsync()
        continuity = Number(delta) <= 1e-5
        continuityText = `；与 embedding 输出最大差 ${Number(
          delta
        ).toExponential(3)}`
      }
    }
    const ok = inputOk && outputOk && continuity
    $('verifyStatus').className = 'status ' + (ok ? 'ok' : 'bad')
    $('verifyStatus').innerHTML = ok
      ? `<b>✓ JAX.js 验算通过</b>：输入/输出指纹与 Python trace 一致${continuityText}。`
      : `<b>✗ 验算失败</b>：${!inputOk ? 'input fingerprint 不一致；' : ''}${
          !outputOk ? 'output fingerprint 不一致；' : ''
        }${!continuity ? '层间输入/输出不连续。' : ''}`
  } catch (err) {
    $('verifyStatus').className = 'status bad'
    $('verifyStatus').textContent = 'JAX.js 验算异常：' + err.message
  } finally {
    btn.disabled = false
    btn.textContent = 'JAX.js 验算这一层'
  }
}
function render (i) {
  if (!trace.events.length) return
  pos = Math.max(0, Math.min(i, trace.events.length - 1))
  const e = trace.events[pos]
  $('slider').value = pos
  $('title').textContent = `Step ${e.step} · ${e.op}`
  $('details').innerHTML = [
    ['layer', e.layer ?? '—'],
    ['name', e.name ?? '—'],
    ['phase', e.phase],
    ['snapshot', e.snapshot_id ?? '—']
  ]
    .map(([k, v]) => `<div class="muted">${k}</div><div>${esc(v)}</div>`)
    .join('')
  $('tensors').textContent = JSON.stringify(e.tensors || {}, null, 2)
  $('metadata').textContent = JSON.stringify(e.metadata || {}, null, 2)
  const isLayer = e.op === 'layer_output'
  $('verify').disabled = !isLayer
  $('layerTitle').textContent = isLayer
    ? `Layer ${e.layer} verification`
    : 'Layer verification'
  if (isLayer)
    $('verifyStatus').textContent =
      '点击按钮，用 jax-js 在浏览器重放本层记录的 input/output。'
  document
    .querySelectorAll('.dot')
    .forEach((d, j) => d.classList.toggle('current', j === pos))
  renderProbabilities(e)
}
function renderPrompt () {
  const xs = trace.prompt_tokens || []
  $('promptTokens').innerHTML = xs.length
    ? '<table class="token-table"><tr><th>#</th><th>ID</th><th>文本</th></tr>' +
      xs
        .map(
          (x, i) =>
            `<tr><td>${i + 1}</td><td>${esc(x.token_id)}</td><td class="mono">${
              esc(x.token_text) || '∅'
            }</td></tr>`
        )
        .join('') +
      '</table>'
    : '—'
}
function renderProbabilities (e) {
  const xs = e?.metadata?.top_k
  if (!Array.isArray(xs) || !xs.length) {
    $('probabilities').textContent = '—'
    return
  }
  $('probabilities').innerHTML =
    '<table class="probability-table"><tr><th>Rank</th><th>ID</th><th>文本</th><th>概率</th><th>分布</th></tr>' +
    xs
      .map((x, i) => {
        const p = Number(x.probability) || 0
        return `<tr><td>${i + 1}</td><td>${esc(x.token_id)}</td><td>${
          esc(x.token_text) || '∅'
        }</td><td>${(p * 100).toFixed(
          4
        )}%</td><td><div class="probbar"><div class="fill" style="width:${Math.min(
          100,
          p * 100
        )}%"></div></div></td></tr>`
      })
      .join('') +
    '</table>'
}
function renderJaxLogits (logits, cfg) {
  const data = logits.dataSync()
  const start = (data.length - Number(cfg.vocab_size))
  const final = Array.from(data.slice(start, start + Number(cfg.vocab_size)))
  const metadata = new Map()
  for (const e of trace.events) {
    for (const x of e?.metadata?.top_k || []) {
      if (x && x.token_id != null && !metadata.has(Number(x.token_id))) {
        metadata.set(Number(x.token_id), x.token_text || '')
      }
    }
  }
  const top = final
    .map((logit, tokenId) => ({
      tokenId,
      tokenText: metadata.get(tokenId) || '',
      logit: Number(logit)
    }))
    .sort((a, b) => b.logit - a.logit)
    .slice(0, 5)
  $('jaxLogits').innerHTML =
    '<table class="probability-table"><tr><th>Rank</th><th>Token ID</th><th>Token</th><th>JAX.js logit</th></tr>' +
    top
      .map(
        (x, i) =>
          `<tr><td>${i + 1}</td><td>${esc(x.tokenId)}</td><td class="mono">${
            esc(x.tokenText) || '∅'
          }</td><td class="mono">${x.logit.toPrecision(10)}</td></tr>`
      )
      .join('') +
    '</table>'
}
function step (n) {
  render(pos + n)
}
function stop () {
  clearTimeout(timer)
  timer = null
  $('play').textContent = '▶ Play'
}
function play () {
  if (timer) return
  $('play').textContent = '⏸ Pause'
  const tick = () => {
    if (pos >= trace.events.length - 1) {
      stop()
      return
    }
    step(1)
    timer = setTimeout(tick, 500 / +$('speed').value)
  }
  tick()
}
$('first').onclick = () => render(0)
$('prev').onclick = () => step(-1)
$('next').onclick = () => step(1)
$('last').onclick = () => render(trace.events.length - 1)
$('slider').oninput = e => render(+e.target.value)
$('verify').onclick = verifyLayer
$('speed').oninput = e => {
  $('speedText').textContent = e.target.value + '×'
  if (timer) {
    stop()
    play()
  }
}
$('play').onclick = () => (timer ? stop() : play())

async function loadWeights () {
  const button = $('verifyFull')
  button.disabled = true
  button.textContent = '载入中…'
  $('fullStatus').className = 'status'
  $('jaxLogits').textContent = '计算中…'
  try {
    const response = await fetch('/weights.json')
    if (!response.ok) throw new Error(`权重请求失败 (${response.status})`)
    const payload = await response.json()
    const backend = (await init('wasm')).includes('wasm') ? 'wasm' : 'cpu'
    defaultDevice(backend)
    const weights = weightsFromPayload(payload)
    const tokenIds = (trace.prompt_tokens || []).map(x => Number(x.token_id))
    if (!tokenIds.length) throw new Error('trace 中没有 prompt tokens')
    const cfg = normalizeConfig(payload.config)
    const logits = forward(
      np.array(Int32Array.from(tokenIds), { dtype: np.int32 }).reshape([1, tokenIds.length]),
      cfg,
      weights
    )
    const fp = await fingerprint(logits)
    const reference = trace.events.find(x => x.op === 'probability_output')?.metadata?.logits_fingerprint
    if (!reference) throw new Error('trace 中没有 Python logits 指纹')
    const ok = closeEnough(fp.sum, reference.sum, Math.sqrt(Math.abs(fp.sumSquares))) &&
      closeEnough(fp.sumSquares, reference.sum_squares, Math.abs(reference.sum_squares))
    renderJaxLogits(logits, cfg)
    $('fullStatus').className = 'status ' + (ok ? 'ok' : 'bad')
    $('fullStatus').innerHTML = ok
      ? `<b>✓ 完整前向校验通过</b>：浏览器执行 forward.js（${backend}），${payload.tensors.length} 个权重张量的 logits 指纹与 Python 一致。`
      : `<b>✗ 完整前向校验失败</b>：浏览器执行 forward.js（${backend}），logits 指纹不一致（sum=${fp.sum}, sumSquares=${fp.sumSquares}）。JAX.js 的最终 token logits 仍已列在下方。`
  } catch (err) {
    $('fullStatus').className = 'status bad'
    $('fullStatus').textContent = '完整前向校验异常：' + err.message + '\n' + (err.stack || '')
    $('jaxLogits').textContent = '无法生成 JAX.js logits：' + err.message
  } finally {
    button.disabled = false
    button.textContent = '载入权重并运行 forward.js'
  }
}
$('verifyFull').onclick = loadWeights
load()
