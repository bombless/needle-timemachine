import { numpy as np } from 'https://esm.sh/@jax-js/jax@0.1.23'
let trace = { events: [] },
  pos = 0,
  timer = null,
  jaxReady = true
let weights = null
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
  try {
    const payload = await (await fetch('/weights.json')).json()
    if (payload.format !== 'needle-timemachine.weights/v1') throw new Error('不支持的权重格式')
    weights = Object.fromEntries(payload.tensors.map(t => [t.name, decode(t)]))
    const logits = fullForward(trace.prompt_tokens.map(x => Number(x.token_id)), payload.config)
    const fp = await fingerprint(logits)
    const reference = trace.events.find(x => x.op === 'probability_output')?.metadata?.logits_fingerprint
    if (!reference) throw new Error('trace 中没有 Python logits 指纹')
    const ok = closeEnough(fp.sum, reference.sum, Math.sqrt(Math.abs(fp.sumSquares))) &&
      closeEnough(fp.sumSquares, reference.sum_squares, Math.abs(reference.sum_squares))
    $('fullStatus').className = 'status ' + (ok ? 'ok' : 'bad')
    $('fullStatus').innerHTML = ok
      ? `<b>✓ 完整前向校验通过</b>：${payload.tensors.length} 个权重张量，logits 指纹与 Python 一致。`
      : `<b>✗ 完整前向校验失败</b>：logits 指纹不一致（sum=${fp.sum}, sumSquares=${fp.sumSquares}）。`
  } catch (err) {
    $('fullStatus').className = 'status bad'
    $('fullStatus').textContent = '完整前向校验异常：' + err.message + '\n' + (err.stack || '')
  } finally {
    button.disabled = false
    button.textContent = '载入权重并验算'
  }
}
function w (name) {
  const value = weights[name]
  if (!value) throw new Error('缺少权重: ' + name)
  return value
}
function layerScale (name, l, d) {
  return w(name).slice([l, 0], [1, d])
}
function transposeLast (x) { return np.transpose(x, [...Array(x.shape.length - 2).keys(), x.shape.length - 1, x.shape.length - 2]) }
function dense (x, kernel) { return np.matmul(x, kernel) }
function rms (x) {
  const xf = x.astype(np.float32)
  return xf.mul(np.sqrt(np.add(np.mean(xf.mul(xf), -1, true), 1e-6)).reciprocal())
}
function zcrms (x, scale) {
  return rms(x).mul(scale.add(1))
}
function sigmoid (x) { return np.reciprocal(np.add(1, expf(x.neg()))) }
function silu (x) { return x.mul(sigmoid(x)) }
function expf (x) {
  // jax-js keeps integer inputs to unary ops unless an array conversion fixes the dtype.
  return np.exp(np.array(x, { dtype: 'float32' }))
}
function rope (x, cos, sin) {
  const d = x.shape[x.shape.length - 1], half = d / 2
  const a = x.slice([0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], half])
  const b = x.slice([0, 0, 0, half], [x.shape[0], x.shape[1], x.shape[2], half])
  return np.concatenate([a.mul(cos).sub(b.mul(sin)), a.mul(sin).add(b.mul(cos))], -1)
}
function fullForward (tokenIds, cfg) {
  const ids = np.array([tokenIds], { dtype: np.int32 })
  const B = 1, T = tokenIds.length, D = Number(cfg.d_model), L = Number(cfg.num_layers)
  const H = Number(cfg.num_heads), KVH = Number(cfg.num_kv_heads), hd = Number(cfg.attn_dim || D) / H
  const lanes = Number(cfg.mhc_lanes), nC = lanes * D, attnDim = H * hd
  let x = np.take(w('embedding.embedding'), ids.ref, 0).mul(Math.sqrt(D))
  x = np.broadcastTo(x.reshape([B, T, 1, D]), [B, T, lanes, D])
  const pos = np.arange(T).reshape([T, 1])
  const inv = expf(np.arange(0, hd, 2).astype(np.float32).mul(-Math.log(Number(cfg.rope_theta || 10000)) / hd))
  const angles = pos.mul(inv), cos = np.cos(angles.ref), sin = np.sin(angles)
  const emb = []
  const sites = (cfg.engram_layers || [2, 15]).map(Number)
  const orders = (cfg.engram_orders || [2, 3]).map(Number)
  const configuredHeads = Number(cfg.engram_heads)
  const heads = Number.isFinite(configuredHeads) && configuredHeads > 0
    ? configuredHeads
    : Math.max(1, Math.floor(D / (orders.length * 128)))
  const sub = Math.floor(D / (orders.length * heads)), slots = Number(cfg.engram_slots || 8192)
  let engram = null
  if (sites.length) {
    const all = []
    for (let s = 0; s < sites.length; s++) {
      const table = w(`engrams.${s}.embedding`), fetched = []
      for (let oi = 0; oi < orders.length; oi++) for (let h = 0; h < heads; h++) {
        let hash = np.full([B, T], (0x9e3779b9 * (oi * heads + h + 1)) >>> 0, { dtype: np.uint32 })
        for (let j = 0; j < orders[oi]; j++) hash = hash.add(ids.ref).mul(0x01000193)
        hash = np.bitwiseXor(hash, np.rightShift(hash.ref, np.array(15, { dtype: 'int32' })))
        const ix = hash.mod(slots).astype(np.int32)
        const valid = np.greaterEqual(np.arange(T), orders[oi] - 1).reshape([1, T, 1])
        fetched.push(np.take(table.slice([oi * heads + h, 0, 0], [1, slots, sub]).reshape([slots, sub]), ix, 0).mul(valid))
      }
      const e = np.concatenate(fetched, -1)
      const key = dense(e, transposeLast(w(`engrams.${s}.key_proj`)))
      let value = dense(e, transposeLast(w(`engrams.${s}.value_proj`)))
      const taps = w(`engrams.${s}.taps`)
      let convolved = np.zerosLike(value)
      for (let j = 0; j < 4; j++) {
        const delay = j * Math.max(...orders)
        const shifted = delay ? np.pad(value, [[0, 0], [delay, 0], [0, 0]]).slice([0, 0, 0], [B, T, D]) : value
        const tapValid = np.greaterEqual(np.arange(T), delay).reshape([1, T, 1])
        convolved = convolved.add(shifted.mul(taps.slice([j, 0], [1, D])).mul(tapValid))
      }
      all.push([key, convolved])
    }
    engram = all
  }
  for (let l = 0; l < L; l++) {
    emb.push(x)
    const prev = x, flat = prev.reshape([B, T, lanes * D]), nx = rms(flat)
    const pre = sigmoid(dense(nx, w('stack.mhc_phi_pre').slice([l, 0, 0], [1, nC, lanes]).reshape([nC, lanes])).mul(w('stack.mhc_a_pre').slice([l])).add(w('stack.mhc_b_pre').slice([l, 0], [1, lanes])).add(np.array(8 * (l % lanes) - 4)))
    let u = np.sum(pre.reshape([B, T, lanes, 1]).mul(prev), 2)
    if (engram && sites.includes(l)) {
      const s = sites.indexOf(l), ek = engram[s][0], ev = engram[s][1]
      const alpha = sigmoid(np.sum(rms(u).reshape([B, T, 1, D]).mul(rms(ek)), -1).div(Math.sqrt(D)))
      u = u.add(np.sum(alpha.reshape([B, T, 1, 1]).mul(ev), 0))
    }
    const z = zcrms(u, layerScale('stack.layers.block.ZCRMSNorm_0.scale', l, D).reshape([1, 1, 1, D]))
    const layerDense = (name, input, output) =>
      transposeLast(w(name).slice([l, 0, 0], [1, input, output]).reshape([input, output]))
    let q = dense(z, layerDense('stack.layers.block.self_attn.q_proj.kernel', D, attnDim))
    let k = dense(z, layerDense('stack.layers.block.self_attn.k_proj.kernel', D, KVH * hd))
    const v = dense(z, layerDense('stack.layers.block.self_attn.v_proj.kernel', D, KVH * hd))
    q = zcrms(q.reshape([B, T, H, hd]), w('stack.layers.block.self_attn.q_norm.scale').slice([l, 0], [1, hd]))
    k = zcrms(k.reshape([B, T, KVH, hd]), w('stack.layers.block.self_attn.k_norm.scale').slice([l, 0], [1, hd]))
    const qq = rope(q.transpose([0, 2, 1, 3]), cos, sin)
    const kk = rope(k.transpose([0, 2, 1, 3]), cos, sin)
    let vv = v.reshape([B, T, KVH, hd]).transpose([0, 2, 1, 3])
    let kk2 = kk, qq2 = qq
    if (H !== KVH) { kk2 = np.repeat(kk, H / KVH, 1); vv = np.repeat(vv, H / KVH, 1) }
    let score = np.matmul(qq2, kk2.transpose([0, 1, 3, 2])).div(Math.sqrt(hd))
    const causal = np.tril(np.ones([T, T], { dtype: np.bool }))
    score = np.where(causal.reshape([1, 1, T, T]), score, -1e9)
    const att = expf(score.sub(np.max(score, -1, true)))
    const out = np.sum(att.mul(vv), -1, true).mul(0).add(np.matmul(att, vv)).transpose([0, 2, 1, 3]).reshape([B, T, attnDim])
    const gate = sigmoid(dense(z, layerDense('stack.layers.block.self_attn.gate_proj.kernel', D, attnDim)))
    const projected = dense(out.mul(gate), layerDense('stack.layers.block.self_attn.out_proj.kernel', attnDim, D))
    const attnNorm = zcrms(projected, layerScale('stack.layers.block.post_attn_norm.scale', l, D).reshape([1, 1, D]))
    const blockOut = u.add(attnNorm.mul(sigmoid(w('stack.layers.block.attn_gate').slice([l]))))
    const hz = zcrms(blockOut, layerScale('stack.layers.block.pre_hada_norm.scale', l, D).reshape([1, 1, D]))
    const n = 1 << Math.ceil(Math.log2(D)), padded = D === n ? hz : np.pad(hz, [[0, 0], [0, 0], [0, n - D]])
    const Hm = hadamard(n)
    const h = dense(padded.mul(w(`stack.layers.block.hadamard_mlp.d1`).slice([l, 0], [1, n])), Hm)
    const back = dense(silu(h.mul(w(`stack.layers.block.hadamard_mlp.d2`).slice([l, 0], [1, n]))), Hm).mul(w(`stack.layers.block.hadamard_mlp.d3`).slice([l, 0], [1, n])).slice([0, 0, 0], [B, T, D])
    const y = blockOut.add(back).sub(u)
    const postN = sigmoid(dense(nx, w('stack.mhc_phi_post').slice([l, 0, 0], [1, nC, lanes]).reshape([nC, lanes])).mul(w('stack.mhc_a_post').slice([l])).add(w('stack.mhc_b_post').slice([l, 0], [1, lanes])).add(-4 * (1 - (l % lanes)))).mul(2)
    const res = dense(nx, w('stack.mhc_phi_res').slice([l, 0, 0], [1, nC, lanes * lanes]).reshape([nC, lanes * lanes])).mul(w('stack.mhc_a_res').slice([l])).reshape([B, T, lanes, lanes]).add(w('stack.mhc_b_res').slice([l]).reshape([1, 1, lanes, lanes]))
    let norm = res
    for (let i = 0; i < 20; i++) {
      norm = norm.sub(np.log(np.sum(expf(norm), -1, true)))
      norm = norm.sub(np.log(np.sum(expf(norm), -2, true)))
    }
    norm = expf(norm)
    x = np.add(np.sum(norm.reshape([B, T, lanes, lanes, 1]).mul(prev.reshape([B, T, 1, lanes, D])), 3), postN.reshape([B, T, lanes, 1]).mul(y.reshape([B, T, 1, D]))).astype(np.float32)
  }
  const out = zcrms(np.mean(x, 2), w('stack.final_norm.scale'))
  return dense(out, w('embedding.embedding').transpose([1, 0]))
}
function hadamard (n) {
  let h = np.array([[1]], { dtype: np.float32 })
  while (h.shape[0] < n) h = np.concatenate([np.concatenate([h, h], 1), np.concatenate([h, h.mul(-1)], 1)], 0)
  return h.div(Math.sqrt(n))
}
$('verifyFull').onclick = loadWeights
load()
