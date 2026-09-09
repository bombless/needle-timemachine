import { forward, normalizeConfig, init, defaultDevice, np } from '../forward.js'

const $ = id => document.getElementById(id)
const status = msg => { $('status').textContent = msg }

function parseWeights(buffer) {
  const bytes = new Uint8Array(buffer)
  const magic = new TextDecoder().decode(bytes.subarray(0, 9))
  if (magic !== 'NEEDLEJS1') throw new Error(`bad weights magic: ${magic}`)
