import { init, defaultDevice, numpy as np } from '@jax-js/jax'

const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node)
const fs = isNode ? (await import('node:fs')).default : null

const MAGIC = 'NEEDLEJS1'
const HEADER_BYTES = 4
const CACT_TAG = 0x05e12a83
const CACT_HDR_BYTES = 120
const CACT_REC_BYTES = 44
const D = np.float32

