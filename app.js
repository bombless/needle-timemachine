var __defProp = Object.defineProperty;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __esm = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};
var __export = (target, all2) => {
  for (var name in all2)
    __defProp(target, name, { get: all2[name], enumerable: true });
};

// node_modules/@jax-js/jax/dist/webgpu-Dluq5ZOf.js
var webgpu_Dluq5ZOf_exports = {};
__export(webgpu_Dluq5ZOf_exports, {
  WebGPUBackend: () => WebGPUBackend
});
function dtypeToWgsl(dtype, storage = false) {
  switch (dtype) {
    case "bool":
      return storage ? "i32" : "bool";
    case "int32":
      return "i32";
    case "uint32":
      return "u32";
    case "float32":
      return "f32";
    case "float16":
      return "f16";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}
function maxValueWgsl(dtype) {
  switch (dtype) {
    case "bool":
      return "1";
    case "int32":
      return "2147483647";
    case "uint32":
      return "4294967295u";
    case "float32":
      return "inf()";
    case "float16":
      return "f16(inf())";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}
function constToWgsl(dtype, value) {
  if (dtype === "bool") return value ? "true" : "false";
  if (dtype === "int32") return value.toString();
  if (dtype === "uint32") return value.toString() + "u";
  if (dtype === "float32") {
    if (Number.isNaN(value)) return "nan()";
    if (!Number.isFinite(value)) return value > 0 ? "inf()" : "-inf()";
    return "f32(" + value.toString() + ")";
  }
  if (dtype === "float16") {
    if (Number.isNaN(value)) return "f16(nan())";
    if (!Number.isFinite(value)) return value > 0 ? "f16(inf())" : "f16(-inf())";
    return "f16(" + value.toString() + ")";
  }
  throw new Error(`Unsupported const dtype: ${dtype}`);
}
function reduceOpWgsl(op, dtype, a, b) {
  if (op === "Add") return `(${a} + ${b})`;
  if (op === "Mul") return `(${a} * ${b})`;
  if (op === "Min") return dtype === "bool" ? `(${a} && ${b})` : `min(${a}, ${b})`;
  if (op === "Max") return dtype === "bool" ? `(${a} || ${b})` : `max(${a}, ${b})`;
  throw new Error(`Unsupported reduction op: ${op}`);
}
function calculateGrid(gridSize) {
  let gridX = gridSize;
  let gridY = 1;
  if (gridSize > 65535) {
    gridX = gridOffsetY;
    gridY = Math.ceil(gridSize / gridOffsetY);
  }
  return [gridX, gridY];
}
function yieldTask() {
  const port = (channel ??= (() => {
    const created = new MessageChannel();
    created.port1.onmessage = () => resolvers.shift()?.();
    created.port1.start();
    return created;
  })()).port2;
  return new Promise((resolve) => {
    resolvers.push(resolve);
    port.postMessage(0);
  });
}
async function mapAsyncRead(device, staging) {
  const mapped = staging.mapAsync(GPUMapMode.READ);
  if (!isFirefox) return mapped;
  let settled = false;
  const tracked = mapped.finally(() => {
    settled = true;
  });
  await yieldTask();
  const start = performance.now();
  let lastNudge = -Infinity;
  while (!settled) {
    const elapsed = performance.now() - start;
    if (elapsed >= DEADLINE_MS) break;
    if (performance.now() - lastNudge >= NUDGE_INTERVAL_MS) {
      lastNudge = performance.now();
      device.queue.submit([device.createCommandEncoder().finish()]);
    }
    if (elapsed < BUSY_MS) await yieldTask();
    else await new Promise((resolve) => setTimeout(resolve, BACKOFF_MS));
  }
  return tracked;
}
function uniformDtype(dtype) {
  if (dtype === "float16") return "float32";
  if (dtype === "bool") return "int32";
  return dtype;
}
function replacementFor({ name, dtype, uniformDtype: uniformDtype2 }) {
  const value = AluExp.variable(uniformDtype2, `uniforms.${name}`);
  if (dtype === "float16") return AluExp.cast("float16", value);
  if (dtype === "bool") return AluExp.cmpne(value, AluExp.i32(0));
  return value;
}
function writeUniform(view, offset, dtype, value) {
  switch (dtype) {
    case "float32":
      view.setFloat32(offset, value, true);
      break;
    case "int32":
      view.setInt32(offset, value, true);
      break;
    case "uint32":
      view.setUint32(offset, value, true);
      break;
    default:
      throw new Error(`Unsupported dtype for constant uniform: ${dtype}`);
  }
}
function liftConstants(exp3) {
  const uniforms = [];
  return [exp3.rewrite((node) => {
    if (node.op !== "Const" || node.arg === 0) return;
    const uniform3 = {
      name: `c${uniforms.length}`,
      dtype: node.dtype,
      uniformDtype: uniformDtype(node.dtype),
      value: node.arg
    };
    uniforms.push(uniform3);
    return replacementFor(uniform3);
  }), uniforms];
}
function uniformsData(uniforms) {
  const data = new Uint8Array(uniforms.length * 4);
  const view = new DataView(data.buffer);
  uniforms.forEach((u, i) => writeUniform(view, i * 4, u.uniformDtype, u.value));
  return data;
}
function nullaryKernelSource(device, kernel) {
  if (kernel.nargs !== 0 || kernel.reduction) return null;
  let exp3 = kernel.exp.substitute({ gidx: AluExp.special("int32", "gidx", kernel.size) }).simplify();
  let uniforms = [];
  [exp3, uniforms] = liftConstants(exp3);
  const wb = new WgslBuilder();
  wb.emitPreamble(device, [exp3]);
  if (uniforms.length > 0) wb.emit("struct Uniforms {", wb.pushIndent, ...uniforms.map((u) => `${u.name}: ${dtypeToWgsl(u.uniformDtype)},`), wb.popIndent, "}\n");
  const resultTy = dtypeToWgsl(kernel.dtype, true);
  wb.emit(`@group(0) @binding(0) var<storage, read_write> result : array<${resultTy}>;`);
  if (uniforms.length > 0) wb.emit(`@group(1) @binding(0) var<uniform> uniforms: Uniforms;`);
  const workgroupSize = findPow2(kernel.size, 256);
  const [gridX, gridY] = calculateGrid(Math.ceil(kernel.size / workgroupSize));
  wb.emit("", `@compute @workgroup_size(${workgroupSize})`, "fn main(@builtin(global_invocation_id) id : vec3<u32>) {", wb.pushIndent);
  if (gridY === 1) wb.emit(`if (id.x >= ${kernel.size}) { return; }`, "let gidx: i32 = i32(id.x);");
  else {
    const sizeX = gridX * workgroupSize;
    wb.emit(`if (${sizeX} * id.y + id.x >= ${kernel.size}) { return; }`, `let gidx: i32 = i32(${sizeX} * id.y + id.x);`);
  }
  const gen = new WgslExpCodegen(wb, []);
  gen.countReferences(exp3);
  let rhs = strip1(gen.run(exp3));
  if (resultTy !== dtypeToWgsl(exp3.dtype)) rhs = `${resultTy}(${rhs})`;
  wb.emit(`result[gidx] = ${rhs};`, wb.popIndent, "}");
  return {
    code: wb.toString(),
    numInputs: 0,
    numOutputs: 1,
    hasUniform: uniforms.length > 0,
    passes: [{
      grid: [gridX, gridY],
      uniform: uniforms.length > 0 ? uniformsData(uniforms) : void 0
    }]
  };
}
function bitonicSortUniform(pass) {
  const ar = /* @__PURE__ */ new Uint32Array(3);
  ar[0] = pass.kind === "sort" ? 0 : 1;
  ar[1] = pass.mergeStep ?? 0;
  ar[2] = pass.mergeStage ?? 0;
  return new Uint8Array(ar.buffer);
}
function bitonicSortShader(device, dtype, n, batches, outputIndices) {
  const ty = dtypeToWgsl(dtype, true);
  const paddedN = 1 << Math.ceil(Math.log2(n || 1));
  const numThreads = Math.ceil(paddedN / 2);
  const workgroupSize = findPow2(numThreads, device.limits.maxComputeWorkgroupSizeX);
  const workgroupsPerBatch = numThreads / workgroupSize;
  const numStages = Math.log2(paddedN);
  const numLocalStages = Math.min(numStages, Math.log2(workgroupSize * 2));
  const needsF16 = dtype === "float16";
  const padValue = isFloatDtype(dtype) ? `${ty}(nan())` : maxValueWgsl(dtype);
  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

struct Uniforms {
  kind: u32, // 0 = sort, 1 = merge
  merge_step: u32, // half_block = 2^step
  merge_stage: u32, // only used for merge
}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;
${outputIndices ? `@group(0) @binding(2) var<storage, read_write> output_idx: array<i32>;` : ""}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

var<workgroup> shared_vals: array<${ty}, ${workgroupSize * 2}>;
${outputIndices ? `var<workgroup> shared_idx: array<i32, ${workgroupSize * 2}>;` : ""}

fn compare(a: ${ty}, b: ${ty}) -> bool {
${isFloatDtype(dtype) ? `
  let min_value = min(a, b);
  return a == min_value && b != min_value;` : "  return a < b;"}
}

fn compare_and_swap(i: u32, j: u32) {
  let val_i = shared_vals[i];
  let val_j = shared_vals[j];
${outputIndices ? `
  if (
    compare(val_j, val_i) ||
    (!compare(val_i, val_j) && shared_idx[j] < shared_idx[i])
  ) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
    let tmp_idx = shared_idx[i];
    shared_idx[i] = shared_idx[j];
    shared_idx[j] = tmp_idx;
  }` : `
  if (compare(val_j, val_i)) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
  }`}
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let blockid = wg_id.x + wg_id.y * ${gridOffsetY}u;
  let batch = blockid / ${workgroupsPerBatch}u;
  let wg_in_batch = blockid % ${workgroupsPerBatch}u;

  let tid = local_id.x;
  let base = batch * ${n}u;

  if (uniforms.kind == 0u || (uniforms.kind == 1u && uniforms.merge_step == ${numLocalStages - 1}u)) {
    let wg_base = wg_in_batch * ${workgroupSize * 2}u;

    // Load data into shared memory (2 elements per thread)
    let idx0 = tid * 2u;
    let idx1 = tid * 2u + 1u;
    // Load from input for initial 'sort' pass, then from output (read-write) for 'merge' passes.
    if (uniforms.kind == 0u) {
      shared_vals[idx0] = select(${padValue}, input[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_vals[idx1] = select(${padValue}, input[base + wg_base + idx1], wg_base + idx1 < ${n}u);
${outputIndices ? `
      shared_idx[idx0] = i32(wg_base + idx0);
      shared_idx[idx1] = i32(wg_base + idx1);` : ""}
    } else {
      shared_vals[idx0] = select(${padValue}, output[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_vals[idx1] = select(${padValue}, output[base + wg_base + idx1], wg_base + idx1 < ${n}u);
${outputIndices ? `
      shared_idx[idx0] = select(${n}, output_idx[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_idx[idx1] = select(${n}, output_idx[base + wg_base + idx1], wg_base + idx1 < ${n}u);` : ""}
    }
    workgroupBarrier();

    let initial_stage = select(0u, ${numLocalStages - 1}u, uniforms.kind != 0u);
    for (var stage = initial_stage; stage < ${numLocalStages}u; stage++) {
      for (var step1 = stage + 1u; step1 > 0u; step1--) {
        let step = step1 - 1u;
        let half_block = 1u << step;
        let is_first_step = uniforms.kind == 0u && step == stage;

        let block_offset = (tid / half_block) * half_block;
        let local_offset = tid % half_block;
        let i = block_offset * 2u + local_offset;
        let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);
        compare_and_swap(i, j);

        workgroupBarrier();
      }
    }

    if (wg_base + idx0 < ${n}u) {
      output[base + wg_base + idx0] = shared_vals[idx0];
      ${outputIndices ? `output_idx[base + wg_base + idx0] = shared_idx[idx0];` : ""}
    }
    if (wg_base + idx1 < ${n}u) {
      output[base + wg_base + idx1] = shared_vals[idx1];
      ${outputIndices ? `output_idx[base + wg_base + idx1] = shared_idx[idx1];` : ""}
    }
  } else {
    // Execute single merge pass for a step >= numLocalStages.
    let half_block = 1u << uniforms.merge_step;  // half_block >= workgroupSize * 2
    let thread_in_batch = wg_in_batch * ${workgroupSize} + tid;
    let is_first_step = uniforms.merge_step == uniforms.merge_stage;

    let block_offset = (thread_in_batch / half_block) * half_block;
    let local_offset = thread_in_batch % half_block;
    let i = block_offset * 2u + local_offset;
    let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);

    // Global version of compare_and_swap()
    if (j < ${n}u) {
      let val_i = output[base + i];
      let val_j = output[base + j];
${outputIndices ? `
      let idx_i = output_idx[base + i];
      let idx_j = output_idx[base + j];
      if (compare(val_j, val_i) || (!compare(val_i, val_j) && idx_j < idx_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;
        output_idx[base + i] = idx_j;
        output_idx[base + j] = idx_i;` : `
      if (compare(val_j, val_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;`}
      }
    }
  }
}
`.trim();
  const grid = calculateGrid(batches * workgroupsPerBatch);
  const passes = [{ kind: "sort" }];
  for (let mergeStage = numLocalStages; mergeStage < numStages; mergeStage++) for (let mergeStep = mergeStage; mergeStep >= numLocalStages - 1; mergeStep--) passes.push({
    kind: "merge",
    mergeStep,
    mergeStage
  });
  return [{
    code,
    numInputs: 1,
    numOutputs: outputIndices ? 2 : 1,
    hasUniform: true,
    passes: passes.map((pass) => ({
      grid,
      uniform: bitonicSortUniform(pass)
    }))
  }];
}
function createSort(device, type) {
  const dtype = type.inputDtypes[0];
  const shape2 = type.inputShapes[0];
  const n = shape2[shape2.length - 1];
  return bitonicSortShader(device, dtype, n, prod(shape2.slice(0, -1)), false);
}
function createArgsort(device, type) {
  const dtype = type.inputDtypes[0];
  const shape2 = type.inputShapes[0];
  const n = shape2[shape2.length - 1];
  return bitonicSortShader(device, dtype, n, prod(shape2.slice(0, -1)), true);
}
function wgslCoordinate(linearIndex, view, axis) {
  if (view.shape[axis] <= 1) return "0u";
  const stride = view.strides[axis];
  return `(${stride === 1 ? linearIndex : `(${linearIndex} / ${stride}u)`} % ${view.shape[axis]}u)`;
}
function wgslBroadcastIndex(linearIndex, updateView, indexView, outDim, indexRank) {
  const firstUpdateDim = outDim + indexRank - indexView.ndim;
  const terms = [];
  for (let i = 0; i < indexView.ndim; i++) {
    const stride = indexView.strides[i];
    if (stride === 0) continue;
    const coord = wgslCoordinate(linearIndex, updateView, firstUpdateDim + i);
    terms.push(stride === 1 ? coord : `(${coord} * ${stride}u)`);
  }
  return terms.length === 0 ? "0u" : terms.join(" + ");
}
function scatterAddWgsl(dtype) {
  switch (dtype) {
    case "float32":
      return {
        statement: "atomic_add_f32(&output[output_index], updates[global]);",
        helpers: `
fn atomic_add_f32(address: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var old_bits = atomicLoad(address);
  loop {
    let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
    let result = atomicCompareExchangeWeak(address, old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}`
      };
    case "float16":
      return {
        statement: `
  atomic_add_f16(
    &output[output_index / 2u],
    output_index % 2u != 0u,
    updates[global],
  );`.trim(),
        helpers: `
fn atomic_add_f16(
  address: ptr<storage, atomic<u32>, read_write>,
  high: bool,
  value: f16,
) {
  var old_bits = atomicLoad(address);
  loop {
    var pair = unpack2x16float(old_bits);
    if (high) {
      pair.y += f32(value);
    } else {
      pair.x += f32(value);
    }
    let new_bits = pack2x16float(pair);
    let result = atomicCompareExchangeWeak(address, old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}`
      };
    case "uint32":
    case "int32":
      return {
        statement: "atomicAdd(&output[output_index], updates[global]);",
        helpers: ""
      };
    case "bool":
      return {
        statement: "atomicOr(&output[output_index], updates[global]);",
        helpers: ""
      };
    default:
      throw new Error(`Unsupported atomic Scatter dtype for WebGPU: ${dtype}`);
  }
}
function createScatter(device, type, { op, shape: outputShape, axis: indexedAxes, outDim, uniqueIndices }) {
  const dtype = type.inputDtypes[0];
  const updateView = View.create(type.inputShapes[0]);
  const indexViews = type.inputShapes.slice(1).map((shape2) => View.create(shape2));
  const indexRank = Math.max(...indexViews.map((view) => view.ndim));
  const updateCount = updateView.size;
  const elementType = dtypeToWgsl(dtype, true);
  const needsF16 = dtype === "float16";
  if (needsF16 && !device.features.has("shader-f16")) throw new Error("WebGPU device does not support shader-f16 feature");
  const freeUpdateDims = [...range(outDim), ...range(outDim + indexRank, updateView.ndim)];
  const indexLoads = indexViews.map((indexView, i) => {
    return `  let scatter_index_${i} = i32(index_${i}[${wgslBroadcastIndex("global", updateView, indexView, outDim, indexRank)}]);`;
  });
  const bounds = indexedAxes.map((outputAxis, i) => `scatter_index_${i} < 0 || scatter_index_${i} >= ${outputShape[outputAxis]}`);
  const outputView = View.create(outputShape);
  let freeDim = 0;
  const outputIndex = range(outputShape.length).map((outputAxis) => {
    const indexInput = indexedAxes.indexOf(outputAxis);
    let coord;
    if (indexInput === -1) coord = wgslCoordinate("global", updateView, freeUpdateDims[freeDim++]);
    else coord = `u32(scatter_index_${indexInput})`;
    return outputView.strides[outputAxis] === 1 ? coord : `(${coord} * ${outputView.strides[outputAxis]}u)`;
  }).join(" + ");
  const atomicAdd = op === "add" && !uniqueIndices;
  const outputStorageType = atomicAdd ? `atomic<${dtype === "int32" || dtype === "bool" ? "i32" : "u32"}>` : elementType;
  const updateSource = atomicAdd ? scatterAddWgsl(dtype) : {
    statement: "output[output_index] = updates[global];",
    helpers: ""
  };
  const maxThreads = device.limits.maxComputeWorkgroupSizeX;
  const workgroupSize = findPow2(Math.min(Math.max(updateCount, 1), maxThreads), maxThreads);
  return [{
    code: `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}
${updateSource.helpers}

@group(0) @binding(0) var<storage, read> updates: array<${elementType}>;
${indexViews.map((_, i) => `@group(0) @binding(${i + 1}) var<storage, read> index_${i}: array<${dtypeToWgsl(type.inputDtypes[i + 1], true)}>;`).join("\n")}
@group(0) @binding(${type.inputShapes.length}) var<storage, read_write> output: array<${outputStorageType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;
  if (global >= ${updateCount}u) {
    return;
  }
${indexLoads.join("\n")}
  if (${bounds.join(" || ")}) {
    return;
  }
  let output_index = ${outputIndex};
  ${updateSource.statement}
}
`.trim(),
    numInputs: type.inputShapes.length,
    numOutputs: 1,
    hasUniform: false,
    clearOutputs: true,
    passes: [{ grid: calculateGrid(Math.ceil(updateCount / workgroupSize)) }]
  }];
}
function createTriangularSolve(device, type, params) {
  const dtype = type.inputDtypes[0];
  const aShape = type.inputShapes[0];
  const bShape = type.inputShapes[1];
  const n = aShape[aShape.length - 1];
  const numRhs = bShape[bShape.length - 2];
  const numMatrices = prod(aShape.slice(0, -2));
  const needsF16 = dtype === "float16";
  const ty = dtypeToWgsl(dtype, true);
  const workgroupSize = findPow2(n, device.limits.maxComputeWorkgroupSizeX);
  return [{
    code: `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> a: array<${ty}>;
@group(0) @binding(1) var<storage, read> b: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> x: array<${ty}>;

// Shared memory for the current pivot value x[j]
var<workgroup> x_j: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_idx = wg_id.x + wg_id.y * ${gridOffsetY}u;
  let mat_idx = wg_idx / ${numRhs}u;
  let rhs_idx = wg_idx % ${numRhs}u;

  if (mat_idx >= ${numMatrices}u) {
    return;
  }

  let a_base = mat_idx * ${n * n}u;
  let bx_base = (mat_idx * ${numRhs}u + rhs_idx) * ${n}u;
  let tid = local_id.x;

  // Step 1: Copy b to x (threads collaborate)
  for (var idx = tid; idx < ${n}u; idx += ${workgroupSize}u) {
    x[bx_base + idx] = b[bx_base + idx];
  }
  storageBarrier();

  // Step 2: Back-substitution from j = n-1 down to 0
  for (var jj = 0u; jj < ${n}u; jj++) {
    let j = ${n - 1}u - jj;

    // Thread 0 computes x[j] = x[j] / a[j,j]
    if (tid == 0u) {
      ${params.unitDiagonal ? `x_j = x[bx_base + j];` : `x_j = x[bx_base + j] / a[a_base + j * ${n}u + j];`}
      x[bx_base + j] = x_j;
    }
    workgroupBarrier();  // Sync shared memory x_j

    // All threads subtract x[j] * a[i,j] from x[i] for i < j
    for (var i = tid; i < j; i += ${workgroupSize}u) {
      x[bx_base + i] -= x_j * a[a_base + i * ${n}u + j];
    }
    workgroupBarrier();
    storageBarrier();
  }
}
`.trim(),
    numInputs: 2,
    numOutputs: 1,
    hasUniform: false,
    passes: [{ grid: calculateGrid(numMatrices * numRhs) }]
  }];
}
function choleskyUniform(phase, k, blockSize, rowsBelow) {
  const ar = /* @__PURE__ */ new Uint32Array(4);
  ar[0] = phase;
  ar[1] = k;
  ar[2] = blockSize;
  ar[3] = rowsBelow;
  return new Uint8Array(ar.buffer);
}
function createCholesky(device, type) {
  const dtype = type.inputDtypes[0];
  const shape2 = type.inputShapes[0];
  const n = shape2[shape2.length - 1];
  const batches = prod(shape2.slice(0, -2));
  const needsF16 = dtype === "float16";
  const ty = dtypeToWgsl(dtype, true);
  const workgroupSize = Math.min(256, findPow2(0, device.limits.maxComputeWorkgroupSizeX));
  const useBlocked = n >= CHOLESKY_BLOCK_THRESHOLD;
  const blockSize = useBlocked ? CHOLESKY_BLOCK_SIZE : n;
  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;

struct CholeskyParams {
  phase: u32,
  k: u32,
  block_size: u32,
  rows_below: u32,
}

@group(1) @binding(0) var<uniform> params: CholeskyParams;

// Shared memory for the diagonal element
var<workgroup> L_jj: ${ty};

fn mat_idx(base: u32, row: u32, col: u32) -> u32 {
  return base + row * ${n}u + col;
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let tid = local_id.x;

  if (params.phase == ${CholeskyPhase.Unblocked}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;

    // Zero out output and copy lower triangle from input.
    for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
      let row = idx / ${n}u;
      let col = idx % ${n}u;
      output[base + idx] = select(0, input[base + idx], col <= row);
    }
    storageBarrier();

    // Cholesky-Crout algorithm: process column by column.
    for (var j = 0u; j < ${n}u; j++) {
      for (var i = j + tid; i < ${n}u; i += ${workgroupSize}u) {
        var sum = output[mat_idx(base, i, j)];
        for (var k = 0u; k < j; k++) {
          sum -= output[mat_idx(base, i, k)] * output[mat_idx(base, j, k)];
        }
        output[mat_idx(base, i, j)] = sum;
      }
      storageBarrier();

      if (tid == 0u) {
        L_jj = sqrt(output[mat_idx(base, j, j)]);
        output[mat_idx(base, j, j)] = L_jj;
      }
      workgroupBarrier();

      for (var i = j + 1u + tid; i < ${n}u; i += ${workgroupSize}u) {
        output[mat_idx(base, i, j)] /= L_jj;
      }
      storageBarrier();
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.InitBlocked}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;

    for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
      let row = idx / ${n}u;
      let col = idx % ${n}u;
      output[base + idx] = select(0, input[base + idx], col <= row);
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.FactorBlock}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;
    let k0 = params.k;
    let b = params.block_size;

    for (var j = 0u; j < b; j++) {
      let col = k0 + j;
      for (var i = j + tid; i < b; i += ${workgroupSize}u) {
        let row = k0 + i;
        var sum = output[mat_idx(base, row, col)];
        for (var r = 0u; r < j; r++) {
          sum -= output[mat_idx(base, row, k0 + r)] *
            output[mat_idx(base, col, k0 + r)];
        }
        output[mat_idx(base, row, col)] = sum;
      }
      storageBarrier();

      if (tid == 0u) {
        L_jj = sqrt(output[mat_idx(base, col, col)]);
        output[mat_idx(base, col, col)] = L_jj;
      }
      workgroupBarrier();

      for (var i = j + 1u + tid; i < b; i += ${workgroupSize}u) {
        output[mat_idx(base, k0 + i, col)] /= L_jj;
      }
      storageBarrier();
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.SolvePanel}u) {
    let rows = params.rows_below;
    if (rows == 0u) {
      return;
    }

    let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;
    if (global >= rows * ${batches}u) {
      return;
    }

    let batch = global / rows;
    let row = params.k + params.block_size + (global % rows);
    let base = batch * ${n * n}u;

    for (var j = 0u; j < params.block_size; j++) {
      let col = params.k + j;
      var sum = output[mat_idx(base, row, col)];
      for (var r = 0u; r < j; r++) {
        sum -= output[mat_idx(base, row, params.k + r)] *
          output[mat_idx(base, col, params.k + r)];
      }
      output[mat_idx(base, row, col)] =
        sum / output[mat_idx(base, col, col)];
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.UpdateTrailing}u) {
    let rows = params.rows_below;
    if (rows == 0u) {
      return;
    }

    let tile_count = (rows + ${CHOLESKY_UPDATE_TILE_SIZE - 1}u) / ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let tiles_per_batch = tile_count * (tile_count + 1u) / 2u;
    let wg_global = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (wg_global >= tiles_per_batch * ${batches}u) {
      return;
    }

    let batch = wg_global / tiles_per_batch;
    let tile = wg_global % tiles_per_batch;
    var tile_row = u32((sqrt(f32(8u * tile + 1u)) - 1.0) * 0.5);
    // Correct for GPU sqrt rounding around exact triangular-number boundaries.
    var row_start = tile_row * (tile_row + 1u) / 2u;
    if (row_start > tile) {
      tile_row -= 1u;
      row_start = tile_row * (tile_row + 1u) / 2u;
    }
    let next_row_start = (tile_row + 1u) * (tile_row + 2u) / 2u;
    if (tile >= next_row_start) {
      tile_row += 1u;
      row_start = next_row_start;
    }
    let tile_col = tile - row_start;
    let local_row = tid / ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let local_col = tid % ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let row_local = tile_row * ${CHOLESKY_UPDATE_TILE_SIZE}u + local_row;
    let col_local = tile_col * ${CHOLESKY_UPDATE_TILE_SIZE}u + local_col;
    if (row_local >= rows || col_local >= rows) {
      return;
    }
    if (row_local < col_local) {
      return;
    }

    let row = params.k + params.block_size + row_local;
    let col = params.k + params.block_size + col_local;
    let base = batch * ${n * n}u;
    var sum = output[mat_idx(base, row, col)];
    for (var r = 0u; r < params.block_size; r++) {
      sum -= output[mat_idx(base, row, params.k + r)] *
        output[mat_idx(base, col, params.k + r)];
    }
    output[mat_idx(base, row, col)] = sum;
  }
}
`.trim();
  const passes = [];
  if (!useBlocked) passes.push({
    grid: calculateGrid(batches),
    uniform: choleskyUniform(CholeskyPhase.Unblocked, 0, n, 0)
  });
  else {
    passes.push({
      grid: calculateGrid(batches),
      uniform: choleskyUniform(CholeskyPhase.InitBlocked, 0, blockSize, 0)
    });
    for (let k = 0; k < n; k += blockSize) {
      const b = Math.min(blockSize, n - k);
      const rowsBelow = n - k - b;
      passes.push({
        grid: calculateGrid(batches),
        uniform: choleskyUniform(CholeskyPhase.FactorBlock, k, b, rowsBelow)
      });
      if (rowsBelow > 0) {
        passes.push({
          grid: calculateGrid(Math.ceil(rowsBelow * batches / workgroupSize)),
          uniform: choleskyUniform(CholeskyPhase.SolvePanel, k, b, rowsBelow)
        });
        passes.push({
          grid: calculateGrid(Math.ceil(rowsBelow / CHOLESKY_UPDATE_TILE_SIZE) * (Math.ceil(rowsBelow / CHOLESKY_UPDATE_TILE_SIZE) + 1) * batches / 2),
          uniform: choleskyUniform(CholeskyPhase.UpdateTrailing, k, b, rowsBelow)
        });
      }
    }
  }
  return [{
    code,
    numInputs: 1,
    numOutputs: 1,
    hasUniform: true,
    passes
  }];
}
function createLU(device, type) {
  const dtype = type.inputDtypes[0];
  const shape2 = type.inputShapes[0];
  const m = shape2[shape2.length - 2];
  const n = shape2[shape2.length - 1];
  const r = Math.min(m, n);
  const batches = prod(shape2.slice(0, -2));
  const needsF16 = dtype === "float16";
  const ty = dtypeToWgsl(dtype, true);
  const workgroupSize = findPow2(Math.max(m, n), device.limits.maxComputeWorkgroupSizeX);
  return [{
    code: `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> lu: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> pivots: array<i32>;
@group(0) @binding(3) var<storage, read_write> perm: array<i32>;

var<workgroup> pivot_row: u32;
var<workgroup> pivot_val: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) {
    return;
  }

  let lu_base = batch * ${m * n}u;
  let piv_base = batch * ${r}u;
  let perm_base = batch * ${m}u;
  let tid = local_id.x;

  // Copy input to lu
  for (var idx = tid; idx < ${m * n}u; idx += ${workgroupSize}u) {
    lu[lu_base + idx] = input[lu_base + idx];
  }
  // Initialize permutation
  for (var idx = tid; idx < ${m}u; idx += ${workgroupSize}u) {
    perm[perm_base + idx] = i32(idx);
  }
  storageBarrier();

  // LU decomposition with partial pivoting
  for (var j = 0u; j < ${r}u; j++) {
    // Step 1: Thread 0 finds pivot (max abs value in column j, rows >= j)
    if (tid == 0u) {
      var max_val = abs(lu[lu_base + j * ${n}u + j]);
      var max_row = j;
      for (var i = j + 1u; i < ${m}u; i++) {
        let val = abs(lu[lu_base + i * ${n}u + j]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }
      pivot_row = max_row;
      pivot_val = lu[lu_base + max_row * ${n}u + j];
      pivots[piv_base + j] = i32(max_row);
    }
    workgroupBarrier();

    // Step 2: Swap rows j and pivot_row (threads collaborate)
    let pr = pivot_row;
    if (pr != j) {
      for (var col = tid; col < ${n}u; col += ${workgroupSize}u) {
        let tmp = lu[lu_base + j * ${n}u + col];
        lu[lu_base + j * ${n}u + col] = lu[lu_base + pr * ${n}u + col];
        lu[lu_base + pr * ${n}u + col] = tmp;
      }
      if (tid == 0u) {
        let tmp_p = perm[perm_base + j];
        perm[perm_base + j] = perm[perm_base + pr];
        perm[perm_base + pr] = tmp_p;
      }
    }
    storageBarrier();

    // Step 3: Compute L[i][j] and update submatrix
    // Each thread handles one row i > j
    for (var i = j + 1u + tid; i < ${m}u; i += ${workgroupSize}u) {
      let factor = lu[lu_base + i * ${n}u + j] / pivot_val;
      lu[lu_base + i * ${n}u + j] = factor; // L[i][j]
      for (var k = j + 1u; k < ${n}u; k++) {
        lu[lu_base + i * ${n}u + k] -= factor * lu[lu_base + j * ${n}u + k];
      }
    }
    storageBarrier();
  }
}
`.trim(),
    numInputs: 1,
    numOutputs: 3,
    hasUniform: false,
    passes: [{ grid: calculateGrid(batches) }]
  }];
}
function createJacobiEigh(device, type, params) {
  const dtype = type.inputDtypes[0];
  const shape2 = type.inputShapes[0];
  const n = shape2[shape2.length - 1];
  const batches = prod(shape2.slice(0, -2));
  const needsF16 = dtype === "float16";
  const ty = dtypeToWgsl(dtype, true);
  const tolerance = `${ty}(${params.tolerance})`;
  const workgroupSize = findPow2(Math.max(n, 1), device.limits.maxComputeWorkgroupSizeX);
  return [{
    code: `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> diagonalized: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> vectors: array<${ty}>;

var<workgroup> done: u32;
var<workgroup> rot_active: u32;
var<workgroup> rot_c: ${ty};
var<workgroup> rot_s: ${ty};
var<workgroup> rot_app: ${ty};
var<workgroup> rot_aqq: ${ty};
var<workgroup> rot_apq: ${ty};

fn mat_idx(base: u32, row: u32, col: u32) -> u32 {
  return base + row * ${n}u + col;
}

fn sym_idx(base: u32, row: u32, col: u32) -> u32 {
  return mat_idx(base, max(row, col), min(row, col));
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) {
    return;
  }

  let base = batch * ${n * n}u;
  let tid = local_id.x;

  for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
    let row = idx / ${n}u;
    let col = idx % ${n}u;
    diagonalized[base + idx] = select(
      ${ty}(0),
      input[base + idx],
      row >= col,
    );
    vectors[base + idx] = select(${ty}(0), ${ty}(1), row == col);
  }
  storageBarrier();

  for (var sweep = 0u; sweep < ${params.maxSweeps}u; sweep++) {
    if (tid == 0u) {
      var max_abs = ${ty}(1);
      var max_offdiag = ${ty}(0);
      for (var idx = 0u; idx < ${n * n}u; idx++) {
        let row = idx / ${n}u;
        let col = idx % ${n}u;
        let value = abs(diagonalized[base + idx]);
        max_abs = max(max_abs, value);
        if (row > col) {
          max_offdiag = max(max_offdiag, value);
        }
      }
      done = select(0u, 1u, max_offdiag <= ${tolerance} * max_abs);
    }
    let done_uniform = workgroupUniformLoad(&done);
    if (done_uniform != 0u) {
      break;
    }

    for (var p = 0u; p + 1u < ${n}u; p++) {
      for (var q = p + 1u; q < ${n}u; q++) {
        if (tid == 0u) {
          rot_app = diagonalized[mat_idx(base, p, p)];
          rot_aqq = diagonalized[mat_idx(base, q, q)];
          rot_apq = diagonalized[sym_idx(base, p, q)];
          if (rot_apq == ${ty}(0)) {
            rot_active = 0u;
            rot_c = ${ty}(1);
            rot_s = ${ty}(0);
          } else {
            let tau = (rot_aqq - rot_app) / (${ty}(2) * rot_apq);
            let tau_sign = select(${ty}(-1), ${ty}(1), tau >= ${ty}(0));
            let t = tau_sign / (abs(tau) + sqrt(tau * tau + ${ty}(1)));
            rot_c = inverseSqrt(t * t + ${ty}(1));
            rot_s = t * rot_c;
            rot_active = 1u;
          }
        }
        workgroupBarrier();

        if (rot_active != 0u) {
          for (var k = tid; k < ${n}u; k += ${workgroupSize}u) {
            if (k != p && k != q) {
              let kp = sym_idx(base, k, p);
              let kq = sym_idx(base, k, q);
              let akp = diagonalized[kp];
              let akq = diagonalized[kq];
              let next_kp = rot_c * akp - rot_s * akq;
              let next_kq = rot_s * akp + rot_c * akq;
              diagonalized[kp] = next_kp;
              diagonalized[kq] = next_kq;
            } else if (k == p) {
              diagonalized[mat_idx(base, p, p)] =
                rot_c * rot_c * rot_app - ${ty}(2) * rot_s * rot_c * rot_apq + rot_s * rot_s * rot_aqq;
              diagonalized[sym_idx(base, p, q)] = ${ty}(0);
            } else {
              diagonalized[mat_idx(base, q, q)] =
                rot_s * rot_s * rot_app + ${ty}(2) * rot_s * rot_c * rot_apq + rot_c * rot_c * rot_aqq;
            }

            let vp = mat_idx(base, k, p);
            let vq = mat_idx(base, k, q);
            let vkp = vectors[vp];
            let vkq = vectors[vq];
            vectors[vp] = rot_c * vkp - rot_s * vkq;
            vectors[vq] = rot_s * vkp + rot_c * vkq;
          }
        }
        storageBarrier();
      }
    }
  }
}
`.trim(),
    numInputs: 1,
    numOutputs: 2,
    hasUniform: false,
    passes: [{ grid: calculateGrid(batches) }]
  }];
}
function fftUniform(phase, radix, prev, normalize) {
  return new Uint8Array(new Uint32Array([
    phase,
    radix,
    prev,
    normalize ? 1 : 0
  ]).buffer);
}
function createFft(device, type, params) {
  const dtype = type.inputDtypes[0];
  const shape2 = type.inputShapes[0];
  const n = shape2[shape2.length - 1];
  const batches = prod(shape2.slice(0, -1));
  if (prod(params.factors) !== n) throw new Error(`fft: factorization ${params.factors} does not match size ${n}`);
  const needsF16 = dtype === "float16";
  const ty = dtypeToWgsl(dtype, true);
  const workgroupSize = Math.min(256, findPow2(0, device.limits.maxComputeWorkgroupSizeX));
  const maxFactor = Math.max(1, ...params.factors);
  const angleScale = params.inverse ? "6.283185307179586" : "-6.283185307179586";
  const digitReversal = params.factors.map((factor) => `
  digit = remaining % ${factor}u;
  remaining = remaining / ${factor}u;
  stride = stride * ${factor}u;
  reversed = reversed + digit * (${n}u / stride);`).join("");
  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input_real: array<${ty}>;
@group(0) @binding(1) var<storage, read> input_imag: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> output_real: array<${ty}>;
@group(0) @binding(3) var<storage, read_write> output_imag: array<${ty}>;

struct FftParams {
  phase: u32,
  radix: u32,
  prev: u32,
  normalize: u32,
}

@group(1) @binding(0) var<uniform> fft_params: FftParams;

fn digit_reversed_index(index: u32) -> u32 {
  var remaining = index;
  var stride = 1u;
  var reversed = 0u;
  var digit = 0u;
${digitReversal}
  return reversed;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;

  if (fft_params.phase == 0u) {
    if (global >= ${batches * n}u) {
      return;
    }
    let batch = global / ${n}u;
    let out_idx = global % ${n}u;
    let source = batch * ${n}u + digit_reversed_index(out_idx);
    output_real[global] = input_real[source];
    output_imag[global] = input_imag[source];
    return;
  }

  let butterflies_per_batch = ${n}u / fft_params.radix;
  if (global >= ${batches}u * butterflies_per_batch) {
    return;
  }

  let batch = global / butterflies_per_batch;
  let local = global % butterflies_per_batch;
  let j = local % fft_params.prev;
  let group = local / fft_params.prev;
  let span = fft_params.prev * fft_params.radix;
  let start = batch * ${n}u + group * span + j;
  let scale = select(1.0, 1.0 / f32(${n}u), fft_params.normalize != 0u);

  var scratch_real: array<f32, ${maxFactor}>;
  var scratch_imag: array<f32, ${maxFactor}>;

  for (var q = 0u; q < fft_params.radix; q++) {
    let idx = start + q * fft_params.prev;
    let angle = ${angleScale} * f32(q * j) / f32(span);
    let c = cos(angle);
    let s = sin(angle);
    let xr = f32(output_real[idx]);
    let xi = f32(output_imag[idx]);
    scratch_real[q] = xr * c - xi * s;
    scratch_imag[q] = xr * s + xi * c;
  }

  for (var p = 0u; p < fft_params.radix; p++) {
    var sum_real = 0.0;
    var sum_imag = 0.0;
    for (var q = 0u; q < fft_params.radix; q++) {
      let angle = ${angleScale} * f32(q * p) / f32(fft_params.radix);
      let c = cos(angle);
      let s = sin(angle);
      let xr = scratch_real[q];
      let xi = scratch_imag[q];
      sum_real += xr * c - xi * s;
      sum_imag += xr * s + xi * c;
    }
    let idx = start + p * fft_params.prev;
    output_real[idx] = ${ty}(sum_real * scale);
    output_imag[idx] = ${ty}(sum_imag * scale);
  }
}
`.trim();
  const passes = [{
    grid: calculateGrid(Math.ceil(batches * n / workgroupSize)),
    uniform: fftUniform(0, 1, 1, false)
  }];
  let prev = 1;
  for (let i = 0; i < params.factors.length; i++) {
    const radix = params.factors[i];
    passes.push({
      grid: calculateGrid(Math.ceil(batches * n / radix / workgroupSize)),
      uniform: fftUniform(1, radix, prev, params.inverse && i === params.factors.length - 1)
    });
    prev *= radix;
  }
  return [{
    code,
    numInputs: 2,
    numOutputs: 2,
    hasUniform: true,
    passes
  }];
}
function createRoutineShader(device, routine) {
  switch (routine.name) {
    case "Sort":
      return createSort(device, routine.type);
    case "Argsort":
      return createArgsort(device, routine.type);
    case "Scatter":
      return createScatter(device, routine.type, routine.params);
    case "TriangularSolve":
      return createTriangularSolve(device, routine.type, routine.params);
    case "Cholesky":
      return createCholesky(device, routine.type);
    case "LU":
      return createLU(device, routine.type);
    case "JacobiEigh":
      return createJacobiEigh(device, routine.type, routine.params);
    case "Fft":
      return createFft(device, routine.type, routine.params);
    default:
      throw new UnsupportedRoutineError(routine.name, "webgpu");
  }
}
function createTracingBatch(device) {
  return {
    querySet: device.createQuerySet({
      type: "timestamp",
      count: MAX_TIMESTAMP_QUERIES
    }),
    resolve: device.createBuffer({
      size: MAX_TIMESTAMP_QUERIES * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    }),
    dst: device.createBuffer({
      size: MAX_TIMESTAMP_QUERIES * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    }),
    nextIndex: 0,
    entries: []
  };
}
function acquireTracingSlot(device) {
  if (!device.features.has("timestamp-query")) return void 0;
  let batch = activeBatch.get(device);
  if (batch && batch.nextIndex >= MAX_TIMESTAMP_QUERIES) {
    flushTracingBatch(device, batch);
    batch = void 0;
  }
  if (!batch) {
    batch = createTracingBatch(device);
    activeBatch.set(device, batch);
    onFlushTrace(() => {
      const b = activeBatch.get(device);
      if (b && b.entries.length > 0) flushTracingBatch(device, b);
      activeBatch.delete(device);
    });
  }
  const beginIndex = batch.nextIndex;
  const endIndex = beginIndex + 1;
  batch.nextIndex += 2;
  return {
    batch,
    beginIndex,
    endIndex
  };
}
function maybeAcquireTracingSlot(device) {
  if (!isTracing()) return void 0;
  return acquireTracingSlot(device);
}
function recordTrace(device, slot, source, numPasses, wgslSource) {
  const info = traceSourceInfo(source);
  info.properties.push(["passes", `${numPasses}`]);
  info.properties.push(["source", wgslSource]);
  slot.batch.entries.push({
    ...info,
    beginIndex: slot.beginIndex,
    endIndex: slot.endIndex
  });
  scheduleAutoFlush(device);
}
function scheduleAutoFlush(device) {
  queueMicrotask(() => {
    const batch = activeBatch.get(device);
    if (batch && batch.entries.length > 0) {
      flushTracingBatch(device, batch);
      activeBatch.set(device, createTracingBatch(device));
    }
  });
}
function flushTracingBatch(device, batch) {
  if (batch.entries.length === 0) return;
  const usedQueries = batch.nextIndex;
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(batch.querySet, 0, usedQueries, batch.resolve, 0);
  encoder.copyBufferToBuffer(batch.resolve, 0, batch.dst, 0, usedQueries * 8);
  device.queue.submit([encoder.finish()]);
  const { entries } = batch;
  batch.dst.mapAsync(GPUMapMode.READ).then(() => {
    try {
      const times = new BigInt64Array(batch.dst.getMappedRange());
      const anchorGpuNs = times[entries[entries.length - 1].endIndex];
      const anchorCpuMs = performance.now();
      for (const entry of entries) emitTrace("webgpu", entry, anchorCpuMs + Number(times[entry.beginIndex] - anchorGpuNs) / 1e6, anchorCpuMs + Number(times[entry.endIndex] - anchorGpuNs) / 1e6);
    } finally {
      batch.dst.unmap();
      batch.querySet.destroy();
      batch.resolve.destroy();
      batch.dst.destroy();
    }
  });
}
function pipelineSource(device, kernel) {
  const nullaryKernel = nullaryKernelSource(device, kernel);
  if (nullaryKernel) return nullaryKernel;
  const tune = tuneWebgpu(kernel);
  if (DEBUG >= 3) console.info(`kernel.exp: ${kernel.exp}
tune.exp: ${tune.exp}`);
  const { nargs, reduction: re } = kernel;
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);
  const wb = new WgslBuilder();
  wb.emitPreamble(device, [tune.exp, tune.epilogue]);
  const usedArgs = Array.from({ length: nargs }, () => null);
  tune.exp.fold((exp3) => {
    if (exp3.op === "GlobalIndex") usedArgs[exp3.arg[0]] = exp3.dtype;
  });
  tune.epilogue?.fold((exp3) => {
    if (exp3.op === "GlobalIndex") usedArgs[exp3.arg[0]] = exp3.dtype;
  });
  for (let i = 0; i < nargs; i++) {
    const ty = dtypeToWgsl(usedArgs[i] ?? "float32", true);
    wb.emit(`@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`);
  }
  const resultTy = dtypeToWgsl(kernel.dtype, true);
  wb.emit(`@group(0) @binding(${nargs}) var<storage, read_write> result : array<${resultTy}>;`);
  const groupCount = re ? tune.size.groups ?? 1 : 1;
  const groupedReduction = re && groupCount > 1;
  if (groupedReduction && tune.threadCount % groupCount !== 0) throw new Error("WebGPU grouped reduction has invalid thread count");
  if (groupedReduction && groupCount > device.limits.maxComputeWorkgroupSizeX) throw new Error("WebGPU grouped reduction exceeds workgroup size limit");
  const workgroupSize = groupedReduction ? groupCount : findPow2(tune.threadCount, 256);
  const gridSize = groupedReduction ? tune.threadCount / groupCount : Math.ceil(tune.threadCount / workgroupSize);
  const [gridX, gridY] = calculateGrid(gridSize);
  if (groupedReduction) {
    const partialTy = dtypeToWgsl(re.dtype);
    for (let i = 0; i < (tune.size.upcast ?? 1); i++) wb.emit(`var<workgroup> partial${i}: array<${partialTy}, ${groupCount}>;`);
  }
  wb.emit("", `@compute @workgroup_size(${workgroupSize})`);
  if (groupedReduction) {
    wb.emit("fn main(", wb.pushIndent, "@builtin(local_invocation_id) lid : vec3<u32>,", "@builtin(workgroup_id) wg_id : vec3<u32>,", wb.popIndent, ") {", wb.pushIndent);
    if (gridY === 1) wb.emit(`if (wg_id.x >= ${gridSize}u) { return; }`, "let gidx: i32 = i32(wg_id.x);");
    else wb.emit(`if (${gridX}u * wg_id.y + wg_id.x >= ${gridSize}u) { return; }`, `let gidx: i32 = i32(${gridX}u * wg_id.y + wg_id.x);`);
    wb.emit("let group: i32 = i32(lid.x);");
  } else {
    wb.emit("fn main(@builtin(global_invocation_id) id : vec3<u32>) {", wb.pushIndent);
    if (gridY === 1) wb.emit(`if (id.x >= ${tune.threadCount}) { return; }`, "let gidx: i32 = i32(id.x);");
    else {
      const sizeX = gridX * workgroupSize;
      wb.emit(`if (${sizeX} * id.y + id.x >= ${tune.threadCount}) { return; }`, `let gidx: i32 = i32(${sizeX} * id.y + id.x);`);
    }
  }
  wb.emitPhonyAssignments(args);
  const gen = new WgslExpCodegen(wb, args);
  if (!re) {
    gen.countReferences(tune.exp);
    let rhs = strip1(gen.run(tune.exp));
    if (resultTy !== dtypeToWgsl(tune.exp.dtype)) rhs = `${resultTy}(${rhs})`;
    wb.emit(`result[gidx] = ${rhs};`);
  } else {
    const unroll = tune.size.unroll ?? 1;
    const upcast = tune.size.upcast ?? 1;
    const acc = [...Array(upcast)].map((_, i) => `acc${i}`);
    for (let i = 0; i < upcast; i++) wb.emit(`var ${acc[i]}: ${dtypeToWgsl(re.dtype)} = ${constToWgsl(re.dtype, re.identity)};`);
    wb.emit(`for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`, wb.pushIndent);
    const exps = [];
    const cache = /* @__PURE__ */ new Map();
    for (let up = 0; up < upcast; up++) {
      exps.push([]);
      for (let un = 0; un < unroll; un++) {
        const exp3 = tune.exp.substitute({
          upcast: AluExp.i32(up),
          unroll: AluExp.i32(un)
        });
        exps[up].push(exp3.simplify(cache));
        gen.countReferences(exps[up][un]);
      }
    }
    const items = exps.map((ar) => ar.map((x) => gen.run(x)).map(strip1));
    for (let i = 0; i < upcast; i++) {
      let rhs = items[i][0];
      for (let j = 1; j < unroll; j++) if (re.op === "Add") rhs = `${rhs} + ${items[i][j]}`;
      else if (re.op === "Mul") rhs = `${rhs} * ${items[i][j]}`;
      else if (re.op === "Min") rhs = re.dtype === "bool" ? `(${rhs} && ${items[i][j]})` : `min(${rhs}, ${items[i][j]})`;
      else if (re.op === "Max") rhs = re.dtype === "bool" ? `(${rhs} || ${items[i][j]})` : `max(${rhs}, ${items[i][j]})`;
      else throw new Error(`Unsupported reduction op: ${re.op}`);
      if (re.op === "Add") wb.emit(`${acc[i]} += ${rhs};`);
      else if (re.op === "Mul") wb.emit(`${acc[i]} *= ${rhs};`);
      else if (re.op === "Min") if (re.dtype === "bool") wb.emit(`${acc[i]} = ${acc[i]} && ${rhs};`);
      else wb.emit(`${acc[i]} = min(${acc[i]}, ${rhs});`);
      else if (re.op === "Max") if (re.dtype === "bool") wb.emit(`${acc[i]} = ${acc[i]} || ${rhs};`);
      else wb.emit(`${acc[i]} = max(${acc[i]}, ${rhs});`);
      else throw new Error(`Unsupported reduction op: ${re.op}`);
    }
    wb.emit(wb.popIndent, "}");
    if (groupedReduction) {
      for (let i = 0; i < upcast; i++) wb.emit(`partial${i}[lid.x] = ${acc[i]};`);
      wb.emit("workgroupBarrier();");
      for (let stride = groupCount / 2; stride >= 1; stride /= 2) {
        wb.emit(`if (lid.x < ${stride}u) {`, wb.pushIndent);
        for (let i = 0; i < upcast; i++) wb.emit(`partial${i}[lid.x] = ${reduceOpWgsl(re.op, re.dtype, `partial${i}[lid.x]`, `partial${i}[lid.x + ${stride}u]`)};`);
        wb.emit(wb.popIndent, "}", "workgroupBarrier();");
      }
    }
    gen.reset();
    const outputIdxExps = [];
    const fusionExps = [];
    for (let i = 0; i < upcast; i++) {
      const exp3 = tune.outputIdxExp.substitute({ upcast: AluExp.i32(i) });
      outputIdxExps.push(exp3.simplify(cache));
      gen.countReferences(outputIdxExps[i]);
      fusionExps.push(tune.epilogue.substitute({
        acc: AluExp.variable(re.dtype, acc[i]),
        upcast: AluExp.i32(i)
      }).simplify(cache));
      gen.countReferences(fusionExps[i]);
    }
    if (groupedReduction) {
      wb.emit("if (lid.x == 0u) {", wb.pushIndent);
      for (let i = 0; i < upcast; i++) wb.emit(`${acc[i]} = partial${i}[0u];`);
    }
    for (let i = 0; i < upcast; i++) {
      const index = strip1(gen.run(outputIdxExps[i]));
      let rhs = strip1(gen.run(fusionExps[i]));
      if (resultTy !== dtypeToWgsl(fusionExps[i].dtype)) rhs = `${resultTy}(${rhs})`;
      wb.emit(`result[${index}] = ${rhs};`);
    }
    if (groupedReduction) wb.emit(wb.popIndent, "}");
  }
  wb.emit(wb.popIndent, "}");
  return {
    code: wb.toString(),
    numInputs: nargs,
    numOutputs: 1,
    hasUniform: false,
    passes: [{ grid: [gridX, gridY] }]
  };
}
function pipelineSubmit(device, exe, inputs, outputs) {
  const { data: pipelines, source } = exe;
  const commandEncoder = device.createCommandEncoder();
  for (const { pipeline, ...shader } of pipelines) {
    if (inputs.length !== shader.numInputs || outputs.length !== shader.numOutputs) throw new Error(`webgpu: expected ${shader.numInputs} inputs and ${shader.numOutputs} outputs, got ${inputs.length} inputs and ${outputs.length} outputs`);
    if (shader.clearOutputs) for (const output of outputs) commandEncoder.clearBuffer(output);
    const filteredPasses = shader.passes.filter(({ grid }) => prod(grid) > 0);
    if (filteredPasses.length === 0) continue;
    const slot = maybeAcquireTracingSlot(device);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [...inputs.map((buffer, i) => ({
        binding: i,
        resource: { buffer }
      })), ...outputs.map((buffer, i) => ({
        binding: inputs.length + i,
        resource: { buffer }
      }))]
    });
    let uniformBindGroup = null;
    let uniformAlignment = 0;
    if (shader.hasUniform) {
      const [uniformBuffer, alignment] = combineUniforms(device, filteredPasses.map(({ uniform: uniform3 }) => uniform3));
      uniformAlignment = alignment;
      uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [{
          binding: 0,
          resource: {
            buffer: uniformBuffer,
            size: alignment
          }
        }]
      });
    }
    for (let i = 0; i < filteredPasses.length; i++) {
      const { grid } = filteredPasses[i];
      let timestampWrites;
      if (slot) {
        const isFirst = i === 0;
        const isLast = i === filteredPasses.length - 1;
        if (isFirst || isLast) timestampWrites = {
          querySet: slot.batch.querySet,
          ...isFirst ? { beginningOfPassWriteIndex: slot.beginIndex } : {},
          ...isLast ? { endOfPassWriteIndex: slot.endIndex } : {}
        };
      }
      const passEncoder = commandEncoder.beginComputePass({ timestampWrites });
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      if (uniformBindGroup) passEncoder.setBindGroup(1, uniformBindGroup, [i * uniformAlignment]);
      passEncoder.dispatchWorkgroups(grid[0], grid[1]);
      passEncoder.end();
    }
    if (slot) recordTrace(device, slot, source, filteredPasses.length, shader.code);
  }
  device.queue.submit([commandEncoder.finish()]);
}
function combineUniforms(device, uniforms) {
  for (const buf of uniforms) if (!buf || buf.byteLength === 0 || buf.byteLength !== uniforms[0].byteLength) throw new Error("webgpu: Uniform mismatch between shader passes");
  const minAlign = device.limits.minUniformBufferOffsetAlignment;
  const alignment = Math.ceil(uniforms[0].byteLength / minAlign) * minAlign;
  const buffer = device.createBuffer({
    size: alignment * uniforms.length,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true
  });
  const bufferMapped = new Uint8Array(buffer.getMappedRange());
  for (let i = 0; i < uniforms.length; i++) bufferMapped.set(uniforms[i], i * alignment);
  buffer.unmap();
  return [buffer, alignment];
}
async function compileError(shaderModule, scope, code) {
  let message = `Failed to compile shader: ${scope ? scope.message : "(no error scope)"}`;
  const info = await shaderModule.getCompilationInfo();
  for (const msg of info.messages) message += `
  [${msg.type} at ${msg.lineNum}:${msg.linePos}] ${msg.message}`;
  if (code) message += `

${code}`;
  return message;
}
var threefrySrc, erfSrc, headerWgsl, WgslBuilder, WgslExpCodegen, gridOffsetY, NUDGE_INTERVAL_MS, BUSY_MS, BACKOFF_MS, DEADLINE_MS, isFirefox, channel, resolvers, SyncReader, CholeskyPhase, CHOLESKY_BLOCK_SIZE, CHOLESKY_BLOCK_THRESHOLD, CHOLESKY_UPDATE_TILE_SIZE, MAX_TIMESTAMP_QUERIES, activeBatch, MAX_REUSABLE_BUFFER_BYTES, MAX_REUSABLE_BUFFERS_PER_SIZE, WebGPUBackend, ShaderPipelineCache;
var init_webgpu_Dluq5ZOf = __esm({
  "node_modules/@jax-js/jax/dist/webgpu-Dluq5ZOf.js"() {
    init_backend_D_Uwkp6d();
    threefrySrc = `
fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
  let ks0: u32 = key.x;
  let ks1: u32 = key.y;
  let ks2: u32 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  var x0: u32 = ctr.x + ks0;
  var x1: u32 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return vec2<u32>(x0, x1);
}`;
    erfSrc = `
const _erf_p: f32 = 0.3275911;
const _erf_a1: f32 = 0.254829592;
const _erf_a2: f32 = -0.284496736;
const _erf_a3: f32 = 1.421413741;
const _erf_a4: f32 = -1.453152027;
const _erf_a5: f32 = 1.061405429;
fn erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
fn erfc(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  let E = P_t * exp(-x * x);
  return select(2.0 - E, E, x >= 0.0);
}`;
    headerWgsl = String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();
    WgslBuilder = class {
      pushIndent = Symbol("pushIndent");
      popIndent = Symbol("popIndent");
      lines = [];
      #indent = "";
      emit(...lines) {
        for (const line of lines) if (line === this.pushIndent) this.#indent += "  ";
        else if (line === this.popIndent) this.#indent = this.#indent.slice(0, -2);
        else this.lines.push(line ? this.#indent + line : "");
      }
      emitPreamble(device, exps) {
        let hasFloat16 = false;
        let distinctOps = /* @__PURE__ */ new Map();
        for (const exp3 of exps) {
          if (exp3 == null) continue;
          hasFloat16 ||= exp3.some((e2) => e2.dtype === "float16");
          distinctOps = mapSetUnion(distinctOps, exp3.distinctOps());
        }
        if (hasFloat16) {
          if (!device.features.has("shader-f16")) throw new Error("WebGPU device does not support shader-f16 feature");
          this.emit("enable f16;");
        }
        this.emit(headerWgsl);
        if (distinctOps.has("Threefry2x32")) this.emit(threefrySrc);
        if (distinctOps.has("Erf") || distinctOps.has("Erfc")) this.emit(erfSrc);
        this.emit("");
      }
      /**
      * Insert phony assignments, in case some inputs are not in use.
      * <https://github.com/gpuweb/gpuweb/discussions/4582#discussioncomment-9146686>
      */
      emitPhonyAssignments(args) {
        if (args.length > 0) this.emit(args.map((arg) => `_ = &${arg};`).join(" "));
      }
      toString() {
        return this.lines.join("\n");
      }
    };
    WgslExpCodegen = class {
      wb;
      args;
      #gensymCount = 0;
      #references = /* @__PURE__ */ new Map();
      #seen = /* @__PURE__ */ new Set();
      #context = /* @__PURE__ */ new Map();
      constructor(wb, args) {
        this.wb = wb;
        this.args = args;
      }
      #gensym() {
        return `alu${this.#gensymCount++}`;
      }
      #isGensym(text) {
        return text.match(/^alu[0-9]+$/);
      }
      /**
      * Count references for an expression.
      *
      * Used to get an ahead-of-time reference count for each node in the AluExp.
      * Expressions with reference count greater than 1 are stored in temporary
      * variables to avoid recomputation.
      */
      countReferences(exp3) {
        this.#references.set(exp3, (this.#references.get(exp3) ?? 0) + 1);
        if (!this.#seen.has(exp3)) {
          this.#seen.add(exp3);
          for (const src of exp3.src) this.countReferences(src);
        }
      }
      reset() {
        this.#references.clear();
        this.#seen.clear();
        this.#context.clear();
      }
      /**
      * Generate code for an expression.
      *
      * Calls itself recursively and eliminates common subexpressions by storing
      * them in temporary variables, emitted to the current builder scope. This is
      * a side-effect that leads to multiline code generation.
      */
      run(exp3) {
        if (this.#context.has(exp3)) return this.#context.get(exp3);
        const { op, src, dtype, arg } = exp3;
        let source = "";
        if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
          const a = this.run(src[0]);
          const b = this.run(src[1]);
          if (op === "Add") if (dtype === "bool") source = `(${a} || ${b})`;
          else source = `(${a} + ${b})`;
          else if (op === "Sub") source = `(${a} - ${b})`;
          else if (op === "Mul") if (dtype === "bool") source = `(${a} && ${b})`;
          else source = `(${a} * ${b})`;
          else if (op === "Idiv") source = isFloatDtype(dtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
          else if (op === "Mod") source = `(${a} % ${b})`;
          else if (op === "Min") if (dtype === "bool") source = `(${a} && ${b})`;
          else source = `min(${strip1(a)}, ${strip1(b)})`;
          else if (op === "Max") if (dtype === "bool") source = `(${a} || ${b})`;
          else source = `max(${strip1(a)}, ${strip1(b)})`;
          else if (op === "BitCombine") if (arg === "and") source = `(${a} & ${b})`;
          else if (arg === "or") source = `(${a} | ${b})`;
          else source = dtype === "bool" ? `(${a} != ${b})` : `(${a} ^ ${b})`;
          else if (op === "BitShift") if (arg === "shl") source = `(${a} << ${b})`;
          else source = `(${a} >> ${b})`;
          else if (op === "Cmplt") source = `(${a} < ${b})`;
          else if (op === "Cmpne") if (isFloatDtype(src[0].dtype)) {
            const x = this.#isGensym(a) ? a : this.#gensym();
            if (x !== a) this.wb.emit(`let ${x} = ${a};`);
            source = `(${x} != ${b} || min(${x}, ${dtypeToWgsl(src[0].dtype)}(inf())) != ${x})`;
          } else source = `(${a} != ${b})`;
        } else if (AluGroup.Unary.has(op)) if (op === "Reciprocal" && src[0].op === "Sqrt") source = `inverseSqrt(${this.run(src[0].src[0])})`;
        else {
          const a = this.run(src[0]);
          if (op === "Sin") source = `sin(${strip1(a)})`;
          else if (op === "Cos") source = `cos(${strip1(a)})`;
          else if (op === "Asin") source = `asin(${strip1(a)})`;
          else if (op === "Atan") source = `atan(${strip1(a)})`;
          else if (op === "Exp") source = `exp(${strip1(a)})`;
          else if (op === "Log") source = `log(${strip1(a)})`;
          else if (op === "Erf" || op === "Erfc") {
            const funcName = op === "Erf" ? "erf" : "erfc";
            if (dtype !== "float32") source = `${dtypeToWgsl(dtype)}(${funcName}(f32(${strip1(a)})))`;
            else source = `${funcName}(${strip1(a)})`;
          } else if (op === "Sqrt") source = `sqrt(${strip1(a)})`;
          else if (op === "Reciprocal") source = `(1.0 / ${a})`;
          else if (op === "Floor") source = `floor(${strip1(a)})`;
          else if (op === "Ceil") source = `ceil(${strip1(a)})`;
          else if (op === "Cast") {
            const srcTy = dtypeToWgsl(src[0].dtype);
            const dstTy = dtypeToWgsl(dtype);
            if (isFloatDtype(src[0].dtype) && !(isFloatDtype(dtype) || dtype === "bool")) {
              const maxVal = maxValueWgsl(dtype);
              const x = this.#isGensym(a) ? a : this.#gensym();
              if (x !== a) this.wb.emit(`let ${x}: ${srcTy} = ${strip1(a)};`);
              source = `select(${dstTy}(${x}), ${maxVal}, ${x} >= ${srcTy}(${maxVal}))`;
            } else source = `${dstTy}(${strip1(a)})`;
          } else if (op === "Bitcast") source = `bitcast<${dtypeToWgsl(dtype)}>(${strip1(a)})`;
        }
        else if (op === "Where") source = `select(${strip1(this.run(src[2]))}, ${strip1(this.run(src[1]))}, ${strip1(this.run(src[0]))})`;
        else if (op === "Threefry2x32") {
          const x = this.#gensym();
          const [k0, k1, c0, c1] = src.map((x2) => strip1(this.run(x2)));
          this.wb.emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
          if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
          else if (arg === 0) source = `${x}.x`;
          else if (arg === 1) source = `${x}.y`;
          else throw new UnsupportedOpError(op, dtype, "webgpu", arg);
        } else if (op === "Const") return constToWgsl(dtype, arg);
        else if (op === "Special") return arg[0];
        else if (op === "Variable") return arg;
        else if (op === "GlobalIndex") {
          source = `${this.args[arg[0]]}[${strip1(this.run(src[0]))}]`;
          if (dtype === "bool") source = `(${source} != 0)`;
        }
        if (!source) throw new UnsupportedOpError(op, dtype, "webgpu", arg);
        const typeName = dtypeToWgsl(dtype);
        if ((this.#references.get(exp3) ?? 0) > 1) {
          const name = this.#gensym();
          this.#context.set(exp3, name);
          this.wb.emit(`let ${name}: ${typeName} = ${strip1(source)};`);
          return name;
        } else {
          this.#context.set(exp3, source);
          return source;
        }
      }
    };
    gridOffsetY = 16384;
    NUDGE_INTERVAL_MS = 1;
    BUSY_MS = 20;
    BACKOFF_MS = 5;
    DEADLINE_MS = 1e3;
    isFirefox = typeof navigator !== "undefined" && navigator.userAgent.includes("Firefox") && !navigator.userAgent.includes("Seamonkey");
    resolvers = [];
    SyncReader = class SyncReader2 {
      device;
      static alphaModes = ["opaque", "premultiplied"];
      static width = 256;
      static height = 256;
      initialized = false;
      deviceStorage;
      deviceContexts;
      hostStorage;
      hostContext;
      constructor(device) {
        this.device = device;
      }
      #init() {
        if (typeof OffscreenCanvas === "undefined") throw new Error("OffscreenCanvas is not available in this environment, so you cannot read data from WebGPU synchronously. Consider using the async API.");
        const makeCanvas = () => new OffscreenCanvas(SyncReader2.width, SyncReader2.height);
        this.deviceStorage = SyncReader2.alphaModes.map(makeCanvas);
        this.deviceContexts = this.deviceStorage.map((canvas, i) => {
          const context = canvas.getContext("webgpu");
          context.configure({
            device: this.device,
            format: "bgra8unorm",
            usage: GPUTextureUsage.COPY_DST,
            alphaMode: SyncReader2.alphaModes[i]
          });
          return context;
        });
        this.hostStorage = makeCanvas();
        this.hostContext = this.hostStorage.getContext("2d", { willReadFrequently: true });
        this.initialized = true;
      }
      read(buffer, start, count) {
        if (!this.initialized) this.#init();
        const deviceStorage = this.deviceStorage;
        const deviceContexts = this.deviceContexts;
        const hostContext = this.hostContext;
        const pixelsSize = Math.ceil(count / 4);
        const bytesPerRow = SyncReader2.width * 4;
        const valsGPU = /* @__PURE__ */ new ArrayBuffer(pixelsSize * 4);
        for (let i = 0; i < deviceContexts.length; i++) {
          const texture = deviceContexts[i].getCurrentTexture();
          const readData = (width, height, offset2) => {
            const encoder = this.device.createCommandEncoder();
            encoder.copyBufferToTexture({
              buffer,
              bytesPerRow,
              offset: offset2 + start
            }, { texture }, {
              width,
              height,
              depthOrArrayLayers: 1
            });
            const commandBuffer = encoder.finish();
            this.device.queue.submit([commandBuffer]);
            hostContext.clearRect(0, 0, width, height);
            hostContext.drawImage(deviceStorage[i], 0, 0);
            const values = hostContext.getImageData(0, 0, width, height).data;
            const span = new Uint8ClampedArray(valsGPU, offset2, 4 * width * height);
            const alphaMode = SyncReader2.alphaModes[i];
            for (let k = 0; k < span.length; k += 4) if (alphaMode === "premultiplied") span[k + 3] = values[k + 3];
            else {
              span[k] = values[k + 2];
              span[k + 1] = values[k + 1];
              span[k + 2] = values[k];
            }
          };
          const pixelsPerCanvas = SyncReader2.width * SyncReader2.height;
          const wholeChunks = Math.floor(pixelsSize / pixelsPerCanvas);
          let remainder3 = pixelsSize % pixelsPerCanvas;
          const remainderRows = Math.floor(remainder3 / SyncReader2.width);
          remainder3 = remainder3 % SyncReader2.width;
          let offset = 0;
          for (let j = 0; j < wholeChunks; j++) {
            readData(SyncReader2.width, SyncReader2.height, offset);
            offset += pixelsPerCanvas * 4;
          }
          if (remainderRows > 0) {
            readData(SyncReader2.width, remainderRows, offset);
            offset += remainderRows * SyncReader2.width * 4;
          }
          if (remainder3 > 0) readData(remainder3, 1, offset);
        }
        return new Uint8Array(valsGPU, 0, count);
      }
    };
    CholeskyPhase = {
      Unblocked: 0,
      InitBlocked: 1,
      FactorBlock: 2,
      SolvePanel: 3,
      UpdateTrailing: 4
    };
    CHOLESKY_BLOCK_SIZE = 16;
    CHOLESKY_BLOCK_THRESHOLD = 256;
    CHOLESKY_UPDATE_TILE_SIZE = 16;
    MAX_TIMESTAMP_QUERIES = 4096;
    activeBatch = /* @__PURE__ */ new WeakMap();
    MAX_REUSABLE_BUFFER_BYTES = 64 * 1024 * 1024;
    MAX_REUSABLE_BUFFERS_PER_SIZE = 64;
    WebGPUBackend = class {
      device;
      type = "webgpu";
      maxArgs;
      pipelines;
      syncReader;
      buffers;
      nextSlot;
      #cachedShaderMap = /* @__PURE__ */ new Map();
      #reusableZsb;
      #bufferPool = /* @__PURE__ */ new Map();
      constructor(device) {
        this.device = device;
        if (DEBUG >= 3 && device.adapterInfo) console.info("webgpu adapter:", device.adapterInfo.vendor, device.adapterInfo.architecture);
        this.maxArgs = this.device.limits.maxStorageBuffersPerShaderStage - 1;
        this.pipelines = new ShaderPipelineCache(device);
        this.syncReader = new SyncReader(device);
        this.buffers = /* @__PURE__ */ new Map();
        this.nextSlot = 1;
        this.#reusableZsb = this.#createBuffer(4);
        device.addEventListener("uncapturederror", (event) => {
          console.error("Uncaptured error in WebGPU backend:", event.error.message);
        });
      }
      malloc(size2, initialData) {
        if (initialData && initialData.byteLength !== size2) throw new Error("initialData size does not match buffer size");
        const allocatedSize = Math.ceil(size2 / 4) * 4 || 4;
        const buffer = size2 === 0 ? this.#reusableZsb : this.#acquireBuffer(allocatedSize);
        if (initialData && size2 > 0) if (initialData.byteLength % 4 === 0) this.device.queue.writeBuffer(buffer, 0, initialData);
        else {
          const aligned = initialData.byteLength - initialData.byteLength % 4;
          if (aligned > 0) this.device.queue.writeBuffer(buffer, 0, initialData, 0, aligned);
          const remainder3 = /* @__PURE__ */ new Uint8Array(4);
          remainder3.set(initialData.subarray(aligned));
          this.device.queue.writeBuffer(buffer, aligned, remainder3);
        }
        const slot = this.nextSlot++;
        this.buffers.set(slot, {
          buffer,
          size: size2,
          allocatedSize,
          ref: 1
        });
        return slot;
      }
      incRef(slot) {
        const buffer = this.buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        buffer.ref++;
      }
      decRef(slot) {
        const buffer = this.buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        buffer.ref--;
        if (buffer.ref === 0) {
          this.buffers.delete(slot);
          if (buffer.buffer !== this.#reusableZsb) this.#releaseBuffer(buffer.buffer, buffer.allocatedSize);
        }
      }
      async read(slot, start, count) {
        const { buffer, size: size2 } = this.#getBuffer(slot);
        if (buffer === this.#reusableZsb) return /* @__PURE__ */ new Uint8Array();
        if (start === void 0) start = 0;
        if (count === void 0) count = size2 - start;
        const paddedSize = Math.ceil(count / 4) * 4;
        const staging = this.#createBuffer(paddedSize, { read: true });
        try {
          const commandEncoder = this.device.createCommandEncoder();
          commandEncoder.copyBufferToBuffer(buffer, start, staging, 0, paddedSize);
          this.device.queue.submit([commandEncoder.finish()]);
          await mapAsyncRead(this.device, staging);
          const arrayBuffer = staging.getMappedRange();
          return new Uint8Array(arrayBuffer.slice(), 0, count);
        } finally {
          staging.destroy();
        }
      }
      readSync(slot, start, count) {
        const { buffer, size: size2 } = this.#getBuffer(slot);
        if (buffer === this.#reusableZsb) return /* @__PURE__ */ new Uint8Array();
        if (start === void 0) start = 0;
        if (count === void 0) count = size2 - start;
        return this.syncReader.read(buffer, start, count);
      }
      #cachedShader(kernel) {
        const cacheKey = FpHash.hash(kernel);
        let result = this.#cachedShaderMap.get(cacheKey);
        if (!result) {
          result = pipelineSource(this.device, kernel);
          this.#cachedShaderMap.set(cacheKey, result);
        }
        return result;
      }
      async prepareKernel(kernel) {
        const shader = this.#cachedShader(kernel);
        const pipeline = await this.pipelines.prepare(shader);
        return new Executable(kernel, [{
          ...shader,
          pipeline
        }]);
      }
      prepareKernelSync(kernel) {
        const shader = this.#cachedShader(kernel);
        const pipeline = this.pipelines.prepareSync(shader);
        return new Executable(kernel, [{
          ...shader,
          pipeline
        }]);
      }
      async prepareRoutine(routine) {
        const shaders = createRoutineShader(this.device, routine);
        return new Executable(routine, await Promise.all(shaders.map(async (shader) => {
          const pipeline = await this.pipelines.prepare(shader);
          return {
            ...shader,
            pipeline
          };
        })));
      }
      prepareRoutineSync(routine) {
        return new Executable(routine, createRoutineShader(this.device, routine).map((shader) => {
          const pipeline = this.pipelines.prepareSync(shader);
          return {
            ...shader,
            pipeline
          };
        }));
      }
      dispatch(exe, inputs, outputs) {
        const inputBuffers = inputs.map((slot) => this.#getBuffer(slot).buffer);
        const outputBuffers = outputs.map((slot) => this.#getBuffer(slot).buffer);
        pipelineSubmit(this.device, exe, inputBuffers, outputBuffers);
      }
      #getBuffer(slot) {
        const buffer = this.buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        return {
          buffer: buffer.buffer,
          size: buffer.size
        };
      }
      #acquireBuffer(size2) {
        if (size2 > MAX_REUSABLE_BUFFER_BYTES) return this.#createBuffer(size2);
        const bucket = this.#bufferPool.get(size2);
        const buffer = bucket?.pop();
        if (bucket && bucket.length === 0) this.#bufferPool.delete(size2);
        return buffer ?? this.#createBuffer(size2);
      }
      #releaseBuffer(buffer, size2) {
        if (size2 > MAX_REUSABLE_BUFFER_BYTES) {
          buffer.destroy();
          return;
        }
        const bucket = this.#bufferPool.get(size2);
        if (!bucket) {
          this.#bufferPool.set(size2, [buffer]);
          return;
        }
        if (bucket.length >= MAX_REUSABLE_BUFFERS_PER_SIZE) {
          buffer.destroy();
          return;
        }
        bucket.push(buffer);
      }
      /**
      * Create a GPU buffer.
      *
      * By default, this creates a general-purpose buffer with the given size.
      *
      * - If `mapped` is true, initialize the buffer in mapped mode so that it can
      *   be populated with data from the CPU. (Call `.unmap()` later.)
      * - If `read` is true, create a staging buffer for returning data to CPU.
      *   (Call `.mapAsync()` later.)
      */
      #createBuffer(size2, { mapped = false, read = false } = {}) {
        if (read && mapped) throw new Error("mapped and read cannot both be true");
        return this.device.createBuffer({
          size: size2,
          usage: read ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          mappedAtCreation: mapped
        });
      }
    };
    ShaderPipelineCache = class {
      device;
      cache;
      inProgress;
      constructor(device) {
        this.device = device;
        this.cache = /* @__PURE__ */ new Map();
        this.inProgress = /* @__PURE__ */ new Map();
      }
      #getLayout(shader) {
        if (shader.numInputs + shader.numOutputs > this.device.limits.maxStorageBuffersPerShaderStage) {
          const actual = shader.numInputs + shader.numOutputs;
          const max2 = this.device.limits.maxStorageBuffersPerShaderStage;
          throw new Error(`Too many buffers (${actual}) for WebGPU pipeline (max: ${max2})`);
        }
        const bindGroupLayouts = [this.device.createBindGroupLayout({ entries: range(shader.numInputs + shader.numOutputs).map((i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: i < shader.numInputs ? "read-only-storage" : "storage" }
        })) })];
        if (shader.hasUniform) bindGroupLayouts.push(this.device.createBindGroupLayout({ entries: [{
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
            hasDynamicOffset: true
          }
        }] }));
        return this.device.createPipelineLayout({ bindGroupLayouts });
      }
      async prepare(shader) {
        const existingPipeline = this.cache.get(shader.code);
        if (existingPipeline) return existingPipeline;
        const existingPromise = this.inProgress.get(shader.code);
        if (existingPromise) return await existingPromise;
        if (DEBUG >= 2) console.info("=========== WebGPU shader ===========\n" + shader.code);
        const shaderModule = this.device.createShaderModule({ code: shader.code });
        const promise = (async () => {
          this.device.pushErrorScope("validation");
          try {
            const pipeline2 = await this.device.createComputePipelineAsync({
              layout: this.#getLayout(shader),
              compute: {
                module: shaderModule,
                entryPoint: "main"
              }
            });
            await this.device.popErrorScope();
            return pipeline2;
          } catch (_error) {
            const emsg = await compileError(shaderModule, await this.device.popErrorScope(), shader.code);
            throw new Error(emsg);
          }
        })();
        this.inProgress.set(shader.code, promise);
        const pipeline = await promise;
        this.cache.set(shader.code, pipeline);
        return pipeline;
      }
      prepareSync(shader) {
        const existingPipeline = this.cache.get(shader.code);
        if (existingPipeline) return existingPipeline;
        if (DEBUG >= 2) console.info("=========== WebGPU shader ===========\n" + shader.code);
        const shaderModule = this.device.createShaderModule({ code: shader.code });
        this.device.pushErrorScope("validation");
        const pipeline = this.device.createComputePipeline({
          layout: this.#getLayout(shader),
          compute: {
            module: shaderModule,
            entryPoint: "main"
          }
        });
        this.device.popErrorScope().then(async (scope) => {
          if (scope !== null) {
            const emsg = await compileError(shaderModule, scope, shader.code);
            console.error(emsg);
          }
        });
        this.cache.set(shader.code, pipeline);
        return pipeline;
      }
    };
  }
});

// node_modules/@jax-js/jax/dist/webgl-gbVkTlDB.js
var webgl_gbVkTlDB_exports = {};
__export(webgl_gbVkTlDB_exports, {
  WebGLBackend: () => WebGLBackend
});
function generateShader(kernel) {
  const tune = tuneNullopt(kernel);
  if (DEBUG >= 3) console.info(`webgl kernel.exp: ${kernel.exp}
tune.exp: ${tune.exp}`);
  const { nargs, reduction: re } = kernel;
  const outputDtype = kernel.dtype;
  const outputSize = computeTextureDimensions(Math.ceil(kernel.size / 4) || 1);
  const inputDtypes = Array(nargs).fill("float32");
  const builtins = {
    erf: false,
    threefry: false
  };
  const collectInfo = (exp3) => {
    if (exp3.op === "GlobalIndex") inputDtypes[exp3.arg[0]] = exp3.dtype;
    else if (exp3.op === "Erf" || exp3.op === "Erfc") builtins.erf = true;
    else if (exp3.op === "Threefry2x32") builtins.threefry = true;
  };
  tune.exp.fold(collectInfo);
  tune.epilogue?.fold(collectInfo);
  const shader = [];
  let indent = "";
  const pushIndent = Symbol("pushIndent");
  const popIndent = Symbol("popIndent");
  const emit = (...lines) => {
    for (const line of lines) if (line === pushIndent) indent += "  ";
    else if (line === popIndent) indent = indent.slice(0, -2);
    else shader.push(line ? indent + line : line);
  };
  emit("#version 300 es", "precision highp float;", "precision highp int;", "");
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);
  const resultType2 = glslType(outputDtype);
  for (let i = 0; i < nargs; i++) emit(`uniform highp sampler2D ${args[i]};`);
  emit("out vec4 out0;");
  const fetchFunctions = /* @__PURE__ */ new Set();
  for (const dtype of inputDtypes) fetchFunctions.add(dtype);
  for (const dtype of fetchFunctions) emit(generateLoadFunction(dtype));
  if (builtins.erf) emit(erfSrc2);
  if (builtins.threefry) emit(threefrySrc2);
  emit(`${resultType2} compute(int gidx) {`, pushIndent, `${resultType2} result = ${constToGlsl(outputDtype, 0)};`, `if (gidx < ${kernel.size}) {`, pushIndent);
  if (!re) emit(`result = ${strip1(generateExpression(tune.exp, args, inputDtypes))};`);
  else {
    emit(`${glslType(re.dtype)} acc = ${constToGlsl(re.dtype, re.identity)};`, `for (int ridx = 0; ridx < ${tune.size.reduce}; ridx++) {`, pushIndent);
    const code = generateExpression(tune.exp, args, inputDtypes);
    if (re.op === "Add") emit(`acc += ${strip1(code)};`);
    else if (re.op === "Mul") emit(`acc *= ${strip1(code)};`);
    else if (re.op === "Min") if (re.dtype !== "bool") emit(`acc = min(acc, ${strip1(code)});`);
    else emit(`acc = acc && ${code};`);
    else if (re.op === "Max") if (re.dtype !== "bool") emit(`acc = max(acc, ${strip1(code)});`);
    else emit(`acc = acc || ${code};`);
    else throw new Error(`Unsupported reduction op: ${re.op}`);
    emit(popIndent, "}");
    emit(`result = ${generateExpression(tune.epilogue, args, inputDtypes)};`);
  }
  emit(popIndent, "}", "return result;", popIndent, "}\n");
  emit("void main() {", pushIndent, "ivec2 fragCoord = ivec2(gl_FragCoord.xy);", `int texelIdx = fragCoord.y * ${outputSize.width} + fragCoord.x;`, `${resultType2} result0 = compute(texelIdx * 4);`, `${resultType2} result1 = compute(texelIdx * 4 + 1);`, `${resultType2} result2 = compute(texelIdx * 4 + 2);`, `${resultType2} result3 = compute(texelIdx * 4 + 3);`, `out0 = vec4(${range(4).map((i) => toRGBA32F(outputDtype, `result${i}`)).join(", ")});`);
  emit(popIndent, "}");
  return {
    code: shader.join("\n"),
    numInputs: nargs,
    outputSize: [outputSize.width, outputSize.height],
    outputDtype
  };
}
function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "Unknown shader compile error");
  return s;
}
function link(gl, vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "Unknown program link error");
  return p;
}
function compileShader(gl, shader) {
  if (DEBUG >= 1) console.info("=========== WebGL shader ===========\n" + shader.code);
  const program = link(gl, vertexShaderSource, shader.code);
  const inputLocations = [];
  for (let i = 0; i < shader.numInputs; i++) inputLocations.push(gl.getUniformLocation(program, `in${i}`));
  return {
    ...shader,
    program,
    inputLocations
  };
}
function computeTextureDimensions(numTexels) {
  const maxDim = 16384;
  let width = Math.min(Math.ceil(Math.sqrt(numTexels)), maxDim);
  width = Math.min(1 << Math.ceil(Math.log2(width)), maxDim);
  const height = Math.min(Math.ceil(numTexels / width), maxDim);
  return {
    width,
    height
  };
}
function glslType(dtype) {
  switch (dtype) {
    case "float32":
      return "float";
    case "int32":
      return "int";
    case "uint32":
      return "uint";
    case "bool":
      return "bool";
    default:
      throw new Error(`Unsupported dtype for WebGL: ${dtype}`);
  }
}
function generateLoadFunction(dtype) {
  const funcName = `load_${dtype}`;
  const returnType = glslType(dtype);
  let conversion;
  if (isFloatDtype(dtype)) conversion = "val";
  else if (dtype === "int32") conversion = "floatBitsToInt(val)";
  else if (dtype === "uint32") conversion = "floatBitsToUint(val)";
  else if (dtype === "bool") conversion = "floatBitsToInt(val) != 0";
  else throw new Error(`Unsupported dtype for WebGL fetch: ${dtype}`);
  return `
${returnType} ${funcName}(highp sampler2D tex, int idx) {
  ivec2 texSize = textureSize(tex, 0);
  int texel = idx / 4;
  int component = idx - texel * 4;
  ivec2 coord = ivec2(texel % texSize.x, texel / texSize.x);
  vec4 texVal = texelFetch(tex, coord, 0);
  float val;
  if (component == 0) val = texVal.x;
  else if (component == 1) val = texVal.y;
  else if (component == 2) val = texVal.z;
  else val = texVal.w;
  return ${conversion};
}
`;
}
function toRGBA32F(dtype, source) {
  switch (dtype) {
    case "float32":
      return source;
    case "int32":
      return `intBitsToFloat(${source})`;
    case "uint32":
      return `uintBitsToFloat(${source})`;
    case "bool":
      return `intBitsToFloat(${source} ? 1 : 0)`;
    default:
      throw new Error(`Unsupported dtype for WebGL output: ${dtype}`);
  }
}
function constToGlsl(dtype, value) {
  switch (dtype) {
    case "bool":
      return value ? "true" : "false";
    case "int32":
      return value.toString();
    case "uint32":
      return value.toString() + "u";
    case "float32":
      if (Number.isNaN(value)) return "uintBitsToFloat(0x7fc00000u)";
      if (!Number.isFinite(value)) return value > 0 ? "uintBitsToFloat(0x7f800000u)" : "uintBitsToFloat(0xff800000u)";
      return "float(" + value.toString() + ")";
    default:
      throw new Error(`Unsupported dtype for WebGL constant: ${dtype}`);
  }
}
function generateExpression(exp3, args, inputDtypes) {
  const expContext = /* @__PURE__ */ new Map();
  const gen = (e2) => {
    if (expContext.has(e2)) return expContext.get(e2);
    const { op, src, dtype, arg } = e2;
    let source = "";
    if (AluGroup.Binary.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === "Add") if (dtype === "bool") source = `(${a} || ${b})`;
      else source = `(${a} + ${b})`;
      else if (op === "Sub") source = `(${a} - ${b})`;
      else if (op === "Mul") if (dtype === "bool") source = `(${a} && ${b})`;
      else source = `(${a} * ${b})`;
      else if (op === "Idiv") if (isFloatDtype(dtype)) source = `trunc(${a} / ${b})`;
      else source = `(${a} / ${b})`;
      else if (op === "Mod") if (isFloatDtype(dtype)) source = `(${a} - ${b} * trunc(${a} / ${b}))`;
      else source = `(${a} % ${b})`;
      else if (op === "Min") if (dtype === "bool") source = `(${a} && ${b})`;
      else source = `min(${a}, ${b})`;
      else if (op === "Max") if (dtype === "bool") source = `(${a} || ${b})`;
      else source = `max(${a}, ${b})`;
      else if (op === "BitCombine") {
        let infix = arg === "and" ? "&" : arg === "or" ? "|" : "^";
        if (dtype === "bool") infix = infix + infix;
        source = `(${a} ${infix} ${b})`;
      } else if (op === "BitShift") if (arg === "shl") source = `(${a} << ${b})`;
      else source = `(${a} >> ${b})`;
    } else if (AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === "Cmplt") source = `(${a} < ${b})`;
      else if (op === "Cmpne") if (isFloatDtype(src[0].dtype)) source = `(${a} != ${b} || isnan(${a}) || isnan(${b}))`;
      else source = `(${a} != ${b})`;
    } else if (AluGroup.Unary.has(op)) {
      const a = gen(src[0]);
      if (op === "Sin") source = `sin(${strip1(a)})`;
      else if (op === "Cos") source = `cos(${strip1(a)})`;
      else if (op === "Asin") source = `asin(${strip1(a)})`;
      else if (op === "Atan") source = `atan(${strip1(a)})`;
      else if (op === "Exp") source = `exp(${strip1(a)})`;
      else if (op === "Log") source = `log(${strip1(a)})`;
      else if (op === "Erf") source = `erf(${strip1(a)})`;
      else if (op === "Erfc") source = `erfc(${strip1(a)})`;
      else if (op === "Sqrt") source = `sqrt(${strip1(a)})`;
      else if (op === "Floor") source = `floor(${strip1(a)})`;
      else if (op === "Ceil") source = `ceil(${strip1(a)})`;
      else if (op === "Reciprocal") source = `(1.0 / ${a})`;
      else if (op === "Cast") source = `${glslType(dtype)}(${strip1(a)})`;
      else if (op === "Bitcast") {
        const dtype0 = src[0].dtype;
        if (dtype === dtype0) source = a;
        else if (dtype === "float32") {
          if (dtype0 === "int32") source = `intBitsToFloat(${strip1(a)})`;
          else if (dtype0 === "uint32") source = `uintBitsToFloat(${strip1(a)})`;
        } else if (dtype === "int32") {
          if (dtype0 === "float32") source = `floatBitsToInt(${strip1(a)})`;
          else if (dtype0 === "uint32") source = `int(${strip1(a)})`;
        } else if (dtype === "uint32") {
          if (dtype0 === "float32") source = `floatBitsToUint(${strip1(a)})`;
          else if (dtype0 === "int32") source = `uint(${strip1(a)})`;
        }
      }
    } else if (op === "Threefry2x32") {
      const [k0, k1, c0, c1] = src.map((x) => strip1(gen(x)));
      const mode = arg;
      const call = `threefry2x32(uvec2(${k0}, ${k1}), uvec2(${c0}, ${c1}))`;
      if (mode === "xor") source = `(${call}.x ^ ${call}.y)`;
      else if (mode === 0) source = `${call}.x`;
      else if (mode === 1) source = `${call}.y`;
    } else if (op === "Where") {
      const [cond, t, f] = src.map(gen);
      source = `(${cond} ? ${t} : ${f})`;
    } else if (op === "Const") source = constToGlsl(dtype, arg);
    else if (op === "Special") source = arg[0];
    else if (op === "Variable") source = arg;
    else if (op === "GlobalIndex") {
      const gid = arg[0];
      const bufidx = gen(src[0]);
      source = `load_${inputDtypes[gid]}(${args[gid]}, ${strip1(bufidx)})`;
    }
    if (!source) throw new UnsupportedOpError(op, dtype, "webgl", arg);
    expContext.set(e2, source);
    return source;
  };
  return gen(exp3);
}
var threefrySrc2, erfSrc2, WebGLBackend, vertexShaderSource;
var init_webgl_gbVkTlDB = __esm({
  "node_modules/@jax-js/jax/dist/webgl-gbVkTlDB.js"() {
    init_backend_D_Uwkp6d();
    threefrySrc2 = `
uvec2 threefry2x32(uvec2 key, uvec2 ctr) {
  uint ks0 = key.x;
  uint ks1 = key.y;
  uint ks2 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  uint x0 = ctr.x + ks0;
  uint x1 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return uvec2(x0, x1);
}`;
    erfSrc2 = `
const float _erf_p = 0.3275911;
const float _erf_a1 = 0.254829592;
const float _erf_a2 = -0.284496736;
const float _erf_a3 = 1.421413741;
const float _erf_a4 = -1.453152027;
const float _erf_a5 = 1.061405429;
float erf(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
float erfc(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  float E = P_t * exp(-x * x);
  return x >= 0.0 ? E : 2.0 - E;
}`;
    WebGLBackend = class {
      type = "webgl";
      maxArgs = 8;
      gl;
      #fbo;
      #buffers;
      #programCache;
      #nextSlot;
      constructor(gl) {
        this.gl = gl;
        this.#fbo = gl.createFramebuffer();
        this.#buffers = /* @__PURE__ */ new Map();
        this.#programCache = /* @__PURE__ */ new Map();
        this.#nextSlot = 1;
      }
      /**
      * Allocate a buffer with a specific dtype.
      *
      * All buffers use RGBA32F texture format internally. Data is stored as raw
      * bits and reinterpreted using floatBitsToInt/intBitsToFloat in shaders.
      * This mirrors how WebGPU handles untyped byte buffers.
      */
      malloc(size2, initialData) {
        const gl = this.gl;
        const numFloats = Math.ceil(size2 / 4) || 1;
        const { width, height } = computeTextureDimensions(Math.ceil(numFloats / 4) || 1);
        const texture = gl.createTexture();
        if (!texture) throw new Error("Failed to create texture");
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        const totalFloats = width * height * 4;
        let pixels = null;
        if (initialData) {
          pixels = new Float32Array(totalFloats);
          new Uint8Array(pixels.buffer).set(initialData);
        }
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, pixels);
        gl.bindTexture(gl.TEXTURE_2D, null);
        const slot = this.#nextSlot++;
        this.#buffers.set(slot, {
          ref: 1,
          size: size2,
          texture,
          width,
          height
        });
        return slot;
      }
      incRef(slot) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        buffer.ref++;
      }
      decRef(slot) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        buffer.ref--;
        if (buffer.ref === 0) {
          this.gl.deleteTexture(buffer.texture);
          this.#buffers.delete(slot);
        }
      }
      async read(slot, start, count) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        const gl = this.gl;
        if (start === void 0) start = 0;
        if (count === void 0) count = buffer.size - start;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, buffer.texture, 0);
        const totalBytes = buffer.width * buffer.height * 4 * 4;
        const floatData = new Float32Array(totalBytes / 4);
        const pbo = gl.createBuffer();
        if (!pbo) throw new Error("Failed to create PBO");
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
        gl.bufferData(gl.PIXEL_PACK_BUFFER, totalBytes, gl.STREAM_READ);
        gl.readPixels(0, 0, buffer.width, buffer.height, gl.RGBA, gl.FLOAT, 0);
        const readError = gl.getError();
        if (readError !== gl.NO_ERROR) {
          gl.deleteBuffer(pbo);
          throw new Error(`WebGL error after readPixels: ${readError}`);
        }
        const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        if (!sync) throw new Error("Failed to create sync object");
        gl.flush();
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        await new Promise((resolve, reject) => {
          const poll = () => {
            const status2 = gl.clientWaitSync(sync, 0, 0);
            if (status2 === gl.TIMEOUT_EXPIRED) {
              setTimeout(poll, 5);
              return;
            }
            if (status2 === gl.WAIT_FAILED) {
              gl.deleteSync(sync);
              gl.deleteBuffer(pbo);
              reject(/* @__PURE__ */ new Error("clientWaitSync failed"));
              return;
            }
            resolve();
          };
          poll();
        });
        gl.deleteSync(sync);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, floatData);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        gl.deleteBuffer(pbo);
        const byteData = new Uint8Array(floatData.buffer);
        return new Uint8Array(byteData.slice(start, start + count));
      }
      readSync(slot, start, count) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        const gl = this.gl;
        if (start === void 0) start = 0;
        if (count === void 0) count = buffer.size - start;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, buffer.texture, 0);
        const totalFloats = buffer.width * buffer.height * 4;
        const floatData = new Float32Array(totalFloats);
        gl.readPixels(0, 0, buffer.width, buffer.height, gl.RGBA, gl.FLOAT, floatData);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        const byteData = new Uint8Array(floatData.buffer);
        return new Uint8Array(byteData.slice(start, start + count));
      }
      async prepareKernel(kernel) {
        return this.prepareKernelSync(kernel);
      }
      prepareKernelSync(kernel) {
        const shader = generateShader(kernel);
        const cached = this.#programCache.get(shader.code);
        if (cached) return new Executable(kernel, cached);
        const dispatch = compileShader(this.gl, shader);
        this.#programCache.set(shader.code, dispatch);
        return new Executable(kernel, dispatch);
      }
      prepareRoutine(routine) {
        throw new UnsupportedRoutineError(routine.name, "webgl");
      }
      prepareRoutineSync(routine) {
        throw new UnsupportedRoutineError(routine.name, "webgl");
      }
      dispatch(exe, inputs, outputs) {
        const gl = this.gl;
        if (gl.isContextLost()) throw new Error("WebGL context lost - cannot dispatch");
        const { program, inputLocations } = exe.data;
        if (inputs.length !== exe.data.numInputs) throw new Error(`Expected ${exe.data.numInputs} inputs, got ${inputs.length}`);
        if (outputs.length !== 1) throw new Error(`Expected 1 output, got ${outputs.length}`);
        const outputBuffer = this.#buffers.get(outputs[0]);
        if (!outputBuffer) throw new SlotError(outputs[0]);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputBuffer.texture, 0);
        const status2 = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status2 !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`Framebuffer incomplete: ${status2}`);
        gl.viewport(0, 0, outputBuffer.width, outputBuffer.height);
        gl.useProgram(program);
        for (let i = 0; i < inputs.length; i++) {
          const inputBuffer = this.#buffers.get(inputs[i]);
          if (!inputBuffer) throw new SlotError(inputs[i]);
          gl.activeTexture(gl.TEXTURE0 + i);
          gl.bindTexture(gl.TEXTURE_2D, inputBuffer.texture);
          if (inputLocations[i] !== null) gl.uniform1i(inputLocations[i], i);
        }
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        const error = gl.getError();
        if (error !== gl.NO_ERROR) {
          let errorName;
          if (error === gl.INVALID_ENUM) errorName = "INVALID_ENUM";
          else if (error === gl.INVALID_VALUE) errorName = "INVALID_VALUE";
          else if (error === gl.INVALID_OPERATION) errorName = "INVALID_OPERATION";
          else if (error === gl.INVALID_FRAMEBUFFER_OPERATION) errorName = "INVALID_FRAMEBUFFER_OPERATION";
          else if (error === gl.OUT_OF_MEMORY) errorName = "OUT_OF_MEMORY";
          else if (error === gl.CONTEXT_LOST_WEBGL) errorName = "CONTEXT_LOST_WEBGL";
          else errorName = `UNKNOWN(${error})`;
          throw new Error(`WebGL error after drawArrays: ${errorName}`);
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.useProgram(null);
      }
    };
    vertexShaderSource = `#version 300 es
precision highp float;
const vec2 pos[3] = vec2[](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() { gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0); }
`;
  }
});

// node_modules/@jax-js/jax/dist/backend-D_Uwkp6d.js
function unzip2(pairs) {
  const lst1 = [];
  const lst2 = [];
  for (const [x, y] of pairs) {
    lst1.push(x);
    lst2.push(y);
  }
  return [lst1, lst2];
}
function zip(xs, ys) {
  return xs.map((x, i) => [x, ys[i]]);
}
function zipn(...arrays) {
  const minLength = Math.min(...arrays.map((x) => x.length));
  return Array.from({ length: minLength }, (_, i) => arrays.map((arr) => arr[i]));
}
function sorted(arr) {
  return [...arr].sort((a, b) => a - b);
}
function rep(length, value) {
  if (value instanceof Function) return new Array(length).fill(0).map((_, i) => value(i));
  return new Array(length).fill(value);
}
function prod(arr) {
  return arr.reduce((acc, x) => acc * x, 1);
}
function gcd(...values) {
  let a = 0;
  for (let b of values) while (b !== 0) [a, b] = [b, a % b];
  return Math.abs(a);
}
function intdiv(a, b) {
  return Math.floor(a / b);
}
function clamp(x, min2, max2) {
  return Math.max(min2, Math.min(max2, x));
}
function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object") return false;
  if (a === null || b === null) return false;
  if (Object.keys(a).length !== Object.keys(b).length) return false;
  for (const key of Object.keys(a)) if (!deepEqual(a[key], b[key])) return false;
  return true;
}
function mapSetUnion(a, b) {
  if (!b) return a;
  for (const [key, setB] of b.entries()) {
    const setA = a.get(key);
    if (setA) for (const val of setB) setA.add(val);
    else a.set(key, setB);
  }
  return a;
}
function lexCompare(a, b) {
  const minLength = Math.min(a.length, b.length);
  for (let i = 0; i < minLength; i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return a.length - b.length;
}
function isNumberPair(x) {
  return Array.isArray(x) && x.length === 2 && typeof x[0] === "number" && typeof x[1] === "number";
}
function checkAxis(axis, ndim2) {
  if (axis < -ndim2 || axis >= ndim2) throw new Error(`Axis ${axis} out of bounds for array of dimension ${ndim2}`);
  return axis < 0 ? axis + ndim2 : axis;
}
function normalizeAxis(axis, ndim2, sort2 = true) {
  if (axis === null) return range(ndim2);
  else if (typeof axis === "number") return [checkAxis(axis, ndim2)];
  else {
    const seen = /* @__PURE__ */ new Set();
    for (const a of axis) {
      const ca = checkAxis(a, ndim2);
      if (seen.has(ca)) throw new Error(`Duplicate axis ${ca} passed to function`);
      seen.add(ca);
    }
    return sort2 ? sorted(seen) : [...seen];
  }
}
function checkSquare(name, shape2) {
  if (shape2.length < 2 || shape2[shape2.length - 1] !== shape2[shape2.length - 2]) throw new Error(`${name}: input must be at least 2D square matrix, got shape ${JSON.stringify(shape2)}`);
  return shape2[shape2.length - 1];
}
function range(start, stop, step = 1) {
  if (stop === void 0) {
    stop = start;
    start = 0;
  }
  const result = [];
  for (let i = start; i < stop; i += step) result.push(i);
  return result;
}
function isPermutation(axis, n) {
  if (axis.length !== n) return false;
  const seen = /* @__PURE__ */ new Set();
  for (const x of axis) {
    if (x < 0 || x >= n) return false;
    seen.add(x);
  }
  return seen.size === n;
}
function invertPermutation(axis) {
  const n = axis.length;
  if (!isPermutation(axis, n)) throw new Error("invertPermutation: axis is not a permutation");
  const result = new Array(n);
  for (let i = 0; i < n; i++) result[axis[i]] = i;
  return result;
}
function findPow2(hint, max2) {
  if (max2 < 1) throw new Error("max must be a positive integer");
  let ret = 1;
  while ((hint === 0 || ret < hint) && 2 * ret <= max2) ret *= 2;
  return ret;
}
function generalBroadcast(a, b) {
  const out = [];
  let i = a.length - 1;
  let j = b.length - 1;
  for (; i >= 0 && j >= 0; i--, j--) {
    const x = a[i];
    const y = b[j];
    if (x === y) out.push(x);
    else if (x === 1) out.push(y);
    else if (y === 1) out.push(x);
    else throw new TypeError(`Incompatible array broadcast shapes: ${a} vs ${b}`);
  }
  for (; i >= 0; i--) out.push(a[i]);
  for (; j >= 0; j--) out.push(b[j]);
  return out.reverse();
}
function recursiveFlatten(ar) {
  if (!Array.isArray(ar)) return [ar];
  return ar.flat(Infinity);
}
function strip1(str) {
  if (str[0] === "(" && str[str.length - 1] === ")") return str.slice(1, -1);
  return str;
}
function runWithCache(cache, key, thunk) {
  const keyStr = JSON.stringify(key);
  if (cache.has(keyStr)) return cache.get(keyStr);
  else {
    const value = thunk();
    cache.set(keyStr, value);
    return value;
  }
}
async function runWithCacheAsync(cache, key, thunk) {
  const keyStr = JSON.stringify(key);
  if (cache.has(keyStr)) return cache.get(keyStr);
  else {
    const value = await thunk();
    cache.set(keyStr, value);
    return value;
  }
}
function promoteTypes(dtype1, dtype2) {
  if (dtype1 === dtype2) return dtype1;
  const rank = {
    ["bool"]: 0,
    ["uint32"]: 1,
    ["int32"]: 2,
    ["float16"]: 3,
    ["float32"]: 4,
    ["float64"]: 5
  };
  return rank[dtype1] > rank[dtype2] ? dtype1 : dtype2;
}
function dtypedArray(dtype, data) {
  const { buffer, byteLength, byteOffset } = data;
  const length = byteLength / byteWidth(dtype);
  switch (dtype) {
    case "float32":
      return new Float32Array(buffer, byteOffset, length);
    case "int32":
    case "bool":
      return new Int32Array(buffer, byteOffset, length);
    case "uint32":
      return new Uint32Array(buffer, byteOffset, length);
    case "float16":
      return new Float16Array(buffer, byteOffset, length);
    case "float64":
      return new Float64Array(buffer, byteOffset, length);
    default:
      throw new Error(`Unimplemented dtype: ${dtype}`);
  }
}
function dtypedJsArray(dtype, data) {
  switch (dtype) {
    case "float32":
      return new Float32Array(data);
    case "int32":
    case "bool":
      return new Int32Array(data);
    case "uint32":
      return new Uint32Array(data);
    case "float16":
      return new Float16Array(data);
    case "float64":
      return new Float64Array(data);
    default:
      throw new Error(`Unimplemented dtype: ${dtype}`);
  }
}
function accessorGlobal(dtype, gid, st, indices2) {
  const [index, valid] = st.toAluExp(indices2);
  const [, len] = st.views[0].dataRange();
  if (valid.resolve()) return AluExp.globalIndex(dtype, gid, len, index);
  return AluExp.where(valid, AluExp.globalIndex(dtype, gid, len, index), AluExp.const(dtype, 0));
}
function accessorAluExp(exp3, st, indices2) {
  const [index, valid] = st.toAluExp(indices2);
  if (valid.resolve()) return exp3.substitute({ idx: index });
  return AluExp.where(valid, exp3.substitute({ idx: index }), AluExp.const(exp3.dtype, 0));
}
function threefry2x32(k0, k1, c0, c1) {
  const rotl32 = (x, r) => (x << r | x >>> 32 - r) >>> 0;
  const ks0 = k0 >>> 0;
  const ks1 = k1 >>> 0;
  const ks2 = (ks0 ^ ks1 ^ 466688986) >>> 0;
  let x0 = c0 + ks0 >>> 0;
  let x1 = c1 + ks1 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
  x0 = x0 + ks1 >>> 0;
  x1 = x1 + ks2 + 1 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 17) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 29) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 16) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 24) ^ x0;
  x0 = x0 + ks2 >>> 0;
  x1 = x1 + ks0 + 2 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
  x0 = x0 + ks0 >>> 0;
  x1 = x1 + ks1 + 3 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 17) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 29) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 16) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 24) ^ x0;
  x0 = x0 + ks1 >>> 0;
  x1 = x1 + ks2 + 4 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
  x0 = x0 + ks2 >>> 0;
  x1 = x1 + ks0 + 5 >>> 0;
  return [x0, x1];
}
function _erfapprox$1(x) {
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const t = 1 / (1 + p * x);
  return ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
}
function erf(x) {
  if (x >= 0) return 1 - _erfapprox$1(x);
  else return _erfapprox$1(-x) - 1;
}
function erfc(x) {
  if (x >= 0) return _erfapprox$1(x);
  else return 2 - _erfapprox$1(-x);
}
function runCpuRoutine(routine, inputs, outputs) {
  const { name, type } = routine;
  const inputAr = inputs.map((buf, i) => dtypedArray(type.inputDtypes[i], buf));
  const outputAr = outputs.map((buf, i) => dtypedArray(type.outputDtypes[i], buf));
  switch (name) {
    case "Sort":
      return runSort(type, inputAr, outputAr);
    case "Argsort":
      return runArgsort(type, inputAr, outputAr);
    case "Scatter":
      return runScatter(type, inputAr, outputAr, routine.params);
    case "TriangularSolve":
      return runTriangularSolve(type, inputAr, outputAr, routine.params);
    case "Cholesky":
      return runCholesky(type, inputAr, outputAr);
    case "LU":
      return runLU(type, inputAr, outputAr);
    case "JacobiEigh":
      return runJacobiEigh(type, inputAr, outputAr, routine.params);
    case "Fft":
      return runFft(type, inputAr, outputAr, routine.params);
    default:
  }
}
function cpuRoutineJSForWorkers() {
  return `
const DType = ${JSON.stringify(DType)};
const Routines = ${JSON.stringify(Routines)};
const ${byteWidth.name} = ${byteWidth.toString()};
const ${isFloatDtype.name} = ${isFloatDtype.toString()};
const ${dtypedArray.name} = ${dtypedArray.toString()};
const ScatterOp = ${JSON.stringify(ScatterOp)};
${runCpuRoutine.toString()}
${runSort.toString()}
${runArgsort.toString()}
${shapeStrides.toString()}
${runScatter.toString()}
${runTriangularSolve.toString()}
${runCholesky.toString()}
${runLU.toString()}
${runJacobiEigh.toString()}
${runFft.toString()}
const __minify_safe_runCpuRoutine = ${runCpuRoutine.name};
`;
}
function runSort(type, [x], [y]) {
  const xs = type.inputShapes[0];
  if (xs.length === 0) throw new Error("sort: cannot sort a scalar");
  const n = xs[xs.length - 1];
  y.set(x);
  for (let i = 0; i < y.length; i += n) y.subarray(i, i + n).sort();
}
function runArgsort(type, [x], [y, yi]) {
  const xs = type.inputShapes[0];
  if (xs.length === 0) throw new Error("argsort: cannot sort a scalar");
  const n = xs[xs.length - 1];
  for (let offset = 0; offset < y.length; offset += n) {
    const ar = x.subarray(offset, offset + n);
    const out = y.subarray(offset, offset + n);
    const outi = yi.subarray(offset, offset + n);
    for (let i = 0; i < n; i++) outi[i] = i;
    outi.sort((a, b) => {
      const x2 = ar[a];
      const y2 = ar[b];
      if (isNaN(x2)) return isNaN(y2) ? 0 : 1;
      if (isNaN(y2)) return -1;
      return x2 === y2 ? 0 : x2 < y2 ? -1 : 1;
    });
    for (let i = 0; i < n; i++) out[i] = ar[outi[i]];
  }
}
function shapeStrides(shape2) {
  const strides = new Array(shape2.length);
  let stride = 1;
  for (let i = shape2.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape2[i];
  }
  return strides;
}
function runScatter(type, [updates, ...indices2], [output], { op, shape: outputShape, axis: indexedAxes, outDim, uniqueIndices }) {
  output.fill(0);
  const accumulate = op !== "update" && !uniqueIndices;
  const updateShape = type.inputShapes[0];
  const indexShapes = type.inputShapes.slice(1);
  const indexRank = Math.max(...indexShapes.map((s) => s.length));
  const updateStrides = shapeStrides(updateShape);
  const outputStrides = shapeStrides(outputShape);
  const indexStrides = indexShapes.map(shapeStrides);
  const outputAxisToIndex = outputShape.map((_, d) => indexedAxes.indexOf(d));
  const freeUpdateDims = updateShape.map((_, i) => i).filter((i) => i < outDim || i >= outDim + indexRank);
  for (let i = 0; i < updates.length; i++) {
    let outputIndex = 0;
    let valid = true;
    let freeDim = 0;
    for (let outputAxis = 0; outputAxis < outputShape.length; outputAxis++) {
      const indexInput = outputAxisToIndex[outputAxis];
      let coord;
      if (indexInput === -1) {
        const updateDim = freeUpdateDims[freeDim++];
        coord = Math.floor(i / updateStrides[updateDim]) % updateShape[updateDim];
      } else {
        const indexShape = indexShapes[indexInput];
        const firstUpdateDim = outDim + indexRank - indexShape.length;
        let indexOffset = 0;
        for (let j = 0; j < indexShape.length; j++) {
          if (indexShape[j] === 1) continue;
          const updateDim = firstUpdateDim + j;
          const updateCoord = Math.floor(i / updateStrides[updateDim]) % updateShape[updateDim];
          indexOffset += updateCoord * indexStrides[indexInput][j];
        }
        coord = indices2[indexInput][indexOffset];
      }
      if (coord < 0 || coord >= outputShape[outputAxis]) {
        valid = false;
        break;
      }
      outputIndex += coord * outputStrides[outputAxis];
    }
    if (!valid) continue;
    if (accumulate && type.inputDtypes[0] === "bool") output[outputIndex] |= updates[i];
    else if (accumulate) output[outputIndex] += updates[i];
    else output[outputIndex] = updates[i];
  }
}
function runTriangularSolve(type, [a, b], [x], { unitDiagonal }) {
  const as = type.inputShapes[0];
  const bs = type.inputShapes[1];
  if (as.length < 2) throw new Error(`triangular_solve: a must be at least 2D, got ${as}`);
  if (bs.length < 2) throw new Error(`triangular_solve: b must be at least 2D, got ${bs}`);
  const n = as[as.length - 2];
  if (n !== as[as.length - 1] || n !== bs[bs.length - 1]) throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
  const batch = bs[bs.length - 2];
  for (let counter = 0; counter < a.length / (n * n); counter++) {
    const a1 = a.subarray(counter * n * n, (counter + 1) * n * n);
    for (let t = 0; t < batch; t++) {
      const b1 = b.subarray((counter * batch + t) * n, (counter * batch + t + 1) * n);
      const x1 = x.subarray((counter * batch + t) * n, (counter * batch + t + 1) * n);
      for (let i = n - 1; i >= 0; i--) {
        let sum2 = b1[i];
        for (let j = i + 1; j < n; j++) sum2 -= a1[i * n + j] * x1[j];
        x1[i] = unitDiagonal ? sum2 : sum2 / a1[i * n + i];
      }
    }
  }
}
function runCholesky(type, [x], [y]) {
  const xs = type.inputShapes[0];
  if (xs.length < 2) throw new Error("cholesky: input must be at least 2D");
  const n = xs[xs.length - 2];
  const m = xs[xs.length - 1];
  if (n !== m) throw new Error(`cholesky: input must be square, got [${n}, ${m}]`);
  y.fill(0);
  for (let offset = 0; offset < y.length; offset += n * n) {
    const ar = x.subarray(offset, offset + n * n);
    const out = y.subarray(offset, offset + n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let sum2 = ar[i * n + j];
      for (let k = 0; k < j; k++) sum2 -= out[i * n + k] * out[j * n + k];
      out[i * n + j] = i === j ? Math.sqrt(sum2) : sum2 / out[j * n + j];
    }
  }
}
function runLU(type, [a], [lu2, pivots, perm]) {
  const shape2 = type.inputShapes[0];
  if (shape2.length < 2) throw new Error("lu: input must be at least 2D");
  const m = shape2[shape2.length - 2];
  const n = shape2[shape2.length - 1];
  const r = Math.min(m, n);
  for (let offset = 0; offset < a.length; offset += m * n) {
    const ar = a.subarray(offset, offset + m * n);
    const out = lu2.subarray(offset, offset + m * n);
    const batchIdx = offset / (m * n);
    const piv = pivots.subarray(batchIdx * r, (batchIdx + 1) * r);
    const p = perm.subarray(batchIdx * m, (batchIdx + 1) * m);
    out.set(ar);
    for (let i = 0; i < m; i++) p[i] = i;
    for (let j = 0; j < r; j++) {
      let maxVal = Math.abs(out[j * n + j]);
      let maxRow = j;
      for (let i = j + 1; i < m; i++) {
        const val = Math.abs(out[i * n + j]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = i;
        }
      }
      piv[j] = maxRow;
      if (maxRow !== j) {
        for (let col = 0; col < n; col++) {
          const tmp = out[j * n + col];
          out[j * n + col] = out[maxRow * n + col];
          out[maxRow * n + col] = tmp;
        }
        const tmpP = p[j];
        p[j] = p[maxRow];
        p[maxRow] = tmpP;
      }
      const diag2 = out[j * n + j];
      if (diag2 !== 0) for (let i = j + 1; i < m; i++) {
        const factor = out[i * n + j] / diag2;
        out[i * n + j] = factor;
        for (let col = j + 1; col < n; col++) out[i * n + col] -= factor * out[j * n + col];
      }
    }
  }
}
function runJacobiEigh(type, [input], [diagonalized, vectors], { maxSweeps, tolerance }) {
  const shape2 = type.inputShapes[0];
  if (shape2.length < 2) throw new Error("jacobi_eigh: input must be at least 2D");
  const n = shape2[shape2.length - 1];
  if (n !== shape2[shape2.length - 2]) throw new Error(`jacobi_eigh: input must be square, got ${shape2}`);
  if (!isFloatDtype(type.inputDtypes[0])) throw new TypeError(`jacobi_eigh: input must be floating-point`);
  function symIndex(i, j) {
    return i >= j ? i * n + j : j * n + i;
  }
  function maxAbsMatrix(a) {
    let maxAbs = 1;
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) maxAbs = Math.max(maxAbs, Math.abs(a[i * n + j]));
    return maxAbs;
  }
  function maxAbsOffDiagonal(a) {
    let maxAbs = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) maxAbs = Math.max(maxAbs, Math.abs(a[i * n + j]));
    return maxAbs;
  }
  function applyJacobiRotation(a, v, p, q) {
    const pp = p * n + p;
    const qq = q * n + q;
    const pq = symIndex(p, q);
    const app = a[pp];
    const aqq = a[qq];
    const apq = a[pq];
    if (apq === 0) return;
    const tau = (aqq - app) / (2 * apq);
    const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(tau * tau + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const kp = symIndex(k, p);
      const kq = symIndex(k, q);
      const akp = a[kp];
      const akq = a[kq];
      const nextKp = c * akp - s * akq;
      const nextKq = s * akp + c * akq;
      a[kp] = nextKp;
      a[kq] = nextKq;
    }
    a[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[pq] = 0;
    for (let k = 0; k < n; k++) {
      const kp = k * n + p;
      const kq = k * n + q;
      const vkp = v[kp];
      const vkq = v[kq];
      v[kp] = c * vkp - s * vkq;
      v[kq] = s * vkp + c * vkq;
    }
  }
  const matrixSize = n * n;
  for (let offset = 0; offset < input.length; offset += matrixSize) {
    const x = input.subarray(offset, offset + matrixSize);
    const a = diagonalized.subarray(offset, offset + matrixSize);
    const v = vectors.subarray(offset, offset + matrixSize);
    a.fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) a[i * n + j] = x[i * n + j];
    v.fill(0);
    for (let i = 0; i < n; i++) v[i * n + i] = 1;
    const threshold = tolerance * maxAbsMatrix(a);
    let maxOffDiagonal = maxAbsOffDiagonal(a);
    for (let sweep = 0; sweep < maxSweeps && maxOffDiagonal > threshold; sweep++) {
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) applyJacobiRotation(a, v, p, q);
      maxOffDiagonal = maxAbsOffDiagonal(a);
    }
  }
}
function runFft(type, [real, imag], [outReal, outImag], { factors, inverse }) {
  const shape2 = type.inputShapes[0];
  if (shape2.length < 1) throw new Error("fft: input must be at least 1D");
  const n = shape2[shape2.length - 1];
  if (n < 1) throw new Error(`fft: final axis must be non-empty, got ${n}`);
  if (type.inputDtypes[0] !== type.inputDtypes[1] || !isFloatDtype(type.inputDtypes[0])) throw new Error("fft: expected matching floating-point real/imag arrays");
  const angleScale = (inverse ? 2 : -2) * Math.PI;
  const permutation = new Uint32Array(n);
  for (let index = 0; index < n; index++) {
    let remaining = index;
    let stride = 1;
    let reversed = 0;
    for (const factor of factors) {
      const digit = remaining % factor;
      remaining = Math.floor(remaining / factor);
      stride *= factor;
      reversed += digit * (n / stride);
    }
    permutation[index] = reversed;
  }
  const scratchReal = new Array(Math.max(1, ...factors));
  const scratchImag = new Array(scratchReal.length);
  for (let offset = 0; offset < real.length; offset += n) {
    for (let index = 0; index < n; index++) {
      const source = offset + permutation[index];
      outReal[offset + index] = real[source];
      outImag[offset + index] = imag[source];
    }
    let prev = 1;
    for (let stage = 0; stage < factors.length; stage++) {
      const radix = factors[stage];
      const span = prev * radix;
      const stageScale = inverse && stage === factors.length - 1 ? 1 / n : 1;
      for (let group = 0; group < n; group += span) for (let j = 0; j < prev; j++) {
        for (let q = 0; q < radix; q++) {
          const idx = offset + group + j + q * prev;
          const angle = angleScale * q * j / span;
          const c = Math.cos(angle);
          const s = Math.sin(angle);
          const xr = outReal[idx];
          const xi = outImag[idx];
          scratchReal[q] = xr * c - xi * s;
          scratchImag[q] = xr * s + xi * c;
        }
        for (let p = 0; p < radix; p++) {
          let sumReal = 0;
          let sumImag = 0;
          for (let q = 0; q < radix; q++) {
            const angle = angleScale * q * p / radix;
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            const xr = scratchReal[q];
            const xi = scratchImag[q];
            sumReal += xr * c - xi * s;
            sumImag += xr * s + xi * c;
          }
          const idx = offset + group + j + p * prev;
          outReal[idx] = sumReal * stageScale;
          outImag[idx] = sumImag * stageScale;
        }
      }
      prev = span;
    }
  }
}
function canonicalizeStrides(shape2, strides) {
  const newStrides = [];
  for (let i = 0; i < shape2.length; i++) if (shape2[i] === 1) newStrides.push(0);
  else newStrides.push(strides[i]);
  return newStrides;
}
function defaultStrides(shape2) {
  if (shape2.length === 0) return [];
  const strides = rep(shape2.length, 1);
  for (let i = shape2.length - 1; i > 0; i--) strides[i - 1] = shape2[i] * strides[i];
  return canonicalizeStrides(shape2, strides);
}
function mergeDims(shape2, strides, mask) {
  if (shape2.length === 0) return [];
  if (shape2.length !== strides.length || mask && shape2.length !== mask.length) throw new Error("internal: invalid args to mergeDims");
  const ret = [[
    shape2[0],
    strides[0],
    strides[0] !== 0 ? shape2[0] : 0
  ]];
  let merging = mask ? mask[0][1] - mask[0][0] === 1 : shape2[0] === 1;
  for (let i = 1; i < shape2.length; i++) {
    const [s, st] = [shape2[i], strides[i]];
    if (s === 1) continue;
    const [lastS, lastSt, lastPreExpandS] = ret[ret.length - 1];
    if (merging || lastSt === s * st) ret[ret.length - 1] = [
      lastS * s,
      st,
      merging ? s : lastPreExpandS * s
    ];
    else ret.push([
      s,
      st,
      s
    ]);
    merging = mask ? mask[i][1] - mask[i][0] === 1 : false;
  }
  return ret;
}
function reshapeMask(maskInput, oldShape, newShape) {
  const newMask = [];
  let rMasksI = maskInput.length;
  let rShapeI = oldShape.length;
  let rNewShapeI = newShape.length;
  const rMasks = () => rMasksI ? maskInput[--rMasksI] : [0, 1];
  const rShape = () => rShapeI ? oldShape[--rShapeI] : 1;
  const rNewShape = () => rNewShapeI ? newShape[--rNewShapeI] : 1;
  let currStride = 1;
  let [oldDim, newDim, mask] = [
    rShape(),
    rNewShape(),
    rMasks()
  ];
  while (newMask.length < newShape.length) {
    const [l, r] = mask;
    const nextStride = newDim * currStride;
    if (oldDim === nextStride) {
      newMask.push([intdiv(l, currStride), intdiv(r - 1, currStride) + 1]);
      currStride = 1;
      [oldDim, newDim, mask] = [
        rShape(),
        rNewShape(),
        rMasks()
      ];
    } else if (oldDim > nextStride) {
      if (oldDim % nextStride !== 0) return null;
      if ((l % nextStride !== 0 || r % nextStride !== 0) && intdiv(l, nextStride) !== intdiv(r - 1, nextStride)) return null;
      newMask.push([intdiv(l % nextStride, currStride), intdiv((r - 1) % nextStride, currStride) + 1]);
      [currStride, newDim] = [nextStride, rNewShape()];
    } else {
      const nextMask = rMasks();
      if (!deepEqual(mask, [0, oldDim]) && l !== r && nextMask[1] - nextMask[0] !== 1) return null;
      mask = [nextMask[0] * oldDim + l, (nextMask[1] - 1) * oldDim + r];
      oldDim *= rShape();
    }
  }
  return newMask.reverse();
}
function unravel(shape2, offset) {
  let acc = 1;
  const idxs = [];
  for (let i = shape2.length - 1; i >= 0; i--) {
    const d = shape2[i];
    idxs.push(Math.floor(offset / acc) % d);
    acc *= d;
  }
  return idxs.reverse();
}
function unravelAlu(shape2, offset) {
  let acc = 1;
  const idxs = [];
  for (let i = shape2.length - 1; i >= 0; i--) {
    const d = shape2[i];
    idxs.push(AluExp.mod(AluExp.idiv(offset, AluExp.i32(acc)), AluExp.i32(d)));
    acc *= d;
  }
  return idxs.reverse();
}
function applyLast(ar, f) {
  return ar.toSpliced(ar.length - 1, 1, f(ar[ar.length - 1]));
}
function tuneNullopt(kernel) {
  let exp3 = kernel.exp;
  const vars = {};
  vars.gidx = AluExp.special("int32", "gidx", kernel.size);
  if (kernel.reduction) {
    vars.ridx = AluExp.special("int32", "ridx", kernel.reduction.size);
    if (exp3.dtype !== kernel.reduction.dtype) exp3 = AluExp.cast(kernel.reduction.dtype, exp3);
  }
  return {
    exp: exp3.substitute(vars).rewriteGlobalViews().simplify(),
    epilogue: kernel.reduction?.epilogue.substitute({ gidx: vars.gidx }).rewriteGlobalViews().simplify(),
    outputIdxExp: vars.gidx,
    threadCount: kernel.size,
    size: { reduce: kernel.reduction ? kernel.reduction.size : 0 }
  };
}
function tuneWebgpu(kernel) {
  const reduction = kernel.reduction;
  if (!reduction) return tuneNullopt(kernel);
  const exp3 = AluExp.cast(reduction.dtype, kernel.exp);
  if (exp3.collect((exp4) => exp4.op === "GlobalIndex").length > 0) {
    if (DEBUG >= 4) console.info("Tuning: Found GlobalIndex ops, skipping opt.");
    return tuneNullopt(kernel);
  }
  const globalViews = exp3.collect((exp4) => exp4.op === "GlobalView");
  if (globalViews.length === 0) {
    if (DEBUG >= 4) console.info("Tuning: No GlobalView ops found in kernel.");
    return tuneNullopt(kernel);
  }
  const shape2 = globalViews[0].arg[1].shape;
  const expectedSrc = [...unravelAlu(shape2.slice(0, -1), AluVar.gidx), AluVar.ridx].map((e2) => e2.simplify());
  for (const gv of globalViews) if (!gv.src.length || !deepEqual(gv.src, expectedSrc)) {
    if (DEBUG >= 4) console.info("Tuning: GlobalView src[] not consistent with reduction.");
    return tuneNullopt(kernel);
  }
  if (shape2[shape2.length - 1] !== reduction.size) throw new Error("Invariant violation: shape doesn't match reduction size.");
  const sts = globalViews.map((gv) => gv.arg[1]);
  for (const st of sts) if (!deepEqual(st.shape, shape2)) throw new Error("Invariant violation: GlobalView shape mismatch");
  const dim = new TuneDims(shape2);
  const upcastedAxis = /* @__PURE__ */ new Set();
  while (prod(dim.st.shape.slice(0, dim.groups)) >= 1024) {
    const choices = [];
    const composedSts = sts.map((st) => st.compose(dim.st));
    for (let axis = 0; axis < dim.groups; axis++) for (const [priority, amount] of [
      4,
      3,
      5
    ].entries()) if (!upcastedAxis.has(axis) && dim.st.shape[axis] % amount === 0 && composedSts.some((st) => st.lastStrides[axis] === 0 && st.lastStrides.slice(dim.unroll).every((stride) => stride > 0))) {
      let nonzeroStrides = 0;
      let totalStrides = 0;
      for (const st of composedSts) {
        nonzeroStrides += st.lastStrides[axis] > 0 ? 1 : 0;
        totalStrides += st.lastStrides[axis];
      }
      choices.push([
        nonzeroStrides,
        totalStrides,
        axis,
        priority,
        amount
      ]);
    }
    if (choices.length > 0) {
      choices.sort(lexCompare);
      dim.applyUpcast(choices[0][2], choices[0][4]);
      upcastedAxis.add(choices[0][2]);
    } else break;
  }
  const groupCandidateHasContiguousInputs = sts.map((st) => st.compose(dim.st)).every((st) => {
    return !st.lastStrides.slice(0, dim.groups).some((stride) => stride !== 0) || Math.abs(st.lastStrides[dim.reduce]) <= 1;
  });
  if (reduction.op === "Add" && reduction.dtype === "float32" && groupCandidateHasContiguousInputs && prod(dim.st.shape.slice(0, dim.groups)) < 4096 && prod(dim.st.shape.slice(dim.reduce, dim.unroll)) >= 512) {
    const axis = dim.reduce;
    let amount = 16;
    while (amount > 1 && dim.st.shape[axis] % amount !== 0) amount /= 2;
    if (amount > 1) dim.applyGroup(axis, amount);
  }
  if (!/Mobi|Android/i.test(navigator.userAgent) && dim.reduce < dim.unroll && (prod(dim.st.shape.slice(dim.unroll)) <= 4 || dim.unroll === dim.upcast && prod(dim.st.shape.slice(dim.upcast)) < 64)) {
    const s = dim.st.shape[dim.unroll - 1];
    if (0 < s && s <= 32) dim.applyUnroll(dim.reduce, s);
    else for (const splits of [4, 2]) if (s % splits === 0) {
      dim.applyUnroll(dim.unroll - 1, splits);
      break;
    }
  }
  for (const ax of sorted(upcastedAxis)) {
    const s = dim.st.shape[ax];
    for (const amount of [8, 4]) if (s % amount === 0) {
      dim.applyLocal(ax, amount);
      break;
    }
  }
  const indices2 = [];
  const addIndices = (s, exp4) => {
    if (s.length === 0) return;
    else if (s.length === 1) indices2.push(exp4);
    else indices2.push(...unravelAlu(s, exp4));
  };
  if (0 < dim.groups) {
    const s = dim.st.shape.slice(0, dim.groups);
    addIndices(s, AluExp.special("int32", "gidx", prod(s)));
  }
  if (dim.groups < dim.reduce) {
    const s = dim.st.shape.slice(dim.groups, dim.reduce);
    addIndices(s, AluExp.special("int32", "group", prod(s)));
  }
  if (dim.reduce <= dim.unroll) {
    const s = dim.st.shape.slice(dim.reduce, dim.unroll);
    addIndices(s, AluExp.special("int32", "ridx", prod(s)));
  }
  if (dim.unroll < dim.upcast) addIndices(dim.st.shape.slice(dim.unroll, dim.upcast), AluVar.unroll);
  if (dim.upcast < dim.end) addIndices(dim.st.shape.slice(dim.upcast), AluVar.upcast);
  let newExp = exp3.rewrite((exp4) => {
    if (exp4.op === "GlobalView") {
      const gid = exp4.arg[0];
      const st = exp4.arg[1];
      return accessorGlobal(exp4.dtype, gid, st.compose(dim.st), indices2);
    }
  });
  const [iexpr, vexpr] = dim.st.toAluExp(indices2);
  if (vexpr.min !== 1) throw new Error("Invariant violation: vexpr !== true");
  newExp = newExp.substitute({
    gidx: AluExp.idiv(iexpr, AluExp.i32(reduction.size)).simplify(),
    ridx: AluExp.mod(iexpr, AluExp.i32(reduction.size)).simplify()
  });
  const outputGidx = dim.outputSt.shape.slice(0, dim.groups);
  const outputUpcast = dim.outputSt.shape.slice(dim.groups);
  const outputIndices = [...unravelAlu(outputGidx, AluExp.special("int32", "gidx", prod(outputGidx))), ...unravelAlu(outputUpcast, AluVar.upcast)];
  const [outputIdxExp, _] = dim.outputSt.toAluExp(outputIndices);
  const newEpilogue = reduction.epilogue.rewrite((exp4) => {
    if (exp4.op === "GlobalView") {
      const gid = exp4.arg[0];
      const st = exp4.arg[1];
      return accessorGlobal(exp4.dtype, gid, st.compose(dim.outputSt), outputIndices);
    }
  });
  if (prod(dim.st.shape.slice(dim.groups, dim.upcast)) !== reduction.size) throw new Error(`Invariant violation: reduction size ${reduction.size} does not match tuned dims ${JSON.stringify(dim.st.shape.slice(dim.groups, dim.upcast))}`);
  const size2 = {
    groups: prod(dim.st.shape.slice(dim.groups, dim.reduce)),
    reduce: prod(dim.st.shape.slice(dim.reduce, dim.unroll)),
    unroll: prod(dim.st.shape.slice(dim.unroll, dim.upcast)),
    upcast: prod(dim.st.shape.slice(dim.upcast))
  };
  return {
    exp: newExp.simplify(),
    epilogue: newEpilogue.simplify(),
    outputIdxExp: outputIdxExp.simplify(),
    threadCount: kernel.size / size2.upcast * size2.groups,
    size: size2
  };
}
function isTracing() {
  return traceEnabled;
}
function onFlushTrace(cb) {
  flushCallbacks.push(cb);
}
function humanSize(n) {
  if (n >= 1e9) return `${(n / 1e9).toPrecision(3)}B`;
  if (n >= 1e6) return `${(n / 1e6).toPrecision(3)}M`;
  if (n >= 1e3) return `${(n / 1e3).toPrecision(3)}K`;
  return `${n}`;
}
function traceSourceInfo(source) {
  const properties = [];
  let label;
  let color;
  if (source instanceof Kernel) {
    label = `Kernel[${humanSize(source.size)}]`;
    properties.push(["exp", `${source.exp}`]);
    properties.push(["size", `${source.size}`]);
    properties.push(["nargs", `${source.nargs}`]);
    if (!source.reduction) color = "primary";
    else {
      color = "secondary";
      properties.push(["reduction", `${source.reduction.op}:${source.reduction.size}`]);
    }
  } else {
    color = "tertiary";
    label = source.name;
    properties.push(["inputShapes", source.type.inputShapes.map((s) => `[${s}]`).join(", ")]);
    properties.push(["outputShapes", source.type.outputShapes.map((s) => `[${s}]`).join(", ")]);
    properties.push(["dtype", source.type.inputDtypes.join(", ")]);
  }
  return {
    label,
    color,
    properties
  };
}
function emitTrace(track, info, start, end) {
  performance.measure(info.label, {
    detail: { devtools: {
      trackGroup: "JAX Profiler",
      track,
      color: info.color,
      properties: info.properties
    } },
    start,
    end
  });
}
function alignTo(size2, alignment) {
  return Math.ceil(size2 / alignment) * alignment;
}
function _poly(cg, x, as) {
  if (as.length === 0) throw new Error("_poly needs at least one coefficient");
  cg.f32.const(as[as.length - 1]);
  for (let i = as.length - 2; i >= 0; i--) {
    cg.local.get(x);
    cg.f32.mul();
    if (as[i] !== 0) {
      cg.f32.const(as[i]);
      cg.f32.add();
    }
  }
}
function wasm_exp(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    const k_f = cg.local.declare(cg.f32);
    const k = cg.local.declare(cg.i32);
    const r = cg.local.declare(cg.f32);
    const p = cg.local.declare(cg.f32);
    const scale = cg.local.declare(cg.f32);
    cg.local.get(0);
    cg.f32.const(1 / Math.LN2);
    cg.f32.mul();
    cg.f32.nearest();
    cg.local.tee(k_f);
    cg.i32.trunc_sat_f32_s();
    cg.local.set(k);
    cg.local.get(k);
    cg.i32.const(127);
    cg.i32.gt_s();
    cg.if(cg.void);
    cg.f32.const(Infinity);
    cg.return();
    cg.end();
    cg.local.get(k);
    cg.i32.const(-126);
    cg.i32.lt_s();
    cg.if(cg.void);
    cg.f32.const(0);
    cg.return();
    cg.end();
    cg.local.get(0);
    cg.local.get(k_f);
    cg.f32.const(Math.LN2);
    cg.f32.mul();
    cg.f32.sub();
    cg.local.set(r);
    _poly(cg, r, [
      1,
      1,
      1 / 2,
      1 / 6,
      1 / 24,
      1 / 120,
      1 / 720
    ]);
    cg.local.set(p);
    cg.local.get(k);
    cg.i32.const(127);
    cg.i32.add();
    cg.i32.const(23);
    cg.i32.shl();
    cg.f32.reinterpret_i32();
    cg.local.set(scale);
    cg.local.get(p);
    cg.local.get(scale);
    cg.f32.mul();
  });
}
function wasm_log(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    const bits2 = cg.local.declare(cg.i32);
    const e2 = cg.local.declare(cg.i32);
    const m = cg.local.declare(cg.f32);
    const t = cg.local.declare(cg.f32);
    const t2 = cg.local.declare(cg.f32);
    cg.local.get(0);
    cg.f32.const(0);
    cg.f32.lt();
    cg.if(cg.void);
    cg.f32.const(NaN);
    cg.return();
    cg.end();
    cg.local.get(0);
    cg.i32.reinterpret_f32();
    cg.local.tee(bits2);
    cg.i32.const(23);
    cg.i32.shr_u();
    cg.i32.const(255);
    cg.i32.and();
    cg.i32.const(127);
    cg.i32.sub();
    cg.local.set(e2);
    cg.local.get(e2);
    cg.i32.const(-127);
    cg.i32.eq();
    cg.if(cg.void);
    cg.f32.const(-Infinity);
    cg.return();
    cg.end();
    cg.local.get(e2);
    cg.i32.const(128);
    cg.i32.eq();
    cg.if(cg.void);
    cg.local.get(0);
    cg.return();
    cg.end();
    cg.local.get(bits2);
    cg.i32.const(8388607);
    cg.i32.and();
    cg.i32.const(1065353216);
    cg.i32.or();
    cg.f32.reinterpret_i32();
    cg.local.set(m);
    cg.local.get(m);
    cg.f32.const(1);
    cg.f32.sub();
    cg.local.get(m);
    cg.f32.const(1);
    cg.f32.add();
    cg.f32.div();
    cg.local.set(t);
    cg.local.get(t);
    cg.local.get(t);
    cg.f32.mul();
    cg.local.set(t2);
    _poly(cg, t2, [
      2,
      2 / 3,
      2 / 5,
      2 / 7
    ]);
    cg.local.get(t);
    cg.f32.mul();
    cg.local.get(e2);
    cg.f32.convert_i32_s();
    cg.f32.const(Math.LN2);
    cg.f32.mul();
    cg.f32.add();
  });
}
function _sincos(cg) {
  const y = cg.local.declare(cg.f32);
  const qf = cg.local.declare(cg.f32);
  const q = cg.local.declare(cg.i32);
  const z = cg.local.declare(cg.f32);
  const z2 = cg.local.declare(cg.f32);
  const sz = cg.local.declare(cg.f32);
  const cz = cg.local.declare(cg.f32);
  cg.local.get(0);
  cg.local.get(0);
  cg.f32.const(1 / (2 * Math.PI));
  cg.f32.mul();
  cg.f32.nearest();
  cg.local.tee(qf);
  cg.f32.const(2 * Math.PI);
  cg.f32.mul();
  cg.f32.sub();
  cg.local.set(y);
  cg.local.get(y);
  cg.f32.const(2 / Math.PI);
  cg.f32.mul();
  cg.f32.nearest();
  cg.local.tee(qf);
  cg.i32.trunc_sat_f32_s();
  cg.local.set(q);
  cg.local.get(y);
  cg.local.get(qf);
  cg.f32.const(Math.PI / 2);
  cg.f32.mul();
  cg.f32.sub();
  cg.local.tee(z);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(z2);
  _poly(cg, z2, [
    1,
    -1 / 6,
    1 / 120,
    -1 / 5040
  ]);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(sz);
  _poly(cg, z2, [
    1,
    -1 / 2,
    1 / 24,
    -1 / 720,
    1 / 40320
  ]);
  cg.local.set(cz);
  return {
    q,
    sz,
    cz
  };
}
function wasm_sin(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    const { q, sz, cz } = _sincos(cg);
    const mag = cg.local.declare(cg.f32);
    cg.local.get(cz);
    cg.local.get(sz);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.and();
    cg.select();
    cg.local.tee(mag);
    cg.f32.neg();
    cg.local.get(mag);
    cg.local.get(q);
    cg.i32.const(2);
    cg.i32.and();
    cg.select();
  });
}
function wasm_cos(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    const { q, sz, cz } = _sincos(cg);
    const mag = cg.local.declare(cg.f32);
    cg.local.get(sz);
    cg.local.get(cz);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.and();
    cg.select();
    cg.local.tee(mag);
    cg.f32.neg();
    cg.local.get(mag);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.add();
    cg.i32.const(2);
    cg.i32.and();
    cg.select();
  });
}
function _atan(cg) {
  const x = cg.local.declare(cg.f32);
  const abs_x = cg.local.declare(cg.f32);
  const z = cg.local.declare(cg.f32);
  const z2 = cg.local.declare(cg.f32);
  const p = cg.local.declare(cg.f32);
  cg.local.set(x);
  cg.local.get(x);
  cg.f32.abs();
  cg.local.set(abs_x);
  cg.f32.const(1);
  cg.local.get(abs_x);
  cg.f32.div();
  cg.local.get(abs_x);
  cg.local.get(abs_x);
  cg.f32.const(1);
  cg.f32.ge();
  cg.select();
  cg.local.set(z);
  cg.local.get(z);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(z2);
  _poly(cg, z2, [
    0.999998614341,
    0.661705427875,
    0.0415796528637
  ]);
  _poly(cg, z2, [
    1,
    0.994987933645,
    0.173698870181
  ]);
  cg.f32.div();
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(p);
  cg.f32.const(Math.PI / 2);
  cg.local.get(p);
  cg.f32.sub();
  cg.local.get(p);
  cg.local.get(abs_x);
  cg.f32.const(1);
  cg.f32.ge();
  cg.select();
  cg.local.get(x);
  cg.f32.copysign();
}
function wasm_atan(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.local.get(0);
    _atan(cg);
  });
}
function wasm_asin(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.local.get(0);
    cg.f32.const(1);
    cg.local.get(0);
    cg.local.get(0);
    cg.f32.mul();
    cg.f32.sub();
    cg.f32.sqrt();
    cg.f32.const(1);
    cg.f32.add();
    cg.f32.div();
    _atan(cg);
    cg.f32.const(2);
    cg.f32.mul();
  });
}
function _erfapprox(cg, exp_func) {
  const x = cg.local.declare(cg.f32);
  const t = cg.local.declare(cg.f32);
  cg.local.set(x);
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  cg.f32.const(1);
  cg.f32.const(1);
  cg.f32.const(p);
  cg.local.get(x);
  cg.f32.mul();
  cg.f32.add();
  cg.f32.div();
  cg.local.set(t);
  _poly(cg, t, [
    0,
    a1,
    a2,
    a3,
    a4,
    a5
  ]);
  cg.local.get(x);
  cg.f32.neg();
  cg.local.get(x);
  cg.f32.mul();
  cg.call(exp_func);
  cg.f32.mul();
}
function wasm_erf(cg, exp3) {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.f32.const(1);
    cg.local.get(0);
    cg.f32.abs();
    _erfapprox(cg, exp3);
    cg.f32.sub();
    cg.local.get(0);
    cg.f32.copysign();
  });
}
function wasm_erfc(cg, exp3) {
  return cg.function([cg.f32], [cg.f32], () => {
    const e2 = cg.local.declare(cg.f32);
    cg.local.get(0);
    cg.f32.abs();
    _erfapprox(cg, exp3);
    cg.local.set(e2);
    cg.f32.const(2);
    cg.local.get(e2);
    cg.f32.sub();
    cg.local.get(e2);
    cg.local.get(0);
    cg.f32.const(0);
    cg.f32.lt();
    cg.select();
  });
}
function wasm_threefry2x32(cg) {
  return cg.function([
    cg.i32,
    cg.i32,
    cg.i32,
    cg.i32
  ], [cg.i32, cg.i32], () => {
    const ks0 = cg.local.declare(cg.i32);
    const ks1 = cg.local.declare(cg.i32);
    const ks2 = cg.local.declare(cg.i32);
    const x0 = cg.local.declare(cg.i32);
    const x1 = cg.local.declare(cg.i32);
    const mix = (rot) => {
      cg.local.get(x0);
      cg.local.get(x1);
      cg.i32.add();
      cg.local.set(x0);
      cg.local.get(x1);
      cg.i32.const(rot);
      cg.i32.rotl();
      cg.local.get(x0);
      cg.i32.xor();
      cg.local.set(x1);
    };
    const keySchedule = (k0, k1, round3) => {
      cg.local.get(x0);
      cg.local.get(k0);
      cg.i32.add();
      cg.local.set(x0);
      cg.local.get(x1);
      cg.local.get(k1);
      cg.i32.add();
      cg.i32.const(round3);
      cg.i32.add();
      cg.local.set(x1);
    };
    cg.local.get(0);
    cg.local.set(ks0);
    cg.local.get(1);
    cg.local.set(ks1);
    cg.local.get(0);
    cg.local.get(1);
    cg.i32.xor();
    cg.i32.const(466688986);
    cg.i32.xor();
    cg.local.set(ks2);
    cg.local.get(2);
    cg.local.get(ks0);
    cg.i32.add();
    cg.local.set(x0);
    cg.local.get(3);
    cg.local.get(ks1);
    cg.i32.add();
    cg.local.set(x1);
    mix(13), mix(15), mix(26), mix(6);
    keySchedule(ks1, ks2, 1);
    mix(17), mix(29), mix(16), mix(24);
    keySchedule(ks2, ks0, 2);
    mix(13), mix(15), mix(26), mix(6);
    keySchedule(ks0, ks1, 3);
    mix(17), mix(29), mix(16), mix(24);
    keySchedule(ks1, ks2, 4);
    mix(13), mix(15), mix(26), mix(6);
    keySchedule(ks2, ks0, 5);
    cg.local.get(x0);
    cg.local.get(x1);
  });
}
function hasWasmFeature(feature) {
  const cached = featureSupportCache.get(feature);
  if (cached !== void 0) return cached;
  const testHex = featureProbes[feature];
  let supported = false;
  try {
    supported = typeof WebAssembly !== "undefined" && WebAssembly.validate(decodeHex(testHex));
  } catch {
    supported = false;
  }
  featureSupportCache.set(feature, supported);
  return supported;
}
function decodeHex(hex) {
  const bytes = new Uint8Array(/* @__PURE__ */ new ArrayBuffer(hex.length / 2));
  for (let i = 0; i < bytes.length; i++) bytes[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return bytes;
}
function hasSharedArrayBuffer() {
  return typeof SharedArrayBuffer !== "undefined" && typeof Worker !== "undefined";
}
function createWorkerPool(memory) {
  if (!hasSharedArrayBuffer()) return null;
  try {
    return new WasmWorkerPool(memory, Math.max(1, typeof navigator !== "undefined" && navigator.hardwareConcurrency || 4));
  } catch {
    return null;
  }
}
function isSymbol(exp3, name) {
  return exp3.op === "Variable" && exp3.arg === name || exp3.op === "Special" && exp3.arg[0] === name;
}
function referencesSymbol(exp3, name) {
  return isSymbol(exp3, name) || exp3.src.some((src) => referencesSymbol(src, name));
}
function referencesGidx(exp3) {
  return referencesSymbol(exp3, "gidx");
}
function hasFragmentRisk(tileSize, N) {
  return isFinite(tileSize) && tileSize > N && tileSize % N !== 0;
}
function constInt(exp3) {
  if (exp3.op !== "Const") return null;
  const value = exp3.arg;
  return Number.isInteger(value) ? value : null;
}
function coefficientOfSymbol(exp3, name) {
  if (!referencesSymbol(exp3, name)) return 0;
  if (isSymbol(exp3, name)) return 1;
  if (exp3.op === "Add" || exp3.op === "Sub") {
    const a = coefficientOfSymbol(exp3.src[0], name);
    const b = coefficientOfSymbol(exp3.src[1], name);
    if (a === null || b === null) return null;
    return exp3.op === "Add" ? a + b : a - b;
  }
  if (exp3.op === "Mul") {
    const lhs = constInt(exp3.src[0]);
    if (lhs !== null) {
      const rhsCoeff = coefficientOfSymbol(exp3.src[1], name);
      return rhsCoeff === null ? null : lhs * rhsCoeff;
    }
    const rhs = constInt(exp3.src[1]);
    if (rhs !== null) {
      const lhsCoeff = coefficientOfSymbol(exp3.src[0], name);
      return lhsCoeff === null ? null : rhs * lhsCoeff;
    }
  }
  return null;
}
function rewriteSymbol(exp3, name, rewrite) {
  return exp3.rewrite((node) => isSymbol(node, name) ? rewrite(node) : void 0).simplify();
}
function repeatsAcrossGidxTile(exp3, tileSize) {
  if (!isFinite(tileSize)) return false;
  return rewriteSymbol(exp3, "gidx", (node) => AluExp.add(node, AluExp.i32(tileSize))).getHash() === exp3.getHash();
}
function divisorAtMost(value, limit) {
  for (let i = Math.min(value, limit); i > 1; i--) if (value % i === 0) return i;
  return 1;
}
function commonTileSize(kernelSize, strideMap, minTileSize, unconstrainedTileSize = null) {
  const tileSizes = [];
  for (const stride of strideMap.values()) if (stride.kind !== "gather" && isFinite(stride.tileSize)) tileSizes.push(stride.tileSize);
  if (tileSizes.length === 0) return unconstrainedTileSize;
  const tileSize = Math.min(...tileSizes);
  if (tileSize < minTileSize || kernelSize % tileSize !== 0 || tileSizes.some((size2) => size2 % tileSize !== 0)) return null;
  return tileSize;
}
function tiledRows(kernelSize, tileSize) {
  const rowCount = Math.floor(kernelSize / tileSize);
  return findPow2(0, Math.max(1, Math.min(rowCount / 2, TILED_SIMD_ROWS)));
}
function tiledColumns(tileSize, laneWidth) {
  const columns = tileSize / laneWidth;
  return findPow2(0, Math.max(1, Math.min(columns / 2, Math.floor(TILED_SIMD_COLUMNS / laneWidth))));
}
function microTile(tile2, axis, limit) {
  return divisorAtMost(gcd(tile2, axis), limit);
}
function microTileWithTail(tile2, limit) {
  return Math.max(1, Math.min(tile2, limit));
}
function periodicStride(exp3, kind) {
  if (exp3.src[1].op !== "Const") return null;
  const N = exp3.src[1].arg;
  const inner2 = analyzeStride(exp3.src[0]);
  if (inner2.kind === "broadcast") return inner2;
  if (inner2.kind !== "contiguous" || hasFragmentRisk(inner2.tileSize, N)) return { kind: "gather" };
  return {
    kind,
    tileSize: Math.min(inner2.tileSize, N)
  };
}
function addStrides(lhs, rhs) {
  if (lhs.kind === "gather" || rhs.kind === "gather") return { kind: "gather" };
  const tileSize = Math.min(lhs.tileSize, rhs.tileSize);
  if (lhs.kind === "broadcast") return {
    kind: rhs.kind,
    tileSize
  };
  if (rhs.kind === "broadcast") return {
    kind: lhs.kind,
    tileSize
  };
  return { kind: "gather" };
}
function analyzeStride(exp3) {
  if (!referencesGidx(exp3)) return {
    kind: "broadcast",
    tileSize: Infinity
  };
  if (exp3.op === "Special" && exp3.arg[0] === "gidx") return {
    kind: "contiguous",
    tileSize: Infinity
  };
  if (exp3.op === "Idiv" || exp3.op === "Mod") {
    const stride = periodicStride(exp3, exp3.op === "Idiv" ? "broadcast" : "contiguous");
    if (stride) return stride;
  }
  if (exp3.op === "Mul") {
    for (let i = 0; i < 2; i++) if (exp3.src[i].op === "Const") {
      const inner2 = analyzeStride(exp3.src[1 - i]);
      if (inner2.kind === "broadcast") return inner2;
      return { kind: "gather" };
    }
  }
  if (exp3.op === "Add") return addStrides(analyzeStride(exp3.src[0]), analyzeStride(exp3.src[1]));
  return { kind: "gather" };
}
function simdStrideResult(globalIndex) {
  const index = globalIndex.src[0];
  const result = analyzeStride(index);
  const [_, len] = globalIndex.arg;
  if (result.kind !== "gather" && (result.tileSize < 4 || isFinite(result.tileSize) && result.tileSize % 4 !== 0)) return { kind: "gather" };
  if (result.kind === "contiguous" && (index.min < 0 || index.max >= len)) return { kind: "gather" };
  return result;
}
function collectSimdStrides(exp3) {
  return new Map((exp3?.collect((node) => node.op === "GlobalIndex") ?? []).map((gi) => [gi, simdStrideResult(gi)]));
}
function reductionPointerCandidates(exp3, strideMap) {
  const candidates = [];
  for (const gi of exp3.collect((node) => node.op === "GlobalIndex")) {
    const stride = strideMap?.get(gi) ?? {
      kind: "broadcast",
      tileSize: Infinity
    };
    if (strideMap && stride.kind === "gather") continue;
    const [gid, len] = gi.arg;
    const index = gi.src[0];
    const strideElems = coefficientOfSymbol(index, "ridx");
    if (strideElems === null || !Number.isInteger(strideElems)) continue;
    if (index.min < 0 || index.max >= len) continue;
    candidates.push({
      exp: gi,
      gid,
      dtype: gi.dtype,
      stride,
      baseIndex: rewriteSymbol(index, "ridx", () => AluExp.i32(0)),
      strideBytes: strideElems * byteWidth(gi.dtype)
    });
  }
  return candidates;
}
function reductionTilePlan(kernel, strideMap, canStorePartial = true) {
  if (!kernel.reduction) return null;
  const tileSize = commonTileSize(kernel.size, strideMap, 4, 4);
  if (tileSize === null) return null;
  const tileRows = tiledRows(kernel.size, tileSize);
  const tileVectors = tiledColumns(tileSize, 4);
  return {
    tileSize,
    tileRows,
    tileVectors,
    tileK: canStorePartial ? divisorAtMost(kernel.reduction.size, TILED_SIMD_K) : kernel.reduction.size,
    microRows: microTile(tileRows, Math.floor(kernel.size / tileSize), TILED_SIMD_MICRO_ROWS),
    microVectors: microTileWithTail(tileVectors, TILED_SIMD_MICRO_VECTORS)
  };
}
function reductionKTilePlan(kernel, strideMap) {
  if (!kernel.reduction) return null;
  if (kernel.reduction.size % 4 !== 0) return null;
  const tileSize = commonTileSize(kernel.size, strideMap, 1, 1);
  if (tileSize === null) return null;
  const kUnroll = divisorAtMost(kernel.reduction.size / 4, K_SIMD_MAX_UNROLL);
  const tileRows = tiledRows(kernel.size, tileSize);
  const tileCols = tiledColumns(tileSize, 1);
  return {
    tileSize,
    tileRows,
    tileCols,
    microRows: microTile(tileRows, Math.floor(kernel.size / tileSize), K_SIMD_MICRO_ROWS),
    microCols: microTile(tileCols, tileSize, K_SIMD_MICRO_COLS),
    kUnroll
  };
}
function pointerShareKey(candidate, row, vector, groupIndex) {
  const hash = candidate.exp.getHash().toString();
  const { stride } = candidate;
  if (stride.kind === "broadcast") return isFinite(stride.tileSize) ? `${hash}:row${row}` : `${hash}:all`;
  if (stride.kind === "contiguous" && isFinite(stride.tileSize)) return repeatsAcrossGidxTile(candidate.baseIndex, stride.tileSize) ? `${hash}:vec${vector}` : `${hash}:row${row}:vec${vector}`;
  return `${hash}:g${groupIndex}`;
}
function kReductionPointerShareKey(candidate, outputStride, tileSize, row, col, groupIndex) {
  const hash = candidate.exp.getHash().toString();
  if (outputStride.kind === "broadcast" && isFinite(outputStride.tileSize)) return `${hash}:row${row}`;
  if (repeatsAcrossGidxTile(candidate.baseIndex, tileSize)) return `${hash}:col${col}`;
  return `${hash}:g${groupIndex}`;
}
function translateExp(cg, funcs, exp3, ctx, pointerMap = /* @__PURE__ */ new Map()) {
  const references = /* @__PURE__ */ new Map();
  const seen = /* @__PURE__ */ new Set();
  const countReferences = (exp4) => {
    references.set(exp4, (references.get(exp4) ?? 0) + 1);
    if (!seen.has(exp4)) {
      seen.add(exp4);
      for (const src of exp4.src) countReferences(src);
    }
  };
  const expContext = /* @__PURE__ */ new Map();
  const gen = (exp4) => {
    if (expContext.has(exp4)) return cg.local.get(expContext.get(exp4));
    const { op, src, dtype, arg } = exp4;
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      gen(src[0]), gen(src[1]);
      if (op === "Add") if (dtype === "bool") cg.i32.or();
      else dty(cg, op, dtype).add();
      else if (op === "Sub") dty(cg, op, dtype).sub();
      else if (op === "Mul") if (dtype === "bool") cg.i32.and();
      else dty(cg, op, dtype).mul();
      else if (op === "Idiv") if (isFloatDtype(dtype)) {
        dtyF(cg, op, dtype).div();
        dtyF(cg, op, dtype).trunc();
      } else if (dtype === "uint32") cg.i32.div_u();
      else if (dtype === "int32") cg.i32.div_s();
      else throw new UnsupportedOpError(op, dtype, "wasm");
      else if (op === "Mod") if (isFloatDtype(dtype)) {
        const dt = dtyF(cg, op, dtype);
        const a = cg.local.declare(dt);
        const b = cg.local.declare(dt);
        cg.local.set(b);
        cg.local.tee(a);
        cg.local.get(a);
        cg.local.get(b);
        dt.div();
        dt.trunc();
        cg.local.get(b);
        dt.mul();
        dt.sub();
      } else if (dtype === "uint32") cg.i32.rem_u();
      else if (dtype === "int32") cg.i32.rem_s();
      else throw new UnsupportedOpError(op, dtype, "wasm");
      else if (op === "Min" || op === "Max") if (isFloatDtype(dtype)) if (op === "Min") dtyF(cg, op, dtype).min();
      else dtyF(cg, op, dtype).max();
      else if (dtype === "int32" || dtype === "uint32" || dtype === "bool") {
        const a = cg.local.declare(cg.i32);
        const b = cg.local.declare(cg.i32);
        cg.local.set(b);
        cg.local.tee(a);
        cg.local.get(b);
        cg.local.get(a);
        cg.local.get(b);
        if (dtype === "int32") if (op === "Min") cg.i32.lt_s();
        else cg.i32.gt_s();
        else if (op === "Min") cg.i32.lt_u();
        else cg.i32.gt_u();
        cg.select();
      } else throw new UnsupportedOpError(op, dtype, "wasm");
      else if (op === "BitCombine") if (arg === "and") cg.i32.and();
      else if (arg === "or") cg.i32.or();
      else cg.i32.xor();
      else if (op === "BitShift") if (arg === "shl") cg.i32.shl();
      else cg.i32.shr_u();
      else if (op === "Cmplt") {
        const srcDtype = src[0].dtype;
        if (isFloatDtype(srcDtype)) dtyF(cg, op, srcDtype).lt();
        else if (srcDtype === "int32") cg.i32.lt_s();
        else if (srcDtype === "uint32") cg.i32.lt_u();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === "Cmpne") dty(cg, op, src[0].dtype).ne();
      else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (AluGroup.Unary.has(op)) {
      const callFuncF32 = (func) => {
        if (dtype !== "float32") if (dtype === "float64") cg.f32.demote_f64();
        else throw new UnsupportedOpError(op, dtype, "wasm");
        cg.call(func);
        if (dtype === "float64") cg.f64.promote_f32();
      };
      if (op === "Sin") gen(src[0]), callFuncF32(funcs.sin);
      else if (op === "Cos") gen(src[0]), callFuncF32(funcs.cos);
      else if (op === "Asin") gen(src[0]), callFuncF32(funcs.asin);
      else if (op === "Atan") gen(src[0]), callFuncF32(funcs.atan);
      else if (op === "Exp") gen(src[0]), callFuncF32(funcs.exp);
      else if (op === "Log") gen(src[0]), callFuncF32(funcs.log);
      else if (op === "Erf") gen(src[0]), callFuncF32(funcs.erf);
      else if (op === "Erfc") gen(src[0]), callFuncF32(funcs.erfc);
      else if (op === "Sqrt") gen(src[0]), dtyF(cg, op, dtype).sqrt();
      else if (op === "Reciprocal") {
        const dt = dtyF(cg, op, dtype);
        dt.const(1), gen(src[0]), dt.div();
      } else if (op === "Floor") gen(src[0]), dtyF(cg, op, dtype).floor();
      else if (op === "Ceil") gen(src[0]), dtyF(cg, op, dtype).ceil();
      else if (op === "Cast") {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        const i32repr = dtype0 === "int32" || dtype0 === "uint32" || dtype0 === "bool";
        if (dtype === "int32") if (dtype0 === "float32") cg.i32.trunc_sat_f32_s();
        else if (dtype0 === "float64") cg.i32.trunc_sat_f64_s();
        else if (i32repr) ;
        else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else if (dtype === "uint32") if (dtype0 === "float32") cg.i32.trunc_sat_f32_u();
        else if (dtype0 === "float64") cg.i32.trunc_sat_f64_u();
        else if (i32repr) ;
        else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else if (dtype === "float32") if (dtype0 === "float32") ;
        else if (dtype0 === "float64") cg.f32.demote_f64();
        else if (dtype0 === "int32" || dtype0 === "bool") cg.f32.convert_i32_s();
        else if (dtype0 === "uint32") cg.f32.convert_i32_u();
        else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else if (dtype === "float64") if (dtype0 === "float32") cg.f64.promote_f32();
        else if (dtype0 === "float64") ;
        else if (dtype0 === "int32" || dtype0 === "bool") cg.f64.convert_i32_s();
        else if (dtype0 === "uint32") cg.f64.convert_i32_u();
        else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else if (dtype === "bool") if (dtype0 === "bool") ;
        else if (i32repr) cg.i32.const(0), cg.i32.ne();
        else if (dtype0 === "float32") cg.f32.const(0), cg.f32.ne();
        else if (dtype0 === "float64") cg.f64.const(0), cg.f64.ne();
        else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === "Bitcast") {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        if (dtype !== dtype0) {
          const i32repr = dtype0 === "int32" || dtype0 === "uint32";
          if (dtype === "int32" || dtype === "uint32") if (dtype0 === "float32") cg.i32.reinterpret_f32();
          else if (i32repr) ;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          else if (dtype === "float32") if (i32repr) cg.f32.reinterpret_i32();
          else if (dtype0 === "float32") ;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          else throw new UnsupportedOpError(op, dtype, "wasm");
        }
      } else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (op === "Where") {
      gen(src[1]);
      gen(src[2]);
      gen(src[0]);
      cg.select();
    } else if (op === "Threefry2x32") {
      for (let i = 0; i < 4; i++) gen(src[i]);
      cg.call(funcs.threefry2x32);
      if (arg === "xor") cg.i32.xor();
      else if (arg === 0) cg.drop();
      else if (arg === 1) {
        const local = cg.local.declare(cg.i32);
        cg.local.set(local);
        cg.drop();
        cg.local.get(local);
      } else throw new UnsupportedOpError(op, dtype, "wasm", arg);
    } else if (op === "Const") return dty(cg, op, dtype).const(arg);
    else if (op === "Special") return cg.local.get(ctx[arg[0]]);
    else if (op === "Variable") return cg.local.get(ctx[arg]);
    else if (op === "GlobalIndex") {
      const [gid, len] = arg;
      const pointer = pointerMap.get(exp4);
      if (pointer) cg.local.get(pointer.ptr);
      else {
        gen(src[0]);
        const local = cg.local.declare(cg.i32);
        cg.local.tee(local);
        cg.i32.const(0);
        cg.local.get(local), cg.i32.const(len), cg.i32.lt_u();
        cg.select();
        cg.i32.const(byteWidth(dtype));
        cg.i32.mul();
        cg.local.get(gid);
        cg.i32.add();
      }
      dty(cg, op, dtype).load(Math.log2(byteWidth(dtype)));
    } else throw new UnsupportedOpError(op, dtype, "wasm");
    if ((references.get(exp4) ?? 0) > 1) {
      const local = cg.local.declare(dty(cg, op, dtype));
      cg.local.tee(local);
      expContext.set(exp4, local);
    }
  };
  countReferences(exp3);
  gen(exp3);
}
function translateExpSimd(cg, funcs, exp3, ctx, strideMap, pointerMap = /* @__PURE__ */ new Map(), pointerValueCache = /* @__PURE__ */ new Map()) {
  const references = /* @__PURE__ */ new Map();
  const seen = /* @__PURE__ */ new Set();
  const countReferences = (exp4) => {
    references.set(exp4, (references.get(exp4) ?? 0) + 1);
    if (!seen.has(exp4)) {
      seen.add(exp4);
      for (const src of exp4.src) countReferences(src);
    }
  };
  const expContext = /* @__PURE__ */ new Map();
  const gen = (exp4) => {
    if (expContext.has(exp4)) return cg.local.get(expContext.get(exp4));
    const { op, src, arg, dtype } = exp4;
    const isInt = dtype === "int32" || dtype === "uint32" || dtype === "bool";
    const isSigned = dtype === "int32";
    if (op === "Add") {
      gen(src[0]), gen(src[1]);
      if (dtype === "bool") cg.v128.or();
      else if (isInt) cg.i32x4.add();
      else cg.f32x4.add();
    } else if (op === "Sub") {
      gen(src[0]), gen(src[1]);
      if (isInt) cg.i32x4.sub();
      else cg.f32x4.sub();
    } else if (op === "Mul") {
      gen(src[0]), gen(src[1]);
      if (dtype === "bool") cg.v128.and();
      else if (isInt) cg.i32x4.mul();
      else cg.f32x4.mul();
    } else if (op === "Min") {
      gen(src[0]), gen(src[1]);
      if (isInt) if (isSigned) cg.i32x4.min_s();
      else cg.i32x4.min_u();
      else cg.f32x4.min();
    } else if (op === "Max") {
      gen(src[0]), gen(src[1]);
      if (isInt) if (isSigned) cg.i32x4.max_s();
      else cg.i32x4.max_u();
      else cg.f32x4.max();
    } else if (op === "Sqrt") {
      gen(src[0]);
      cg.f32x4.sqrt();
    } else if (op === "Floor") {
      gen(src[0]);
      cg.f32x4.floor();
    } else if (op === "Ceil") {
      gen(src[0]);
      cg.f32x4.ceil();
    } else if (op === "Const") if (isInt) {
      cg.i32.const(arg);
      cg.i32x4.splat();
    } else {
      cg.f32.const(arg);
      cg.f32x4.splat();
    }
    else if (op === "Cast") {
      gen(src[0]);
      const dtype0 = src[0].dtype;
      const src0IsInt = dtype0 === "int32" || dtype0 === "uint32" || dtype0 === "bool";
      if (isInt && !src0IsInt) if (isSigned) cg.i32x4.trunc_sat_f32x4_s();
      else cg.i32x4.trunc_sat_f32x4_u();
      else if (!isInt && src0IsInt) if (dtype0 === "int32" || dtype0 === "bool") cg.f32x4.convert_i32x4_s();
      else cg.f32x4.convert_i32x4_u();
    } else if (op === "Cmplt") {
      gen(src[0]), gen(src[1]);
      const srcDtype = src[0].dtype;
      if (srcDtype === "float32") cg.f32x4.lt();
      else if (srcDtype === "int32") cg.i32x4.lt_s();
      else if (srcDtype === "uint32") cg.i32x4.lt_u();
      else throw new UnsupportedOpError(op, dtype, "wasm");
      cg.i32.const(1);
      cg.i32x4.splat();
      cg.v128.and();
    } else if (op === "Cmpne") {
      gen(src[0]), gen(src[1]);
      if (src[0].dtype === "float32") cg.f32x4.ne();
      else cg.i32x4.ne();
      cg.i32.const(1);
      cg.i32x4.splat();
      cg.v128.and();
    } else if (op === "Where") {
      gen(src[1]);
      gen(src[2]);
      gen(src[0]);
      cg.i32.const(0);
      cg.i32x4.splat();
      cg.i32x4.ne();
      cg.v128.bitselect();
    } else if (op === "Variable" || op === "Special") throw new Error(`translateExpSimd: unexpected ${op}(${arg})`);
    else if (op === "GlobalIndex") {
      const [gid, len] = arg;
      const indexSubtree = src[0];
      const pointer = pointerMap.get(exp4);
      const stride = pointer?.stride ?? strideMap.get(exp4) ?? { kind: "gather" };
      if (pointer) {
        const cached = pointer.valueKey ? pointerValueCache.get(pointer.valueKey) : void 0;
        if (cached !== void 0) cg.local.get(cached);
        else {
          cg.local.get(pointer.ptr);
          if (stride.kind === "contiguous") if (isInt) cg.i32x4.load(4);
          else cg.f32x4.load(4);
          else if (stride.kind === "broadcast") if (isInt) {
            cg.i32.load(2);
            cg.i32x4.splat();
          } else {
            cg.f32.load(2);
            cg.f32x4.splat();
          }
          else throw new Error("reduction pointer plan cannot use gather loads");
          if (pointer.valueKey) {
            const local = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
            cg.local.tee(local);
            pointerValueCache.set(pointer.valueKey, local);
          }
        }
      } else if (stride.kind === "contiguous") {
        translateExp(cg, funcs, indexSubtree, ctx);
        {
          const maxIdx = Math.max(len - 4, 0);
          const wideIdx = cg.local.declare(cg.i32);
          cg.local.set(wideIdx);
          cg.local.get(wideIdx);
          cg.i32.const(maxIdx);
          cg.local.get(wideIdx);
          cg.i32.const(maxIdx);
          cg.i32.lt_u();
          cg.select();
        }
        cg.i32.const(byteWidth(dtype));
        cg.i32.mul();
        cg.local.get(gid);
        cg.i32.add();
        if (isInt) cg.i32x4.load(4);
        else cg.f32x4.load(4);
      } else if (stride.kind === "broadcast") {
        translateExp(cg, funcs, indexSubtree, ctx);
        const local = cg.local.declare(cg.i32);
        cg.local.tee(local);
        cg.i32.const(0);
        cg.local.get(local), cg.i32.const(len), cg.i32.lt_u();
        cg.select();
        cg.i32.const(byteWidth(dtype));
        cg.i32.mul();
        cg.local.get(gid);
        cg.i32.add();
        if (isInt) {
          cg.i32.load(2);
          cg.i32x4.splat();
        } else {
          cg.f32.load(2);
          cg.f32x4.splat();
        }
      } else {
        const steppingLocal = ctx["gidx"];
        const origValue = cg.local.declare(cg.i32);
        cg.local.get(steppingLocal);
        cg.local.set(origValue);
        if (isInt) {
          cg.i32.const(0);
          cg.i32x4.splat();
        } else {
          cg.f32.const(0);
          cg.f32x4.splat();
        }
        const vec = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
        cg.local.set(vec);
        const idx = cg.local.declare(cg.i32);
        const scalarVal = cg.local.declare(isInt ? cg.i32 : cg.f32);
        for (let lane = 0; lane < 4; lane++) {
          cg.local.get(origValue);
          if (lane > 0) {
            cg.i32.const(lane);
            cg.i32.add();
          }
          cg.local.set(steppingLocal);
          translateExp(cg, funcs, indexSubtree, ctx);
          cg.local.tee(idx);
          cg.i32.const(0);
          cg.local.get(idx), cg.i32.const(len), cg.i32.lt_u();
          cg.select();
          cg.i32.const(byteWidth(dtype));
          cg.i32.mul();
          cg.local.get(gid);
          cg.i32.add();
          if (isInt) cg.i32.load(2);
          else cg.f32.load(2);
          cg.local.set(scalarVal);
          cg.local.get(vec);
          cg.local.get(scalarVal);
          if (isInt) cg.i32x4.replace_lane(lane);
          else cg.f32x4.replace_lane(lane);
          cg.local.set(vec);
        }
        cg.local.get(origValue);
        cg.local.set(steppingLocal);
        cg.local.get(vec);
      }
    } else throw new Error(`translateExpSimd: unsupported op ${op}`);
    if ((references.get(exp4) ?? 0) > 1) {
      const local = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
      cg.local.tee(local);
      expContext.set(exp4, local);
    }
  };
  countReferences(exp3);
  gen(exp3);
}
function dty(cg, op, dtype) {
  switch (dtype) {
    case "float32":
      return cg.f32;
    case "float64":
      return cg.f64;
    case "int32":
    case "uint32":
    case "bool":
      return cg.i32;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}
function dtyF(cg, op, dtype) {
  switch (dtype) {
    case "float32":
      return cg.f32;
    case "float64":
      return cg.f64;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}
function assert(condition, message) {
  if (!condition) throw new Error(message || "Assertion failed");
}
function encodeSigned(n) {
  const out = [];
  let more = true;
  while (more) {
    let byte = n & 127;
    n >>= 7;
    if (n === 0 && (byte & 64) === 0 || n === -1 && (byte & 64) !== 0) more = false;
    else byte |= 128;
    out.push(byte);
  }
  return out;
}
function encodeUnsigned(n) {
  const out = [];
  do {
    let byte = n & 127;
    n = n >>> 7;
    if (n !== 0) byte |= 128;
    out.push(byte);
  } while (n !== 0);
  return out;
}
function encodeString(s) {
  const bytes = new TextEncoder().encode(s);
  return [bytes.length, ...bytes];
}
function encodeBlocktype(type) {
  assert(type.length > 0, "blocktype must have at least one type");
  if (type.length === 1) return [type[0].typeId];
  return [
    96,
    ...encodeUnsigned(0),
    ...encodeUnsigned(type.length),
    ...type.map((t) => t.typeId)
  ];
}
function encodeOpcode(opcode) {
  if (typeof opcode === "number") return [opcode];
  return [opcode[0], ...encodeUnsigned(opcode[1])];
}
function appendLengthEncodedBlock(out, inp) {
  out.push(...encodeUnsigned(inp.length));
  for (const b of inp) out.push(b);
}
function UNARY_OP(op, opcode, inType, outType) {
  return function() {
    assert(this.cg._pop().typeId === this.cg[inType].typeId, `invalid type for ${op} (${inType} -> ${outType})`);
    this.cg._emit(encodeOpcode(opcode));
    this.cg._push(this.cg[outType]);
  };
}
function BINARY_OP(op, opcode, typeA, typeB, outType) {
  return function() {
    const b = this.cg._pop();
    assert(this.cg._pop().typeId === this.cg[typeA].typeId && b.typeId === this.cg[typeB].typeId, `invalid type for ${op} (${typeA}, ${typeB} -> ${outType})`);
    this.cg._emit(encodeOpcode(opcode));
    this.cg._push(this.cg[outType]);
  };
}
function LOAD_OP(op, opcode, outType) {
  return function(align = 0, offset = 0) {
    assert(this.cg._pop().typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg._emit(encodeOpcode(opcode));
    this.cg._emit(encodeUnsigned(align));
    this.cg._emit(encodeUnsigned(offset));
    this.cg._push(this.cg[outType]);
  };
}
function STORE_OP(op, opcode, inType) {
  return function(align = 0, offset = 0) {
    const valType = this.cg._pop();
    const idxType = this.cg._pop();
    assert(valType.typeId === this.cg[inType].typeId, `invalid value type for ${op} (${inType})`);
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg._emit(encodeOpcode(opcode));
    this.cg._emit(encodeUnsigned(align));
    this.cg._emit(encodeUnsigned(offset));
  };
}
function VECTOR_OP(op, vopcode, inTypes, outType) {
  return function() {
    for (const inType of inTypes.toReversed()) assert(this.cg._pop().typeId === this.cg[inType].typeId, `invalid type for ${op} (${inTypes.join(", ")} -> ${outType})`);
    this.cg._emit(encodeOpcode([253, vopcode]));
    this.cg._push(this.cg[outType]);
  };
}
function VECTOR_OPL(op, vopcode, inTypes, outType) {
  return function(lane) {
    for (const inType of inTypes.toReversed()) assert(this.cg._pop().typeId === this.cg[inType].typeId, `invalid type for ${op} (${inTypes} -> ${outType})`);
    this.cg._emit(encodeOpcode([253, vopcode]));
    this.cg._emit(lane);
    this.cg._push(this.cg[outType]);
  };
}
function VECTOR_LOAD_OP(op, vopcode) {
  return function(align = 0, offset = 0) {
    assert(this.cg._pop().typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg._emit(encodeOpcode([253, vopcode]));
    this.cg._emit(encodeUnsigned(align));
    this.cg._emit(encodeUnsigned(offset));
    this.cg._push(this.cg.v128);
  };
}
function isIdentityEpilogue(exp3) {
  return exp3?.op === "Variable" && exp3.arg === "acc";
}
function initializeReductionPointer(cg, funcs, candidate, ctx, valueKey, ridxOffset) {
  const ptr = cg.local.declare(cg.i32);
  translateExp(cg, funcs, candidate.baseIndex, ctx);
  cg.i32.const(byteWidth(candidate.dtype));
  cg.i32.mul();
  cg.local.get(candidate.gid);
  cg.i32.add();
  if (ridxOffset !== void 0 && candidate.strideBytes !== 0) {
    cg.local.get(ridxOffset);
    cg.i32.const(candidate.strideBytes);
    cg.i32.mul();
    cg.i32.add();
  }
  cg.local.set(ptr);
  return {
    ...candidate,
    ptr,
    valueKey
  };
}
function incrementReductionPointers(cg, pointers, multiplier = 1) {
  for (const pointer of pointers) {
    if (pointer.strideBytes === 0) continue;
    cg.local.get(pointer.ptr);
    cg.i32.const(pointer.strideBytes * multiplier);
    cg.i32.add();
    cg.local.set(pointer.ptr);
  }
}
function emitSimdReductionOp(cg, re, reIsInt, valueAlreadyAccumulated) {
  if (!reIsInt && valueAlreadyAccumulated) return;
  switch (re.op) {
    case "Add":
      if (reIsInt) cg.i32x4.add();
      else cg.f32x4.add();
      return;
    case "Mul":
      if (reIsInt) cg.i32x4.mul();
      else cg.f32x4.mul();
      return;
    case "Min":
      if (reIsInt) if (re.dtype === "int32") cg.i32x4.min_s();
      else cg.i32x4.min_u();
      else cg.f32x4.min();
      return;
    case "Max":
      if (reIsInt) if (re.dtype === "int32") cg.i32x4.max_s();
      else cg.i32x4.max_u();
      else cg.f32x4.max();
      return;
    default:
      throw new Error(`invalid SIMD reduction op: ${re.op}`);
  }
}
function emitScalarReductionOp(cg, re, acc) {
  switch (re.op) {
    case "Add":
      cg.local.get(acc);
      if (re.dtype === "bool") cg.i32.or();
      else dty(cg, re.op, re.dtype).add();
      return;
    case "Mul":
      cg.local.get(acc);
      if (re.dtype === "bool") cg.i32.and();
      else dty(cg, re.op, re.dtype).mul();
      return;
    case "Min":
    case "Max":
      if (isFloatDtype(re.dtype)) {
        cg.local.get(acc);
        if (re.op === "Min") dtyF(cg, re.op, re.dtype).min();
        else dtyF(cg, re.op, re.dtype).max();
      } else if ([
        "int32",
        "uint32",
        "bool"
      ].includes(re.dtype)) {
        const local = cg.local.declare(cg.i32);
        cg.local.tee(local);
        cg.local.get(acc);
        cg.local.get(local);
        cg.local.get(acc);
        if (re.op === "Min") if (re.dtype === "int32") cg.i32.lt_s();
        else cg.i32.lt_u();
        else if (re.dtype === "int32") cg.i32.gt_s();
        else cg.i32.gt_u();
        cg.select();
      } else throw new Error(`invalid reduction min/max over ${re.dtype}`);
      return;
    default:
      throw new Error(`invalid wasm reduction op: ${re.op}`);
  }
}
function isSimdEligible(tunedExp, kernel) {
  if (kernel.size < 4) return false;
  if (kernel.reduction) {
    if (!simdSupportedOps.get(kernel.reduction.dtype)?.has(kernel.reduction.op)) return false;
  }
  const check = (exp3, visited) => {
    if (visited.has(exp3)) return true;
    visited.add(exp3);
    if (!simdSupportedOps.get(exp3.dtype)?.has(exp3.op)) return false;
    if (exp3.op === "GlobalIndex") return true;
    for (const child of exp3.src) if (!check(child, visited)) return false;
    return true;
  };
  return check(tunedExp, /* @__PURE__ */ new Set());
}
function canUseKSimdReduction(exp3, re, pointers) {
  if (re.op !== "Add") return false;
  return exp3.collect((node) => node.op === "GlobalIndex").length === pointers.length && pointers.every((candidate) => candidate.dtype === "float32" && (candidate.strideBytes === 0 || candidate.strideBytes === byteWidth(candidate.dtype)));
}
function emitAlignmentGuard(cg, paramBegin, paramEnd, alignment = 4) {
  cg.local.get(paramEnd);
  cg.local.get(paramBegin);
  cg.i32.sub();
  cg.i32.const(alignment);
  cg.i32.rem_u();
  cg.i32.eqz();
  cg.local.get(paramBegin);
  cg.i32.const(alignment);
  cg.i32.rem_u();
  cg.i32.eqz();
  cg.i32.and();
  cg.if(cg.void);
}
function codegenWasm(kernel) {
  const tune = tuneNullopt(kernel);
  const re = kernel.reduction;
  if (DEBUG >= 3) console.info(`kernel.exp: ${kernel.exp}
tune.exp: ${tune.exp}`);
  const simdEligible = isSimdEligible(tune.exp, kernel);
  const hasIdentityEpilogue = isIdentityEpilogue(tune.epilogue);
  const expStrides = simdEligible ? collectSimdStrides(tune.exp) : /* @__PURE__ */ new Map();
  const reductionHasLaneGather = re && [...expStrides.values()].some((stride) => stride.kind === "gather");
  const useSimd = simdEligible && !reductionHasLaneGather;
  const simdReductionPointerCandidates = useSimd && re ? reductionPointerCandidates(tune.exp, expStrides) : [];
  const reductionPointers = re ? reductionPointerCandidates(tune.exp) : [];
  const useKSimdReduction = simdEligible && re && canUseKSimdReduction(tune.exp, re, reductionPointers);
  const kSimdReductionPointerCandidates = useKSimdReduction && re ? reductionPointers.map((candidate) => ({
    ...candidate,
    stride: candidate.strideBytes === 0 ? {
      kind: "broadcast",
      tileSize: Infinity
    } : {
      kind: "contiguous",
      tileSize: Infinity
    }
  })) : [];
  const canStoreSimdPartials = hasIdentityEpilogue || re !== void 0 && kernel.dtype === re.dtype && byteWidth(kernel.dtype) === 4;
  const simdTilePlan = useSimd && re ? reductionTilePlan(kernel, expStrides, canStoreSimdPartials) : null;
  const kSimdTilePlan = reductionHasLaneGather && useKSimdReduction ? reductionKTilePlan(kernel, expStrides) : null;
  const useRelaxedMadd = hasWasmFeature("relaxed-madd") && re?.op === "Add" && tune.exp.dtype === "float32" && tune.exp.op === "Mul";
  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");
  if (hasSharedArrayBuffer()) cg.memory.pages(0, 65536).shared(true);
  const distinctOps = mapSetUnion(tune.exp.distinctOps(), tune.epilogue?.distinctOps());
  const funcs = {};
  if (distinctOps.has("Sin")) funcs.sin = wasm_sin(cg);
  if (distinctOps.has("Cos")) funcs.cos = wasm_cos(cg);
  if (distinctOps.has("Asin")) funcs.asin = wasm_asin(cg);
  if (distinctOps.has("Atan")) funcs.atan = wasm_atan(cg);
  if (distinctOps.has("Exp") || distinctOps.has("Erf") || distinctOps.has("Erfc")) funcs.exp = wasm_exp(cg);
  if (distinctOps.has("Log")) funcs.log = wasm_log(cg);
  if (distinctOps.has("Erf")) funcs.erf = wasm_erf(cg, funcs.exp);
  if (distinctOps.has("Erfc")) funcs.erfc = wasm_erfc(cg, funcs.exp);
  if (distinctOps.has("Threefry2x32")) funcs.threefry2x32 = wasm_threefry2x32(cg);
  const paramBegin = kernel.nargs + 1;
  const paramEnd = kernel.nargs + 2;
  const kernelFunc = cg.function(rep(kernel.nargs + 3, cg.i32), [], () => {
    const gidx = cg.local.declare(cg.i32);
    cg.local.get(paramBegin);
    cg.local.set(gidx);
    const emitLocalPlusConst = (local, amount) => {
      cg.local.get(local);
      cg.i32.const(amount);
      cg.i32.add();
    };
    const bumpLocal = (local, amount) => {
      emitLocalPlusConst(local, amount);
      cg.local.set(local);
    };
    const setLocalConst = (local, value) => {
      cg.i32.const(value);
      cg.local.set(local);
    };
    const copyLocal = (target, source) => {
      cg.local.get(source);
      cg.local.set(target);
    };
    const setRowBase = (target, rowTileBase, rowOffset, tileSize) => {
      cg.local.get(rowTileBase);
      cg.local.get(rowOffset);
      cg.i32.const(tileSize);
      cg.i32.mul();
      cg.i32.add();
      cg.local.set(target);
    };
    const emitOutputAddress = (index) => {
      cg.local.get(kernel.nargs);
      cg.local.get(index);
      cg.i32.const(byteWidth(kernel.dtype));
      cg.i32.mul();
      cg.i32.add();
    };
    const declareTileGidx = (rowBase, col, rowOffset, colOffset) => {
      const local = cg.local.declare(cg.i32);
      cg.local.get(rowBase);
      if (rowOffset !== 0) {
        cg.i32.const(rowOffset);
        cg.i32.add();
      }
      cg.local.get(col);
      cg.i32.add();
      if (colOffset !== 0) {
        cg.i32.const(colOffset);
        cg.i32.add();
      }
      cg.local.set(local);
      return local;
    };
    const emitLoopWithBreaks = (emitBreaks, emitBody) => {
      cg.loop(cg.void);
      cg.block(cg.void);
      emitBreaks();
      emitBody();
      cg.br(1);
      cg.end();
      cg.end();
    };
    const emitLoopWhileLt = (index, emitBound, emitBody) => emitLoopWithBreaks(() => {
      cg.local.get(index);
      emitBound();
      cg.i32.ge_u();
      cg.br_if(0);
    }, emitBody);
    const emitLoopWhileLtAndConstLt = (index, emitBound, constBound, emitBody) => emitLoopWithBreaks(() => {
      cg.local.get(index);
      emitBound();
      cg.i32.ge_u();
      cg.br_if(0);
      cg.local.get(index);
      cg.i32.const(constBound);
      cg.i32.ge_u();
      cg.br_if(0);
    }, emitBody);
    const emitLoopWhileBlockFits = (index, blockSize, emitBound, constBound, emitBody) => emitLoopWithBreaks(() => {
      emitLocalPlusConst(index, blockSize);
      emitBound();
      cg.i32.gt_u();
      cg.br_if(0);
      emitLocalPlusConst(index, blockSize);
      cg.i32.const(constBound);
      cg.i32.gt_u();
      cg.br_if(0);
    }, emitBody);
    const emitRowTileLoop = (rowOffset, rowTileBase, tileRows, tileSize, emitBody) => emitLoopWithBreaks(() => {
      cg.local.get(rowOffset);
      cg.i32.const(tileRows);
      cg.i32.ge_u();
      cg.br_if(0);
      cg.local.get(rowTileBase);
      cg.local.get(rowOffset);
      cg.i32.const(tileSize);
      cg.i32.mul();
      cg.i32.add();
      cg.local.get(paramEnd);
      cg.i32.ge_u();
      cg.br_if(0);
    }, emitBody);
    const emitLoopWhileLocalLt = (index, bound, emitBody) => emitLoopWhileLt(index, () => cg.local.get(bound), emitBody);
    const emitLoopWhileConstLt = (index, bound, emitBody) => emitLoopWhileLt(index, () => cg.i32.const(bound), emitBody);
    const emitSimdExpWithAccumulator = (ctx, pointerMap, pointerValueCache, acc) => {
      if (useRelaxedMadd) {
        translateExpSimd(cg, funcs, tune.exp.src[0], ctx, expStrides, pointerMap, pointerValueCache);
        translateExpSimd(cg, funcs, tune.exp.src[1], ctx, expStrides, pointerMap, pointerValueCache);
        cg.local.get(acc);
        cg.f32x4.relaxed_madd();
        return true;
      }
      translateExpSimd(cg, funcs, tune.exp, ctx, expStrides, pointerMap, pointerValueCache);
      cg.local.get(acc);
      return false;
    };
    const emitSimdReductionForGidxs = (gidxs, pointerMaps, uniquePointers, ridxStart, ridxEnd) => {
      if (!re) throw new Error("internal: missing reduction");
      const reIsInt = kernel.exp.dtype === "int32" || kernel.exp.dtype === "uint32";
      const vecAccs = gidxs.map(() => cg.local.declare(reIsInt ? cg.i32x4 : cg.f32x4));
      const initializeIdentityAccumulators = () => {
        for (const acc of vecAccs) {
          if (reIsInt) {
            cg.i32.const(re.identity);
            cg.i32x4.splat();
          } else {
            cg.f32.const(re.identity);
            cg.f32x4.splat();
          }
          cg.local.set(acc);
        }
      };
      const loadPartialAccumulators = () => {
        for (let i = 0; i < gidxs.length; i++) {
          emitOutputAddress(gidxs[i]);
          if (reIsInt) cg.i32x4.load(4);
          else cg.f32x4.load(4);
          cg.local.set(vecAccs[i]);
        }
      };
      if (ridxStart === void 0) initializeIdentityAccumulators();
      else {
        cg.local.get(ridxStart);
        cg.i32.eqz();
        cg.if(cg.void);
        initializeIdentityAccumulators();
        cg.else();
        loadPartialAccumulators();
        cg.end();
      }
      const ridx = cg.local.declare(cg.i32);
      const emitReductionStep = () => {
        const pointerValueCache = /* @__PURE__ */ new Map();
        for (let i = 0; i < gidxs.length; i++) {
          emitSimdReductionOp(cg, re, reIsInt, emitSimdExpWithAccumulator({
            gidx: gidxs[i],
            ridx
          }, pointerMaps[i], pointerValueCache, vecAccs[i]));
          cg.local.set(vecAccs[i]);
        }
        incrementReductionPointers(cg, uniquePointers);
      };
      if (ridxStart === void 0) setLocalConst(ridx, 0);
      else copyLocal(ridx, ridxStart);
      const emitReductionLoopBody = () => {
        emitReductionStep();
        bumpLocal(ridx, 1);
      };
      if (ridxEnd === void 0) emitLoopWhileConstLt(ridx, re.size, emitReductionLoopBody);
      else emitLoopWhileLocalLt(ridx, ridxEnd, emitReductionLoopBody);
      const storeRawAccumulators = () => {
        for (let i = 0; i < gidxs.length; i++) {
          emitOutputAddress(gidxs[i]);
          cg.local.get(vecAccs[i]);
          cg.v128.store(4);
        }
      };
      if (hasIdentityEpilogue) {
        storeRawAccumulators();
        return;
      }
      const storeEpilogueAccumulators = () => {
        const laneGidx = cg.local.declare(cg.i32);
        const laneAcc = cg.local.declare(reIsInt ? cg.i32 : cg.f32);
        for (let i = 0; i < gidxs.length; i++) for (let lane = 0; lane < 4; lane++) {
          cg.local.get(kernel.nargs);
          cg.local.get(gidxs[i]);
          if (lane > 0) {
            cg.i32.const(lane);
            cg.i32.add();
          }
          cg.local.tee(laneGidx);
          cg.i32.const(byteWidth(kernel.dtype));
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(vecAccs[i]);
          if (reIsInt) cg.i32x4.extract_lane(lane);
          else cg.f32x4.extract_lane(lane);
          cg.local.set(laneAcc);
          translateExp(cg, funcs, tune.epilogue, {
            acc: laneAcc,
            gidx: laneGidx
          });
          dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
        }
      };
      if (ridxEnd !== void 0) {
        cg.local.get(ridxEnd);
        cg.i32.const(re.size);
        cg.i32.lt_u();
        cg.if(cg.void);
        storeRawAccumulators();
        cg.else();
        storeEpilogueAccumulators();
        cg.end();
      } else storeEpilogueAccumulators();
    };
    const initializePointerMaps = (groups, candidates, keyFor, ctxFor, ridxOffset) => {
      const pointerMaps = groups.map(() => /* @__PURE__ */ new Map());
      const sharedPointers = /* @__PURE__ */ new Map();
      const uniquePointers = [];
      for (let i = 0; i < groups.length; i++) {
        const group = groups[i];
        for (const candidate of candidates) {
          const key = keyFor(candidate, group, i);
          let plan = sharedPointers.get(key);
          if (!plan) {
            plan = initializeReductionPointer(cg, funcs, candidate, ctxFor(group), key, ridxOffset);
            sharedPointers.set(key, plan);
            uniquePointers.push(plan);
          }
          pointerMaps[i].set(candidate.exp, plan);
        }
      }
      return {
        pointerMaps,
        uniquePointers
      };
    };
    const emitElementwiseSimdStep = () => {
      emitOutputAddress(gidx);
      translateExpSimd(cg, funcs, tune.exp, { gidx }, expStrides);
      cg.v128.store(4);
      bumpLocal(gidx, 4);
    };
    const emitKSimdReductionForGroups = (groups, pointerMaps, uniquePointers) => {
      if (!re) throw new Error("internal: missing reduction");
      if (!kSimdTilePlan) throw new Error("internal: missing K SIMD plan");
      const vecAccs = groups.map(() => cg.local.declare(cg.f32x4));
      for (const acc of vecAccs) {
        cg.f32.const(re.identity);
        cg.f32x4.splat();
        cg.local.set(acc);
      }
      const ridx = cg.local.declare(cg.i32);
      setLocalConst(ridx, 0);
      emitLoopWhileConstLt(ridx, re.size, () => {
        for (let u = 0; u < kSimdTilePlan.kUnroll; u++) {
          const pointerValueCache = /* @__PURE__ */ new Map();
          for (let i = 0; i < groups.length; i++) {
            if (!emitSimdExpWithAccumulator({
              gidx: groups[i].gidx,
              ridx
            }, pointerMaps[i], pointerValueCache, vecAccs[i])) cg.f32x4.add();
            cg.local.set(vecAccs[i]);
          }
          incrementReductionPointers(cg, uniquePointers, 4);
        }
        bumpLocal(ridx, 4 * kSimdTilePlan.kUnroll);
      });
      for (let i = 0; i < groups.length; i++) {
        const acc = cg.local.declare(cg.f32);
        for (let lane = 0; lane < 4; lane++) {
          cg.local.get(vecAccs[i]);
          cg.f32x4.extract_lane(lane);
          if (lane > 0) cg.f32.add();
        }
        cg.local.set(acc);
        emitOutputAddress(groups[i].gidx);
        translateExp(cg, funcs, tune.epilogue, {
          acc,
          gidx: groups[i].gidx
        });
        dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
      }
    };
    const emitKSimdReductionLoop = (plan) => {
      const colTile = cg.local.declare(cg.i32);
      const col = cg.local.declare(cg.i32);
      const rowTileBase = cg.local.declare(cg.i32);
      const rowOffset = cg.local.declare(cg.i32);
      const rowBase = cg.local.declare(cg.i32);
      const emitBlock = () => {
        const groups = [];
        for (let row = 0; row < plan.microRows; row++) for (let colOffset = 0; colOffset < plan.microCols; colOffset++) groups.push({
          gidx: declareTileGidx(rowBase, col, row * plan.tileSize, colOffset),
          row,
          col: colOffset
        });
        if (!kSimdTilePlan) throw new Error("internal: missing K SIMD plan");
        const { pointerMaps, uniquePointers } = initializePointerMaps(groups, kSimdReductionPointerCandidates, (candidate, group, i) => kReductionPointerShareKey(candidate, expStrides.get(candidate.exp) ?? { kind: "gather" }, kSimdTilePlan.tileSize, group.row, group.col, i), (group) => ({ gidx: group.gidx }));
        emitKSimdReductionForGroups(groups, pointerMaps, uniquePointers);
      };
      emitLoopWhileLocalLt(gidx, paramEnd, () => {
        copyLocal(rowTileBase, gidx);
        setLocalConst(colTile, 0);
        emitLoopWhileConstLt(colTile, plan.tileSize, () => {
          setLocalConst(rowOffset, 0);
          emitRowTileLoop(rowOffset, rowTileBase, plan.tileRows, plan.tileSize, () => {
            copyLocal(col, colTile);
            emitLoopWhileLtAndConstLt(col, () => emitLocalPlusConst(colTile, plan.tileCols), plan.tileSize, () => {
              setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
              emitBlock();
              bumpLocal(col, plan.microCols);
            });
            bumpLocal(rowOffset, plan.microRows);
          });
          bumpLocal(colTile, plan.tileCols);
        });
        bumpLocal(gidx, plan.tileRows * plan.tileSize);
      });
    };
    const emitTiledSimdReductionLoop = (plan) => {
      const colTile = cg.local.declare(cg.i32);
      const col = cg.local.declare(cg.i32);
      const kTile = cg.local.declare(cg.i32);
      const tileEnd = cg.local.declare(cg.i32);
      const rowTileBase = cg.local.declare(cg.i32);
      const rowOffset = cg.local.declare(cg.i32);
      const rowBase = cg.local.declare(cg.i32);
      const emitRowBlock = (microVectors) => {
        const groups = [];
        for (let row = 0; row < plan.microRows; row++) for (let vector = 0; vector < microVectors; vector++) groups.push({
          gidx: declareTileGidx(rowBase, col, row * plan.tileSize, vector * 4),
          row,
          vector
        });
        const { pointerMaps, uniquePointers } = initializePointerMaps(groups, simdReductionPointerCandidates, (candidate, group, i) => pointerShareKey(candidate, group.row, group.vector, i), (group) => ({ gidx: group.gidx }), kTile);
        emitSimdReductionForGidxs(groups.map((group) => group.gidx), pointerMaps, uniquePointers, kTile, tileEnd);
      };
      emitLoopWhileLocalLt(gidx, paramEnd, () => {
        copyLocal(rowTileBase, gidx);
        setLocalConst(colTile, 0);
        emitLoopWhileConstLt(colTile, plan.tileSize, () => {
          setLocalConst(kTile, 0);
          emitLoopWhileConstLt(kTile, re.size, () => {
            emitLocalPlusConst(kTile, plan.tileK);
            cg.local.set(tileEnd);
            setLocalConst(rowOffset, 0);
            emitRowTileLoop(rowOffset, rowTileBase, plan.tileRows, plan.tileSize, () => {
              copyLocal(col, colTile);
              emitLoopWhileBlockFits(col, plan.microVectors * 4, () => emitLocalPlusConst(colTile, plan.tileVectors * 4), plan.tileSize, () => {
                setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
                emitRowBlock(plan.microVectors);
                bumpLocal(col, plan.microVectors * 4);
              });
              emitLoopWhileLtAndConstLt(col, () => emitLocalPlusConst(colTile, plan.tileVectors * 4), plan.tileSize, () => {
                setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
                emitRowBlock(1);
                bumpLocal(col, 4);
              });
              bumpLocal(rowOffset, plan.microRows);
            });
            bumpLocal(kTile, plan.tileK);
          });
          bumpLocal(colTile, plan.tileVectors * 4);
        });
        bumpLocal(gidx, plan.tileRows * plan.tileSize);
      });
    };
    const emitGuardedFastPath = (alignment, emit) => {
      emitAlignmentGuard(cg, paramBegin, paramEnd, alignment);
      emit();
      cg.return();
      cg.end();
    };
    if (kSimdTilePlan) emitGuardedFastPath(kSimdTilePlan.microRows * kSimdTilePlan.tileSize, () => emitKSimdReductionLoop(kSimdTilePlan));
    if (useSimd) {
      if (simdTilePlan) emitGuardedFastPath(simdTilePlan.microRows * simdTilePlan.tileSize, () => emitTiledSimdReductionLoop(simdTilePlan));
      if (!re) emitGuardedFastPath(4, () => {
        emitLoopWhileLocalLt(gidx, paramEnd, emitElementwiseSimdStep);
      });
    }
    emitLoopWhileLocalLt(gidx, paramEnd, () => {
      emitOutputAddress(gidx);
      if (re) {
        const acc = cg.local.declare(dty(cg, null, kernel.exp.dtype));
        dty(cg, null, kernel.exp.dtype).const(re.identity);
        cg.local.set(acc);
        const ridx = cg.local.declare(cg.i32);
        const emitReductionStep = () => {
          translateExp(cg, funcs, tune.exp, {
            gidx,
            ridx
          });
          emitScalarReductionOp(cg, re, acc);
          cg.local.set(acc);
        };
        setLocalConst(ridx, 0);
        emitLoopWhileConstLt(ridx, re.size, () => {
          emitReductionStep();
          bumpLocal(ridx, 1);
        });
        translateExp(cg, funcs, tune.epilogue, {
          acc,
          gidx
        });
      } else translateExp(cg, funcs, tune.exp, { gidx });
      dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
      bumpLocal(gidx, 1);
    });
  });
  cg.export(kernelFunc, "kernel");
  const tiledPlan = simdTilePlan ?? kSimdTilePlan;
  return {
    bytes: cg.finish(),
    workSize: kernel.size,
    chunkAlignment: tiledPlan ? tiledPlan.tileRows * tiledPlan.tileSize : 16,
    minWorkPerWorker: tiledPlan && kernel.size / tiledPlan.tileSize >= 1024 ? tiledPlan.tileSize * 32 : void 0
  };
}
function defaultDevice(device) {
  if (device !== void 0) if (initializedBackends.has(device)) defaultBackend = device;
  else throw new Error(`Backend not initialized: ${device}`);
  return defaultBackend;
}
async function init(...devicesToInit) {
  if (devicesToInit.length === 0) devicesToInit = devices;
  const promises = [];
  for (const device of new Set(devicesToInit)) if (!initializedBackends.has(device)) promises.push((async () => {
    const backend = await createBackend(device);
    if (backend) initializedBackends.set(device, backend);
  })());
  await Promise.all(promises);
  return Array.from(initializedBackends.keys());
}
async function createBackend(device) {
  if (device === "cpu") return new CpuBackend();
  else if (device === "wasm") {
    if (typeof WebAssembly === "undefined") return null;
    return new WasmBackend();
  } else if (device === "webgpu") {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const { WebGPUBackend: WebGPUBackend2 } = await Promise.resolve().then(() => (init_webgpu_Dluq5ZOf(), webgpu_Dluq5ZOf_exports));
    const importantLimits = [
      "maxBufferSize",
      "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX",
      "maxComputeWorkgroupSizeY",
      "maxComputeWorkgroupSizeZ",
      "maxComputeWorkgroupStorageSize",
      "maxComputeWorkgroupsPerDimension",
      "maxStorageBufferBindingSize",
      "maxStorageBuffersPerShaderStage",
      "maxStorageTexturesPerShaderStage"
    ];
    const requestedFeatures = ["shader-f16", "timestamp-query"];
    try {
      return new WebGPUBackend2(await adapter.requestDevice({
        requiredLimits: Object.fromEntries(importantLimits.map((limit) => [limit, adapter.limits[limit]])),
        requiredFeatures: requestedFeatures.filter((feature) => adapter.features.has(feature))
      }));
    } catch (error) {
      console.error("Unexpected error requesting WebGPU device:", error);
      return null;
    }
  } else if (device === "webgl") {
    if (typeof WebGL2RenderingContext === "undefined") return null;
    const gl = new OffscreenCanvas(0, 0).getContext("webgl2", {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      depth: false,
      stencil: false,
      failIfMajorPerformanceCaveat: true
    });
    if (!gl) return null;
    if (!gl.getExtension("EXT_color_buffer_float")) return null;
    const { WebGLBackend: WebGLBackend2 } = await Promise.resolve().then(() => (init_webgl_gbVkTlDB(), webgl_gbVkTlDB_exports));
    return new WebGLBackend2(gl);
  } else throw new Error(`Backend not found: ${device}`);
}
function getBackend(device) {
  device = device ?? defaultBackend;
  const backend = initializedBackends.get(device);
  if (!backend) throw new Error(`${device} backend not ready, call init() first`);
  return backend;
}
var PPrint, DEBUG, _stagingbuf, FpHash, DType, byteWidth, isFloatDtype, AluExp, AluGroup, AluVar, Kernel, Reduction, Routine, Routines, ScatterOp, jstr, View, ShapeTracker, TuneDims, CpuBackend, traceEnabled, flushCallbacks, ALLOCATION_ALIGNMENT, WASM_PAGE_SIZE, MAX_MEMORY32_BYTES, WasmAllocator, featureProbes, featureSupportCache, MIN_ELEMS_PER_THREAD, WORKER_SOURCE, WasmWorkerPool, TILED_SIMD_ROWS, TILED_SIMD_COLUMNS, TILED_SIMD_K, TILED_SIMD_MICRO_ROWS, TILED_SIMD_MICRO_VECTORS, K_SIMD_MICRO_ROWS, K_SIMD_MICRO_COLS, K_SIMD_MAX_UNROLL, simdSupportedOps, magicModuleHeader, moduleVersion, Function_, Memory, CodeGenerator, Local, I32, F32, F64, V128, I32x4, F32x4, compiledProgramCache, WasmBackend, devices, initializedBackends, defaultBackend, Executable, SlotError, UnsupportedOpError, UnsupportedRoutineError;
var init_backend_D_Uwkp6d = __esm({
  "node_modules/@jax-js/jax/dist/backend-D_Uwkp6d.js"() {
    PPrint = class PPrint2 {
      indents;
      lines;
      constructor(indents, lines) {
        this.indents = indents;
        this.lines = lines;
      }
      /** Add a fixed amount of indentation to each line. */
      indent(spaces) {
        return new PPrint2(this.indents.map((i) => i + spaces), this.lines);
      }
      /** Concatenate pretty-printed expressions with newlines. */
      concat(...items) {
        return new PPrint2((this.indents ?? []).concat(...items.map((i) => i.indents)), (this.lines ?? []).concat(...items.map((i) => i.lines)));
      }
      /** Stack one block to the right of another one, sharing 1 common line. */
      stack(other) {
        if (!other.lines.length) return this;
        if (!this.lines.length) return other;
        const indent = this.indents[this.indents.length - 1];
        const s = this.lines[this.lines.length - 1];
        const indentedBlock = other.indent(indent + s.length);
        return new PPrint2(this.indents.concat(indentedBlock.indents.slice(1)), this.lines.slice(0, -1).concat(s + " ".repeat(other.indents[0]) + other.lines[0], ...indentedBlock.lines.slice(1)));
      }
      /** Combine this block of lines into a formatted string. */
      toString() {
        return this.lines.map((line, i) => " ".repeat(this.indents[i]) + line).join("\n");
      }
      static pp(s) {
        const lines = s.toString().split("\n");
        return new PPrint2(Array(lines.length).fill(0), lines);
      }
    };
    DEBUG = 0;
    _stagingbuf = /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(8));
    FpHash = class FpHash2 {
      value = 8773157n;
      #update(x) {
        const base = 873192869n;
        const modulus = 3189051996290219n;
        this.value = (this.value * base + x) % modulus;
      }
      update(x) {
        if (typeof x === "string") {
          this.#update(BigInt(x.length));
          for (let i = 0; i < x.length; i++) this.#update(BigInt(199 + x.charCodeAt(i)));
        } else if (typeof x === "number") if (Number.isInteger(x)) this.#update(68265653n ^ BigInt(x));
        else {
          _stagingbuf.setFloat64(0, x, true);
          this.#update(_stagingbuf.getBigUint64(0, true));
        }
        else if (typeof x === "boolean") this.#update(x ? 69069841n : 63640693n);
        else if (typeof x === "bigint") this.#update(x ^ 71657401n);
        else if (x === null) this.#update(37832657n);
        else if (x === void 0) this.#update(18145117n);
        else x.hash(this);
        return this;
      }
      static hash(...values) {
        const h = new FpHash2();
        for (const x of values) h.update(x);
        return h.value;
      }
    };
    DType = /* @__PURE__ */ (function(DType2) {
      DType2["Float32"] = "float32";
      DType2["Int32"] = "int32";
      DType2["Uint32"] = "uint32";
      DType2["Bool"] = "bool";
      DType2["Float16"] = "float16";
      DType2["Float64"] = "float64";
      return DType2;
    })({});
    byteWidth = (dtype) => {
      switch (dtype) {
        case "float32":
        case "int32":
        case "uint32":
        case "bool":
          return 4;
        case "float16":
          return 2;
        case "float64":
          return 8;
        default:
          throw new TypeError(`Unknown dtype: ${dtype}`);
      }
    };
    isFloatDtype = (dtype) => dtype === "float32" || dtype === "float16" || dtype === "float64";
    AluExp = class AluExp2 {
      op;
      dtype;
      src;
      arg;
      #hash;
      #simplified;
      #range;
      constructor(op, dtype, src, arg = void 0) {
        this.op = op;
        this.dtype = dtype;
        this.src = src;
        this.arg = arg;
        if (AluGroup.RequiredFloat.has(op) && !isFloatDtype(dtype)) throw new TypeError(`Unsupported dtype for ${op}: ${dtype}`);
        switch (op) {
          case "Bitcast":
            if (dtype === "bool" || src[0].dtype === "bool" || byteWidth(dtype) !== byteWidth(src[0].dtype)) throw new TypeError(`Bitcast from ${src[0].dtype} -> ${dtype}`);
            break;
          case "Threefry2x32":
            if (dtype !== "uint32" || src.some((x) => x.dtype !== "uint32")) throw new TypeError("Threefry2x32 requires uint32 types");
            break;
          case "BitCombine":
            if (src[0].dtype !== src[1].dtype || isFloatDtype(src[0].dtype)) throw new TypeError(`BitCombine[${arg}] requires matching integral dtype, got ${src[0].dtype} and ${src[1].dtype}`);
            break;
          case "BitShift":
            if (src[0].dtype === "bool" || src[1].dtype === "bool" || isFloatDtype(src[0].dtype) || isFloatDtype(src[1].dtype)) throw new TypeError(`BitShift[${arg}] requires two integral, non-bool dtypes, got ${src[0].dtype} and ${src[1].dtype}`);
            break;
          case "BitInvert":
            if (isFloatDtype(src[0].dtype)) throw new TypeError(`BitInvert requires an integral dtype, got ${src[0].dtype}`);
            break;
        }
      }
      static add(a, b) {
        return new AluExp2("Add", a.dtype, [a, b]);
      }
      static sub(a, b) {
        return new AluExp2("Sub", a.dtype, [a, b]);
      }
      static mul(a, b) {
        return new AluExp2("Mul", a.dtype, [a, b]);
      }
      static idiv(a, b) {
        return new AluExp2("Idiv", a.dtype, [a, b]);
      }
      static mod(a, b) {
        return new AluExp2("Mod", a.dtype, [a, b]);
      }
      static min(a, b) {
        return new AluExp2("Min", a.dtype, [a, b]);
      }
      static max(a, b) {
        return new AluExp2("Max", a.dtype, [a, b]);
      }
      static sin(a) {
        return new AluExp2("Sin", a.dtype, [a]);
      }
      static cos(a) {
        return new AluExp2("Cos", a.dtype, [a]);
      }
      static asin(a) {
        return new AluExp2("Asin", a.dtype, [a]);
      }
      static atan(a) {
        return new AluExp2("Atan", a.dtype, [a]);
      }
      static exp(a) {
        return new AluExp2("Exp", a.dtype, [a]);
      }
      static log(a) {
        return new AluExp2("Log", a.dtype, [a]);
      }
      static erf(a) {
        return new AluExp2("Erf", a.dtype, [a]);
      }
      static erfc(a) {
        return new AluExp2("Erfc", a.dtype, [a]);
      }
      static sqrt(a) {
        return new AluExp2("Sqrt", a.dtype, [a]);
      }
      static floor(a) {
        if (!isFloatDtype(a.dtype)) return a;
        return new AluExp2("Floor", a.dtype, [a]);
      }
      static ceil(a) {
        if (!isFloatDtype(a.dtype)) return a;
        return new AluExp2("Ceil", a.dtype, [a]);
      }
      static reciprocal(a) {
        return new AluExp2("Reciprocal", a.dtype, [a]);
      }
      static cast(dtype, a) {
        if (a.dtype === dtype) return a;
        return new AluExp2("Cast", dtype, [a]);
      }
      static bitcast(dtype, a) {
        if (a.dtype === dtype) return a;
        return new AluExp2("Bitcast", dtype, [a]);
      }
      static threefry2x32(k0, k1, c0, c1, mode = "xor") {
        return new AluExp2("Threefry2x32", "uint32", [
          k0,
          k1,
          c0,
          c1
        ], mode);
      }
      static bitCombine(a, b, mode) {
        return new AluExp2("BitCombine", a.dtype, [a, b], mode);
      }
      static bitShift(a, b, mode) {
        return new AluExp2("BitShift", a.dtype, [a, b], mode);
      }
      static cmplt(a, b) {
        return new AluExp2("Cmplt", "bool", [a, b]);
      }
      static cmpne(a, b) {
        return new AluExp2("Cmpne", "bool", [a, b]);
      }
      static where(cond, a, b) {
        return new AluExp2("Where", a.dtype, [
          cond,
          a,
          b
        ]);
      }
      static const(dtype, value) {
        if (dtype === "bool") value = Number(Boolean(value));
        else if (dtype === "int32") value = Math.trunc(value) | 0;
        else if (dtype === "uint32") value = Math.trunc(value) >>> 0;
        if (typeof value !== "number") throw new TypeError(`Expected a number for constant, got ${typeof value}: ${value}`);
        return new AluExp2("Const", dtype, [], value);
      }
      static special(dtype, name, n) {
        return new AluExp2("Special", dtype, [], [name, n]);
      }
      static variable(dtype, name) {
        return new AluExp2("Variable", dtype, [], name);
      }
      static globalIndex(dtype, gid, len, bufidx) {
        return new AluExp2("GlobalIndex", dtype, [bufidx], [gid, len]);
      }
      static globalView(dtype, gid, st, indices2) {
        return new AluExp2("GlobalView", dtype, indices2, [gid, st]);
      }
      static f32(value) {
        return AluExp2.const("float32", value);
      }
      static i32(value) {
        return AluExp2.const("int32", value);
      }
      static u32(value) {
        return AluExp2.const("uint32", value);
      }
      static bool(value) {
        return AluExp2.const("bool", Number(value));
      }
      static f16(value) {
        return AluExp2.const("float16", value);
      }
      static f64(value) {
        return AluExp2.const("float64", value);
      }
      not() {
        if (this.dtype !== "bool") throw new Error("not() can only be called on boolean expressions");
        return AluExp2.cmpne(this, AluExp2.const("bool", true));
      }
      /** Compute a reasonable expression hash with low collision rate. */
      getHash() {
        if (this.#hash !== void 0) return this.#hash;
        const hasher = new FpHash();
        hasher.update(this.op);
        hasher.update(this.dtype);
        if (this.op === "Const") hasher.update(this.arg);
        else hasher.update(JSON.stringify(this.arg));
        hasher.update(this.src.length);
        for (const s of this.src) hasher.update(s);
        this.#hash = hasher.value;
        return this.#hash;
      }
      hash(state) {
        state.update(this.getHash());
      }
      /** Substitute variables in this AluExp to values. */
      substitute(variables) {
        return this.rewrite((exp3) => {
          if (exp3.op === "Variable" && Object.hasOwn(variables, exp3.arg)) {
            if (exp3.dtype !== variables[exp3.arg].dtype) throw new Error(`Type mismatch: ${exp3.dtype} vs ${variables[exp3.arg].dtype}`);
            return variables[exp3.arg];
          }
        });
      }
      /** Reindex gid values in this expression as needed. */
      reindexGids(newGids) {
        return this.rewrite((exp3) => {
          if (exp3.op === "GlobalIndex") {
            const [gid, len] = exp3.arg;
            const newGid = newGids[gid];
            if (newGid !== gid) return AluExp2.globalIndex(exp3.dtype, newGid, len, exp3.src[0]);
          } else if (exp3.op === "GlobalView") {
            const gid = exp3.arg[0];
            const newGid = newGids[gid];
            if (newGid !== gid) return AluExp2.globalView(exp3.dtype, newGid, exp3.arg[1], exp3.src);
          }
        });
      }
      #computeRange() {
        if (this.#range !== void 0) return this.#range;
        const src = this.src;
        const minMax4 = (f) => {
          const [r1, r2] = [src[0].#computeRange(), src[1].#computeRange()];
          const values = [
            f(r1[0], r2[0]),
            f(r1[0], r2[1]),
            f(r1[1], r2[0]),
            f(r1[1], r2[1])
          ];
          return [Math.min(...values), Math.max(...values)];
        };
        let ret;
        switch (this.op) {
          case "Add":
            ret = [src[0].min + src[1].min, src[0].max + src[1].max];
            break;
          case "Sub":
            ret = [src[0].min - src[1].max, src[0].max - src[1].min];
            break;
          case "Mul":
            ret = minMax4((a, b) => a * b);
            break;
          case "Idiv":
            ret = minMax4((a, b) => Math.trunc(a / b));
            break;
          case "Mod": {
            let divisorRange = src[1].#computeRange();
            if (divisorRange[0] <= 0 && divisorRange[1] >= 0) divisorRange = [0, Math.max(-divisorRange[0], divisorRange[1])];
            if (divisorRange[1] < 0) divisorRange = [-divisorRange[1], -divisorRange[0]];
            const maxDivisor = isFloatDtype(this.dtype) ? divisorRange[1] : divisorRange[1] - 1;
            ret = [clamp(src[0].min, -maxDivisor, 0), clamp(src[0].max, 0, maxDivisor)];
            break;
          }
          case "Min":
            ret = [Math.min(src[0].min, src[1].min), Math.min(src[0].max, src[1].max)];
            break;
          case "Max":
            ret = [Math.max(src[0].min, src[1].min), Math.max(src[0].max, src[1].max)];
            break;
          case "Sin":
            ret = [-1, 1];
            break;
          case "Cos":
            ret = [-1, 1];
            break;
          case "Asin":
            ret = [-Math.PI / 2, Math.PI / 2];
            break;
          case "Atan":
            ret = [-Math.PI / 2, Math.PI / 2];
            break;
          case "Exp":
            ret = [Math.exp(src[0].min), Math.exp(src[0].max)];
            break;
          case "Log":
            ret = [Math.log(src[0].min), Math.log(src[0].max)];
            break;
          case "Erf":
            ret = [erf(src[0].min), erf(src[0].max)];
            break;
          case "Erfc":
            ret = [erfc(src[0].max), erfc(src[0].min)];
            break;
          case "Sqrt":
            ret = [Math.sqrt(src[0].min), Math.sqrt(src[0].max)];
            break;
          case "Floor":
            ret = [Math.floor(src[0].min), Math.floor(src[0].max)];
            break;
          case "Ceil":
            ret = [Math.ceil(src[0].min), Math.ceil(src[0].max)];
            break;
          case "Reciprocal":
            if (src[0].min <= 0 && src[0].max >= 0) return [-Infinity, Infinity];
            ret = [1 / src[0].max, 1 / src[0].min];
            break;
          case "Cast": {
            const wasFloat = isFloatDtype(src[0].dtype);
            const bounded = Number.isFinite(src[0].min) && Number.isFinite(src[0].max);
            if (this.dtype === "bool") {
              const canBeZero = src[0].min <= 0 && src[0].max >= 0;
              ret = src[0].min === 0 && src[0].max === 0 ? [0, 0] : canBeZero ? [0, 1] : [1, 1];
            } else if (this.dtype === "int32") {
              const a = wasFloat ? clamp(src[0].min, -2147483648, 2147483647) | 0 : src[0].min | 0;
              const b = wasFloat ? clamp(src[0].max, -2147483648, 2147483647) | 0 : src[0].max | 0;
              ret = bounded && a <= b ? [a, b] : [-Infinity, Infinity];
            } else if (this.dtype === "uint32") {
              const a = wasFloat ? clamp(src[0].min, 0, 4294967295) >>> 0 : src[0].min >>> 0;
              const b = wasFloat ? clamp(src[0].max, 0, 4294967295) >>> 0 : src[0].max >>> 0;
              ret = bounded && a <= b ? [a, b] : [0, Infinity];
            } else ret = [src[0].min, src[0].max];
            break;
          }
          case "Cmplt":
            ret = [0, 1];
            break;
          case "Cmpne":
            ret = [0, 1];
            break;
          case "Where":
            ret = [Math.min(src[1].min, src[2].min), Math.max(src[1].max, src[2].max)];
            break;
          case "Const":
            ret = [this.arg, this.arg];
            break;
          case "Special":
            ret = [0, this.arg[1] - 1];
            break;
          default:
            ret = [-Infinity, Infinity];
        }
        if (isNaN(ret[0]) || isNaN(ret[1])) ret = [-Infinity, Infinity];
        if (this.dtype === "bool") {
          ret[0] = clamp(ret[0], 0, 1);
          ret[1] = clamp(ret[1], 0, 1);
        }
        if (this.dtype === "uint32") ret[0] = Math.max(0, ret[0]);
        this.#range = ret;
        return ret;
      }
      get min() {
        return this.#computeRange()[0];
      }
      get max() {
        return this.#computeRange()[1];
      }
      /** Largest known integer that divides self. */
      constFactor() {
        if (this.op === "Const") return Math.abs(this.arg);
        if (this.op === "Add") return gcd(this.src[0].constFactor(), this.src[1].constFactor());
        if (this.op === "Mul") {
          if (this.src[0].op === "Const") return Math.abs(this.src[0].arg);
          if (this.src[1].op === "Const") return Math.abs(this.src[1].arg);
        }
        return 1;
      }
      /**
      * Checks if divisible by an integer v and returns the quotient if it is, or
      * `null` if it's not divisible.
      */
      divides(v) {
        if (v === 1) return this;
        if (this.op === "Const" && this.arg % v === 0) return AluExp2.const(this.dtype, this.arg / v);
        if (this.op === "Add") {
          const a = this.src[0].divides(v);
          if (a !== null) {
            const b = this.src[1].divides(v);
            if (b !== null) return AluExp2.add(a, b);
          }
        }
        if (this.op === "Mul") {
          const a = this.src[0].divides(v);
          if (a !== null) return AluExp2.mul(a, this.src[1]);
          const b = this.src[1].divides(v);
          if (b !== null) return AluExp2.mul(this.src[0], b);
        }
        return null;
      }
      #isConstInt() {
        return this.op === "Const" && (this.dtype === "int32" || this.dtype === "uint32");
      }
      /**
      * Get all expressions by deeply matching an operation.
      *
      * For example: `((2+(3*5))+4).splitOp(+) -> [2,(3*5),4]`.
      */
      *splitOp(sep) {
        if (this.op === sep) for (const src of this.src) yield* src.splitOp(sep);
        else yield this;
      }
      /**
      * Simplify the expression by replacing any known patterns and deduping
      * identical subexpressions.
      */
      simplify(cache = /* @__PURE__ */ new Map()) {
        if (this.#simplified !== void 0) return this.#simplified;
        const hash = this.getHash();
        const prevCachedValue = cache.get(hash);
        if (prevCachedValue !== void 0) return this.#simplified = prevCachedValue;
        const simplified = this.#simplifyInner(cache);
        const simplifiedHash = simplified.getHash();
        const prevSimplified = cache.get(simplifiedHash);
        if (prevSimplified !== void 0) {
          cache.set(hash, prevSimplified);
          this.#simplified = prevSimplified;
          return prevSimplified;
        } else {
          cache.set(hash, simplified);
          cache.set(simplifiedHash, simplified);
          this.#simplified = simplified;
          return simplified;
        }
      }
      #simplifyInner(cache) {
        const src = this.src.map((x) => x.simplify(cache));
        const { op } = this;
        if (src.every((x) => x.op === "Const") && !AluGroup.Variable.has(op)) {
          const newExp = new AluExp2(op, this.dtype, src, this.arg);
          return AluExp2.const(this.dtype, newExp.evaluate({}));
        }
        if (op !== "Const" && this.min === this.max) return AluExp2.const(this.dtype, this.min);
        if (AluGroup.Binary.has(op)) for (let i = 0; i < 2; i++) {
          if (src[i].op !== "Const") continue;
          const x = src[i].arg;
          if (op === "Add" && x === 0) return src[1 - i];
          if (op === "Sub" && i === 1 && x === 0) return src[1 - i];
          if (op === "Mul" && x === 1) return src[1 - i];
          if (op === "Mul" && x === 0) return AluExp2.const(this.dtype, 0);
          if (op === "Idiv" && i === 1 && x === 1 && !isFloatDtype(this.dtype)) return src[1 - i];
          if (op === "Cmpne" && src[i].dtype === "bool" && x === 0) return src[1 - i];
        }
        if ((op === "Add" || op === "Sub") && src[1].op === "Mul") {
          const [a, b] = src[1].src;
          const opNeg = op === "Add" ? "Sub" : "Add";
          if (a.op === "Const" && a.arg === -1) return new AluExp2(opNeg, this.dtype, [src[0], b]);
          else if (b.op === "Const" && b.arg === -1) return new AluExp2(opNeg, this.dtype, [src[0], a]);
        }
        if (op === "Where" && src.slice(1).every((s, i) => s.op === "Const" && s.arg === 1 - i)) return AluExp2.cast(this.dtype, src[0]);
        if (op === "Cmplt") {
          if (src[0].min >= src[1].max) return AluExp2.const("bool", false);
          if (src[0].max < src[1].min) return AluExp2.const("bool", true);
        }
        if (op === "Cmpne") {
          if (src[0].max < src[1].min || src[0].min > src[1].max) return AluExp2.const("bool", true);
        }
        if (op === "Where") {
          if (src[0].max === 0) return src[2];
          if (src[0].min === 1) return src[1];
        }
        if (op === "Mod" && src[1].op === "Const" && src[0].min >= 0 && src[0].max < src[1].arg) return src[0];
        if (op === "Mod" && src[0].op === "Mod" && src[1].#isConstInt() && src[0].src[1].#isConstInt()) {
          const A = src[0].src[1].arg;
          const B = src[1].arg;
          if (A > 0 && B > 0 && (A % B === 0 || B % A === 0)) return AluExp2.mod(src[0].src[0], AluExp2.const(this.dtype, Math.min(A, B))).simplify();
        }
        if (op === "Add" && src[0].op === "Mul" && src[0].src[1].#isConstInt() && src[1].op === "Mod" && src[1].src[1].#isConstInt() && src[0].src[1].arg === src[1].src[1].arg) {
          const [mul2, mod2] = src;
          const check = (exp3) => {
            return exp3.op === "Idiv" && exp3.src[1].#isConstInt() && exp3.src[1].arg === mod2.src[1].arg && exp3.src[0] === mod2.src[0];
          };
          if (check(mul2.src[0])) return mod2.src[0];
          if (mul2.src[0].op === "Mod") {
            const [x, y] = mul2.src[0].src;
            if (check(x)) return AluExp2.mod(mod2.src[0], AluExp2.mul(mod2.src[1], y)).simplify(cache);
          }
        }
        if (op === "Idiv" && src[1].#isConstInt()) {
          const [numer, denom] = src;
          const B = denom.arg;
          for (let i = 0; i < 2; i++) {
            if (numer.op === "Mul" && numer.src[i].#isConstInt()) {
              const A = numer.src[i].arg;
              if (A % B === 0) {
                let ret = numer.src[1 - i];
                if (A / B !== 1) ret = AluExp2.mul(ret, AluExp2.const(ret.dtype, A / B));
                return ret.simplify(cache);
              }
            }
            for (let j = 0; j < 2; j++) if (numer.op === "Add" && numer.src[j].op === "Mul" && numer.src[j].src[i].#isConstInt()) {
              const A = numer.src[j].src[i].arg;
              if (A % B === 0) {
                let ret = numer.src[j].src[1 - i];
                if (A / B !== 1) ret = AluExp2.mul(ret, AluExp2.const(ret.dtype, A / B));
                const y = numer.src[1 - j];
                const yOverB = y.divides(B);
                if (yOverB !== null) return AluExp2.add(ret, yOverB).simplify(cache);
                else if (B > 0 && (ret.min >= 0 && y.min >= 0 || ret.max <= 0 && y.max <= 0)) return AluExp2.add(ret, AluExp2.idiv(y, AluExp2.const(ret.dtype, B))).simplify(cache);
              }
            }
          }
        }
        if (op === "Mod" && src[1].#isConstInt() && src[1].arg > 0 && src[0].min >= 0) {
          const [numer, denom] = src;
          const B = denom.arg;
          for (let i = 0; i < 2; i++) if (numer.op === "Add") {
            if (numer.src[i].#isConstInt()) {
              const A = numer.src[i].arg;
              const x = numer.src[1 - i];
              if (A % B === 0 && x.min >= 0) return AluExp2.mod(x, denom).simplify(cache);
            }
            for (let j = 0; j < 2; j++) if (numer.src[i].op === "Mul" && numer.src[i].src[j].#isConstInt()) {
              const A = numer.src[i].src[j].arg;
              const x = numer.src[1 - i];
              if (A % B === 0 && x.min >= 0) return AluExp2.mod(x, denom).simplify(cache);
            }
          } else if (numer.op === "Mul") {
            if (numer.src[i].#isConstInt()) {
              const A = numer.src[i].arg;
              if (A % B === 0) return AluExp2.const(this.dtype, 0);
              if (A % B === 1) return AluExp2.mod(numer.src[1 - i], denom).simplify(cache);
            }
          }
        }
        if ([
          "Add",
          "Mul",
          "Max",
          "Min"
        ].includes(op)) {
          const p = (a, b) => new AluExp2(op, this.dtype, [a, b]);
          if (src[0].op === "Const") return p(src[1], src[0]).simplify(cache);
          if (src[0].op === op && src[0].src[1].op === "Const") if (src[1].op === "Const") return p(src[0].src[0], p(src[0].src[1], src[1])).simplify(cache);
          else return p(p(src[0].src[0], src[1]), src[0].src[1]).simplify(cache);
          if (src[1].op === op && src[1].src[1].op === "Const") return p(p(src[0], src[1].src[0]), src[1].src[1]).simplify(cache);
        }
        if ((op === "Mod" || op === "Idiv") && src[1].#isConstInt()) {
          const [x, y] = src;
          {
            const factors = [];
            const terms = [];
            for (const u of x.splitOp("Add")) {
              const factor = u.constFactor();
              factors.push(factor);
              terms.push(u.divides(factor));
            }
            const g = gcd(y.arg, ...factors);
            if (g !== 1) {
              let ret = new AluExp2(op, this.dtype, [factors.map((f, i) => AluExp2.mul(AluExp2.const(terms[i].dtype, f / g), terms[i])).reduceRight((a, x2) => AluExp2.add(x2, a)), AluExp2.const(y.dtype, y.arg / g)]);
              if (op === "Mod") ret = AluExp2.mul(ret, AluExp2.const(this.dtype, g));
              return ret.simplify(cache);
            }
          }
          if (y.arg > 0 && x.min >= 0) {
            let [xNoConst, constVal] = [x, 0];
            if (x.op === "Add" && x.src[1].op === "Const") [xNoConst, constVal] = [x.src[0], x.src[1].arg];
            const terms = [];
            const factors = [];
            for (const u of xNoConst.splitOp("Add")) {
              const f = u.constFactor();
              const divided = u.divides(f);
              terms.push(divided ?? u);
              factors.push(divided ? f : 1);
            }
            const quotients = factors.map((f) => Math.floor(f / y.arg));
            const remainders = factors.map((f) => f % y.arg);
            const gcdVal = remainders.reduce((g, r) => gcd(g, r), y.arg);
            if (constVal % y.arg !== constVal || gcdVal !== 1 || remainders.some((r, i) => r === 0 || r !== factors[i] && op === "Mod")) {
              let quo = AluExp2.const(x.dtype, Math.floor(constVal / y.arg));
              let rem = AluExp2.const(x.dtype, Math.floor(constVal % y.arg / gcdVal));
              for (let i = 0; i < terms.length; i++) if (op === "Idiv" && remainders[i] !== 0) rem = AluExp2.add(rem, AluExp2.mul(AluExp2.const(x.dtype, Math.floor(factors[i] / gcdVal)), terms[i]));
              else {
                rem = AluExp2.add(rem, AluExp2.mul(AluExp2.const(x.dtype, Math.floor(remainders[i] / gcdVal)), terms[i]));
                quo = AluExp2.add(quo, AluExp2.mul(AluExp2.const(x.dtype, quotients[i]), terms[i]));
              }
              if (rem.min >= 0) if (op === "Mod") return AluExp2.add(AluExp2.mul(AluExp2.const(x.dtype, gcdVal), AluExp2.mod(rem, AluExp2.const(x.dtype, Math.floor(y.arg / gcdVal)))), AluExp2.const(x.dtype, constVal % gcdVal)).simplify(cache);
              else return AluExp2.add(AluExp2.idiv(rem, AluExp2.const(x.dtype, Math.floor(y.arg / gcdVal))), quo).simplify(cache);
            }
          }
        }
        return src.every((s, i) => s === this.src[i]) ? this : new AluExp2(op, this.dtype, src, this.arg);
      }
      /** Resolve this to a value, or `undefined` if not possible. */
      resolve() {
        const x = this.simplify();
        if (x.op === "Const") return x.arg;
      }
      /**
      * Evaluate the expression on CPU, returning the result.
      *
      * Typically you would compile the AluExp as a representation to a lower-level
      * language. This is just to define the semantics and help debug.
      *
      * Note that the representation of Bool is as a number (0 or 1) here.
      */
      evaluate(context, globals) {
        if (AluGroup.Binary.has(this.op) || AluGroup.Compare.has(this.op)) {
          const x = this.src[0].evaluate(context, globals);
          const y = this.src[1].evaluate(context, globals);
          switch (this.op) {
            case "Add":
              return this.dtype === "bool" ? Number(x || y) : x + y;
            case "Sub":
              return x - y;
            case "Mul":
              return this.dtype === "bool" ? Number(x && y) : x * y;
            case "Idiv":
              return Math.trunc(x / y);
            case "Mod":
              return x % y;
            case "Min":
              return Math.min(x, y);
            case "Max":
              return Math.max(x, y);
            case "BitCombine": {
              let r;
              if (this.arg === "and") r = x & y;
              else if (this.arg === "or") r = x | y;
              else r = x ^ y;
              return this.dtype === "int32" ? r | 0 : r >>> 0;
            }
            case "BitShift":
              if (this.arg === "shl") return this.dtype === "int32" ? x << y | 0 : x << y >>> 0;
              return x >>> y;
            case "Cmplt":
              return Number(x < y);
            case "Cmpne":
              return Number(x != y);
            default:
              throw new Error(`Missing implemementation for ${this.op}`);
          }
        }
        if (AluGroup.Unary.has(this.op)) {
          const x = this.src[0].evaluate(context, globals);
          switch (this.op) {
            case "Sin":
              return Math.sin(x);
            case "Cos":
              return Math.cos(x);
            case "Asin":
              return Math.asin(x);
            case "Atan":
              return Math.atan(x);
            case "Exp":
              return Math.exp(x);
            case "Log":
              return Math.log(x);
            case "Erf":
              return erf(x);
            case "Erfc":
              return erfc(x);
            case "Sqrt":
              return Math.sqrt(x);
            case "Floor":
              return Math.floor(x);
            case "Ceil":
              return Math.ceil(x);
            case "Reciprocal":
              return 1 / x;
            case "Cast": {
              const wasFloat = isFloatDtype(this.src[0].dtype);
              if (this.dtype === "int32") return (wasFloat ? clamp(x, -2147483648, 2147483647) : x) | 0;
              else if (this.dtype === "uint32") return (wasFloat ? clamp(x, 0, 4294967295) : x) >>> 0;
              else if (isFloatDtype(this.dtype)) return x;
              else if (this.dtype === "bool") return Number(Boolean(x));
              else throw new Error(`Unsupported cast to ${this.dtype}`);
            }
            case "Bitcast": {
              const buf = new ArrayBuffer(byteWidth(this.dtype));
              const view = new DataView(buf);
              const fromType = this.src[0].dtype;
              if (fromType === "float32") view.setFloat32(0, x, true);
              else if (fromType === "int32") view.setInt32(0, x, true);
              else if (fromType === "uint32") view.setUint32(0, x, true);
              else if (fromType === "float16") view.setFloat16(0, x, true);
              else if (fromType === "float64") view.setFloat64(0, x, true);
              else throw new Error(`Unsupported bitcast from ${fromType}`);
              if (this.dtype === "float32") return view.getFloat32(0, true);
              else if (this.dtype === "int32") return view.getInt32(0, true);
              else if (this.dtype === "uint32") return view.getUint32(0, true);
              else if (this.dtype === "float16") return view.getFloat16(0, true);
              else if (this.dtype === "float64") return view.getFloat64(0, true);
              else throw new Error(`Unsupported bitcast to ${this.dtype}`);
            }
            default:
              throw new Error(`Missing implemementation for ${this.op}`);
          }
        }
        switch (this.op) {
          case "Where":
            return this.src[0].evaluate(context, globals) ? this.src[1].evaluate(context, globals) : this.src[2].evaluate(context, globals);
          case "Threefry2x32": {
            const [k0, k1, c0, c1] = this.src.map((x) => x.evaluate(context, globals));
            const [x0, x1] = threefry2x32(k0, k1, c0, c1);
            if (this.arg === "xor") return (x0 ^ x1) >>> 0;
            else if (this.arg === 0) return x0;
            else if (this.arg === 1) return x1;
            else throw new Error(`Invalid Threefry2x32 mode: ${this.arg}`);
          }
          case "Const":
            return this.arg;
          case "Special": {
            const x = context[this.arg[0]];
            if (x === void 0) throw new Error(`Missing special: ${this.arg[0]}`);
            return x;
          }
          case "Variable": {
            const x = context[this.arg];
            if (x === void 0) throw new Error(`Missing variable: ${this.arg}`);
            return x;
          }
          case "GlobalIndex": {
            if (!globals) throw new Error("Missing globals function");
            const gid = this.arg[0];
            return globals(gid, this.src[0].evaluate(context, globals));
          }
          case "GlobalView": {
            if (!globals) throw new Error("Missing globals function");
            const gid = this.arg[0];
            const [iexpr, vexpr] = this.arg[1].toAluExp(this.src);
            if (vexpr.evaluate(context, globals)) return globals(gid, iexpr.evaluate(context, globals));
            else return 0;
          }
          default:
            throw new Error(`Missing implemementation for ${this.op}`);
        }
      }
      /** Get this expression in debug format as a string. */
      toString() {
        const BIN_SYM = {
          ["Add"]: "+",
          ["Sub"]: "-",
          ["Mul"]: "*",
          ["Idiv"]: "/",
          ["Mod"]: "%"
        };
        const CMP_SYM = {
          ["Cmplt"]: "<",
          ["Cmpne"]: "!="
        };
        const UNARY_SYM = { ["Reciprocal"]: "1/" };
        return this.fold((node, parts) => {
          switch (node.op) {
            case "Const":
              return "" + (node.dtype === "bool" ? Boolean(node.arg) : node.arg);
            case "Variable":
              return `$${node.arg}:${node.dtype}`;
            case "Special": {
              const [name, n] = node.arg;
              return `#${name}{${n}}`;
            }
            case "GlobalIndex":
              return `G_${node.arg[0]}<${node.dtype}>[${strip1(parts[0])}]`;
            case "GlobalView": {
              const [gid, st] = node.arg;
              const shape2 = st.shape.join(",");
              const lastStrides = st.lastStrides.join(",");
              const cont = st.contiguous ? "c" : "nc";
              return `GV_${gid}<${node.dtype}>{${shape2}:${lastStrides}:${cont}}[${parts.map(strip1).join(", ")}]`;
            }
          }
          if (BIN_SYM[node.op]) return `(${parts[0]} ${BIN_SYM[node.op]} ${parts[1]})`;
          if (CMP_SYM[node.op]) return `(${parts[0]} ${CMP_SYM[node.op]} ${parts[1]})`;
          if (node.op === "BitCombine") {
            const sym = {
              and: "&",
              or: "|",
              xor: "^"
            }[node.arg];
            return `(${parts[0]} ${sym} ${parts[1]})`;
          }
          if (node.op === "BitShift") {
            const sym = node.arg === "shl" ? "<<" : ">>";
            return `(${parts[0]} ${sym} ${parts[1]})`;
          }
          if (UNARY_SYM[node.op]) return `${UNARY_SYM[node.op]}${parts[0]}`;
          if (node.op === "Cast") return `Cast<${node.dtype}>(${strip1(parts[0])})`;
          if (node.op === "Bitcast") return `Bitcast<${node.dtype}>(${strip1(parts[0])})`;
          return `${node.op}(${parts.map(strip1).join(", ")})`;
        });
      }
      /** Generic fold() operation with a reducer over the expression tree. */
      fold(reducer) {
        const visited = /* @__PURE__ */ new Map();
        const recurse = (exp3) => {
          if (visited.has(exp3)) return visited.get(exp3);
          const result = reducer(exp3, exp3.src.map((s) => recurse(s)));
          visited.set(exp3, result);
          return result;
        };
        return recurse(this);
      }
      /** Check if any expression in the tree satisfies a predicate. */
      some(predicate) {
        const visited = /* @__PURE__ */ new Set();
        const recurse = (exp3) => {
          if (visited.has(exp3)) return false;
          if (predicate(exp3)) return true;
          visited.add(exp3);
          return exp3.src.some(recurse);
        };
        return recurse(this);
      }
      /** Rewrite the expression recursively using a visitor. */
      rewrite(visitor) {
        return this.fold((exp3, newSrc) => {
          if (newSrc.length === exp3.src.length && newSrc.every((s, i) => s === exp3.src[i])) return visitor(exp3) ?? exp3;
          else {
            const newExp = new AluExp2(exp3.op, exp3.dtype, newSrc, exp3.arg);
            return visitor(newExp) ?? newExp;
          }
        });
      }
      /** Collect all nodes that satisfy a predicate. */
      collect(predicate) {
        const result = [];
        this.fold((exp3) => {
          if (predicate(exp3)) result.push(exp3);
        });
        return result;
      }
      /** Produce all distinct AluOp in this expression, with their dtypes. */
      distinctOps() {
        const ops = /* @__PURE__ */ new Map();
        this.fold((exp3) => {
          const s = ops.get(exp3.op) ?? /* @__PURE__ */ new Set();
          if (!s.has(exp3.dtype)) {
            s.add(exp3.dtype);
            ops.set(exp3.op, s);
          }
        });
        return ops;
      }
      /** Rewrite GlobalView operations to GlobalIndex operations. */
      rewriteGlobalViews() {
        return this.rewrite((exp3) => {
          if (exp3.op === "GlobalView") {
            const [gid, st] = exp3.arg;
            return accessorGlobal(exp3.dtype, gid, st, exp3.src);
          }
        });
      }
    };
    AluGroup = {
      Binary: /* @__PURE__ */ new Set([
        "Add",
        "Sub",
        "Mul",
        "Idiv",
        "Mod",
        "Min",
        "Max",
        "BitCombine",
        "BitShift"
      ]),
      Unary: /* @__PURE__ */ new Set([
        "Sin",
        "Cos",
        "Asin",
        "Atan",
        "Exp",
        "Log",
        "Erf",
        "Erfc",
        "Sqrt",
        "Floor",
        "Ceil",
        "Reciprocal",
        "Cast",
        "Bitcast"
      ]),
      Compare: /* @__PURE__ */ new Set(["Cmplt", "Cmpne"]),
      Variable: /* @__PURE__ */ new Set([
        "Special",
        "Variable",
        "GlobalIndex",
        "GlobalView"
      ]),
      Reduce: /* @__PURE__ */ new Set([
        "Add",
        "Mul",
        "Min",
        "Max"
      ]),
      RequiredFloat: /* @__PURE__ */ new Set([
        "Sin",
        "Cos",
        "Asin",
        "Atan",
        "Exp",
        "Log",
        "Erf",
        "Erfc",
        "Sqrt",
        "Reciprocal",
        "Floor",
        "Ceil"
      ])
    };
    AluVar = {
      gidx: AluExp.variable("int32", "gidx"),
      ridx: AluExp.variable("int32", "ridx"),
      acc: (dtype) => AluExp.variable(dtype, "acc"),
      idx: AluExp.variable("int32", "idx"),
      unroll: AluExp.variable("int32", "unroll"),
      upcast: AluExp.variable("int32", "upcast")
    };
    Kernel = class {
      nargs;
      size;
      exp;
      reduction;
      constructor(nargs, size2, exp3, reduction) {
        this.nargs = nargs;
        this.size = size2;
        this.exp = exp3;
        this.reduction = reduction;
        this.exp = exp3.simplify();
      }
      hash(state) {
        state.update(this.nargs).update(this.size).update(this.exp).update(this.reduction);
      }
      pprint() {
        let details = PPrint.pp(`exp = ${this.exp}`);
        details = details.concat(PPrint.pp(`size = ${this.size}`));
        if (this.reduction) details = details.concat(PPrint.pp(`reduction = ${this.reduction}`));
        return PPrint.pp("{ ").stack(details).stack(PPrint.pp(" }"));
      }
      toString() {
        return this.pprint().toString();
      }
      /** The dtype of the values output by this kernel. */
      get dtype() {
        if (this.reduction) return this.reduction.epilogue.dtype;
        else return this.exp.dtype;
      }
      /** The number of bytes in the output array when evaluating this kernel. */
      get bytes() {
        return this.size * byteWidth(this.dtype);
      }
    };
    Reduction = class {
      dtype;
      op;
      size;
      epilogue;
      constructor(dtype, op, size2, epilogue = AluVar.acc(dtype)) {
        this.dtype = dtype;
        this.op = op;
        this.size = size2;
        this.epilogue = epilogue;
        if (!AluGroup.Reduce.has(op)) throw new TypeError(`Unsupported reduction: ${op}`);
        this.epilogue = epilogue.simplify();
        if (this.dtype === "float16" && this.op === "Add") {
          this.epilogue = this.epilogue.substitute({ acc: AluExp.cast(this.dtype, AluVar.acc("float32")) });
          this.dtype = "float32";
        }
      }
      hash(state) {
        state.update(this.dtype).update(this.op).update(this.size).update(this.epilogue);
      }
      toString() {
        return `${this.op}{${this.size}} -> ${this.epilogue}`;
      }
      /** Get the identity for this reduction operation. */
      get identity() {
        if (this.dtype === "bool") return this.op === "Add" || this.op === "Max" ? 0 : 1;
        else if (this.dtype === "int32") {
          if (this.op === "Add") return 0;
          else if (this.op === "Mul") return 1;
          else if (this.op === "Min") return -1 >>> 1;
          else if (this.op === "Max") return 1 << 31;
        } else if (this.dtype === "uint32") {
          if (this.op === "Add") return 0;
          else if (this.op === "Mul") return 1;
          else if (this.op === "Min") return -1 >>> 0;
          else if (this.op === "Max") return 0;
        } else if (isFloatDtype(this.dtype)) {
          if (this.op === "Add") return 0;
          else if (this.op === "Mul") return 1;
          else if (this.op === "Min") return Infinity;
          else if (this.op === "Max") return -Infinity;
        }
        throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
      }
      /** Evaluate this operation on CPU. */
      evaluate(...values) {
        if (this.dtype === "bool") {
          if (this.op === "Add" || this.op === "Max") return values.reduce((a, b) => a || b, false);
          else if (this.op === "Mul" || this.op === "Min") return values.reduce((a, b) => a && b, true);
        } else if (this.dtype === "int32") {
          if (this.op === "Add") return values.reduce((a, b) => a + b | 0, 0);
          else if (this.op === "Mul") return values.reduce((a, b) => a * b | 0, 1);
          else if (this.op === "Min") return values.reduce((a, b) => Math.min(a, b), -1 >>> 1);
          else if (this.op === "Max") return values.reduce((a, b) => Math.max(a, b), 1 << 31);
        } else if (this.dtype === "uint32") {
          if (this.op === "Add") return values.reduce((a, b) => a + b >>> 0, 0);
          else if (this.op === "Mul") return values.reduce((a, b) => a * b >>> 0, 1);
          else if (this.op === "Min") return values.reduce((a, b) => Math.min(a, b), -1 >>> 0);
          else if (this.op === "Max") return values.reduce((a, b) => Math.max(a, b), 0);
        } else if (isFloatDtype(this.dtype)) {
          if (this.op === "Add") return values.reduce((a, b) => a + b, 0);
          else if (this.op === "Mul") return values.reduce((a, b) => a * b, 1);
          else if (this.op === "Min") return values.reduce((a, b) => Math.min(a, b), Infinity);
          else if (this.op === "Max") return values.reduce((a, b) => Math.max(a, b), -Infinity);
        }
        throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
      }
    };
    Routine = class {
      name;
      type;
      params;
      constructor(name, type, params) {
        this.name = name;
        this.type = type;
        this.params = params;
      }
    };
    Routines = /* @__PURE__ */ (function(Routines2) {
      Routines2["Sort"] = "Sort";
      Routines2["Argsort"] = "Argsort";
      Routines2["Scatter"] = "Scatter";
      Routines2["TriangularSolve"] = "TriangularSolve";
      Routines2["Cholesky"] = "Cholesky";
      Routines2["LU"] = "LU";
      Routines2["JacobiEigh"] = "JacobiEigh";
      Routines2["Fft"] = "Fft";
      return Routines2;
    })({});
    ScatterOp = /* @__PURE__ */ (function(ScatterOp2) {
      ScatterOp2["Update"] = "update";
      ScatterOp2["Add"] = "add";
      return ScatterOp2;
    })({});
    jstr = JSON.stringify;
    View = class View2 {
      shape;
      strides;
      offset;
      mask;
      #size;
      #contiguous;
      constructor(shape2, strides, offset, mask) {
        this.shape = shape2;
        this.strides = strides;
        this.offset = offset;
        this.mask = mask;
      }
      static create(shape2, strides, offset = 0, mask = null) {
        if (shape2.some((s) => s < 0)) throw new Error("View shape must be non-negative");
        strides = strides ? canonicalizeStrides(shape2, strides) : defaultStrides(shape2);
        if (shape2.includes(0)) return new View2(shape2, rep(shape2.length, 0), 0, null);
        if (mask !== null && mask.every(([b, e2], i) => b === 0 && e2 === shape2[i])) mask = null;
        if (mask !== null) {
          const elimDims = [];
          let hasNoData = false;
          for (let i = 0; i < shape2.length; i++) {
            const [b, e2] = mask[i];
            if (b + 1 >= e2) elimDims.push(i);
            if (b >= e2) hasNoData = true;
          }
          if (elimDims.length) {
            if (hasNoData) {
              strides = rep(shape2.length, 0);
              offset = 0;
              mask = rep(shape2.length, () => [0, 0]);
            }
            for (const i of elimDims) {
              offset += strides[i] * mask[i][0];
              strides[i] = 0;
            }
          }
        }
        return new View2(shape2, strides, offset, mask);
      }
      get ndim() {
        return this.shape.length;
      }
      get size() {
        if (this.#size === void 0) this.#size = prod(this.shape);
        return this.#size;
      }
      /** Whether this is a default, contiguous, unaltered view of the data (identity). */
      get contiguous() {
        if (this.#contiguous === void 0) this.#contiguous = this.size === 0 || this.offset === 0 && this.mask === null && deepEqual(this.strides, defaultStrides(this.shape));
        return this.#contiguous;
      }
      /** Return the range of data being indexed in this view, or [0, 0] if none. */
      dataRange() {
        if (this.size === 0 || this.mask && this.mask[0][0] === this.mask[0][1]) return [0, 0];
        let min2 = this.offset;
        let max2 = this.offset;
        for (let i = 0; i < this.ndim; i++) {
          let [lo, hi] = this.mask ? this.mask[i] : [0, this.shape[i]];
          --hi;
          const s = this.strides[i];
          if (s > 0) {
            min2 += s * lo;
            max2 += s * hi;
          } else if (s < 0) {
            min2 += s * hi;
            max2 += s * lo;
          }
        }
        return [min2, max2 + 1];
      }
      /** Produce an AluExp for evaluating this view at an index. */
      toAluExp(idxs) {
        let iexpr = AluExp.i32(this.offset);
        let vexpr = AluExp.bool(true);
        for (let i = this.ndim - 1; i >= 0; i--) {
          const idx = idxs[i];
          if (this.shape[i] !== 1 && this.strides[i] !== 0) iexpr = AluExp.add(AluExp.mul(idx, AluExp.i32(this.strides[i])), iexpr);
          if (this.mask) {
            if (this.mask[i][0] !== 0) vexpr = AluExp.mul(AluExp.cmplt(idx, AluExp.i32(this.mask[i][0])).not(), vexpr);
            if (this.mask[i][1] !== this.shape[i]) vexpr = AluExp.mul(AluExp.cmplt(idx, AluExp.i32(this.mask[i][1])), vexpr);
          }
        }
        return [iexpr, vexpr];
      }
      /**
      * Try to compose this view with another one. `this` view is applied first,
      * followed by the argument. If this is not possible for the specific views,
      * return `null` instead.
      *
      * If composable, return a combined view with the same shape as `v1`.
      *
      * This is very tricky. The shapes of v1 and v2 may be different, and in that
      * case, we do some math to figure out whether they're compatible.
      */
      compose(v1) {
        const v2 = this;
        if (v2.contiguous) return v1;
        if (v1.contiguous) {
          if (deepEqual(v1.shape, v2.shape)) return v2;
          if (v1.size === v2.size) {
            const ret = v2.reshape(v1.shape);
            if (ret !== null) return ret;
          }
        }
        if (v1.mask !== null) {
          const newV1 = v1.shrink(v1.mask);
          const merged = v2.compose(newV1);
          return merged ? merged.pad(zip(v1.mask, v1.shape).map(([m, s]) => [m[0], s - m[1]])) : null;
        }
        const origin = unravel(v2.shape, v1.offset);
        const terms = rep(v2.ndim, () => []);
        const strides = rep(v1.ndim, 0);
        for (let d1 = 0; d1 < v1.strides.length; d1++) {
          const st = v1.strides[d1];
          if (st === 0) continue;
          const unravelOffset = unravel(v2.shape, v1.offset + st);
          for (let d2 = 0; d2 < v2.ndim; d2++) {
            const o = origin[d2];
            const diff2 = unravelOffset[d2] - o;
            if (diff2 === 0) continue;
            terms[d2].push([d1, diff2]);
            strides[d1] += diff2 * v2.strides[d2];
          }
        }
        let [mergedSize, mergedTermMin, mergedTermMax] = [
          1,
          0,
          0
        ];
        const extents = [];
        for (let i = v2.ndim - 1; i >= 0; i--) {
          const term = terms[i];
          const s = v2.shape[i];
          let [tmin, tmax] = [origin[i], origin[i]];
          for (const [d1, s1] of term) if (s1 > 0) tmax += (v1.shape[d1] - 1) * s1;
          else if (s1 < 0) tmin += (v1.shape[d1] - 1) * s1;
          mergedTermMin += tmin * mergedSize;
          mergedTermMax += tmax * mergedSize;
          mergedSize *= s;
          if (mergedTermMin >= 0 && mergedTermMax < mergedSize) {
            extents.push([
              mergedSize,
              mergedTermMin,
              mergedTermMax
            ]);
            [mergedSize, mergedTermMin, mergedTermMax] = [
              1,
              0,
              0
            ];
          }
        }
        if (mergedTermMin !== 0 || mergedTermMax !== 0) return null;
        extents.reverse();
        const v2Shape = extents.map(([s]) => s);
        if (!deepEqual(v2Shape, v2.shape)) {
          const reshapedV2 = v2.reshape(v2Shape);
          if (reshapedV2 === null) return null;
          if (!deepEqual(reshapedV2.shape, v2.shape)) return reshapedV2.compose(v1);
        }
        if (v2.mask !== null) {
          const newB = rep(v1.ndim, 0);
          const newE = v1.shape.slice();
          let bad = false;
          for (let d2 = 0; d2 < v2.ndim; d2++) {
            const [b, e2] = v2.mask[d2];
            const o = origin[d2];
            const term = terms[d2];
            const [_, tmin, tmax] = extents[d2];
            if (b <= tmin && tmax < e2) continue;
            if (term.length !== 1) if (term.length === 0 && newE.length) newE[0] = 0;
            else bad = true;
            else {
              const [d1, s1] = term[0];
              newB[d1] = Math.max(newB[d1], Math.ceil((s1 > 0 ? b - o : e2 - o - 1) / s1));
              newE[d1] = Math.min(newE[d1], Math.floor((s1 < 0 ? b - o : e2 - o - 1) / s1) + 1);
            }
          }
          for (let d1 = 0; d1 < v1.ndim; d1++) if (newB[d1] !== 0 || newE[d1] !== v1.shape[d1]) return v2.compose(View2.create(v1.shape, v1.strides, v1.offset, zip(newB, newE)));
          if (bad) return null;
        }
        let finalOffset = v2.offset;
        for (let d2 = 0; d2 < v2.ndim; d2++) finalOffset += origin[d2] * v2.strides[d2];
        return View2.create(v1.shape, strides, finalOffset, null);
      }
      /** Attempt to simplify this view into a smaller reshaped form. */
      minify() {
        const minShape = mergeDims(this.shape, this.strides, this.mask).map((x) => x[0]);
        const nv = this.reshape(minShape);
        return nv ? nv : this;
      }
      /** Pad the view with zeros on each dimension. */
      pad(arg) {
        if (arg.length !== this.ndim || !arg.every(([b, e2]) => b >= 0 && e2 >= 0)) throw new Error(`invalid pad ${jstr(arg)} for ${jstr(this.shape)}`);
        if (arg.every(([b, e2]) => b === 0 && e2 === 0)) return this;
        const zvarg = arg.map(([b, e2], i) => [-b, this.shape[i] + e2]);
        const mask = arg.map(([b, _e], i) => [b, this.shape[i] + b]);
        return this.#unsafeResize(zvarg, mask);
      }
      /** Shrink the view by taking a subarray. */
      shrink(arg) {
        if (arg.length !== this.ndim || !arg.every(([b, e2], i) => 0 <= b && b <= e2 && e2 <= this.shape[i])) throw new Error(`invalid shrink ${jstr(arg)} for ${jstr(this.shape)}`);
        return this.#unsafeResize(arg);
      }
      #unsafeResize(arg, mask) {
        const offset = this.strides.map((s, i) => s * arg[i][0]).reduce((a, b) => a + b, 0);
        if (this.mask) {
          const nmask = this.mask.map(([mx, my], i) => [Math.max(0, Math.min(mx - arg[i][0], arg[i][1] - arg[i][0])), Math.max(0, Math.min(my - arg[i][0], arg[i][1] - arg[i][0]))]);
          mask = mask ? mask.map(([mx, my], i) => [Math.max(mx, nmask[i][0]), Math.min(my, nmask[i][1])]) : nmask;
        }
        return View2.create(arg.map(([b, e2]) => e2 - b), this.strides, this.offset + offset, mask);
      }
      /** Expand one or more axes with length "1" by repeating the data. */
      expand(newShape) {
        if (newShape.length !== this.ndim) throw new Error(`Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`);
        for (let i = 0; i < this.ndim; i++) if (newShape[i] !== this.shape[i] && this.shape[i] !== 1) throw new Error(`Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`);
        if (this.size === 0) return View2.create(newShape);
        const mask = this.mask ? this.mask.map((m, i) => this.shape[i] === newShape[i] ? m : m[0] === 0 && m[1] === 1 ? [0, newShape[i]] : [0, 0]) : null;
        return View2.create(newShape, this.strides, this.offset, mask);
      }
      /** Permute the axes of an array. */
      permute(axis) {
        if (!isPermutation(axis, this.ndim)) throw new Error(`Invalid permutation ${jstr(axis)} of len ${this.ndim}`);
        const newShape = axis.map((a) => this.shape[a]);
        const newStrides = axis.map((a) => this.strides[a]);
        const newMask = this.mask ? axis.map((a) => this.mask[a]) : null;
        return View2.create(newShape, newStrides, this.offset, newMask);
      }
      /** Flip (reverse) one or more axes of the view. */
      flip(arg) {
        if (arg.length !== this.ndim) throw new Error(`Invalid flip ${jstr(arg)} for ${jstr(this.shape)}`);
        const strides = this.strides.slice();
        let offset = this.offset;
        const mask = this.mask ? this.mask.slice() : null;
        for (let i = 0; i < this.ndim; i++) {
          const s = this.shape[i];
          if (arg[i]) {
            strides[i] = -strides[i];
            offset += (s - 1) * this.strides[i];
            if (mask) mask[i] = [s - mask[i][1], s - mask[i][0]];
          }
        }
        return View2.create(this.shape, strides, offset, mask);
      }
      /** Reshape the view into a new shape. */
      reshape(newShape) {
        if (deepEqual(this.shape, newShape)) return this;
        if (newShape.some((s) => s < 0)) throw new Error(`Reshape cannot have negative numbers ${jstr(newShape)}`);
        if (this.size !== prod(newShape)) throw new Error(`Reshape size ${jstr(this.shape)} -> ${jstr(newShape)}`);
        if (this.size === 0) return View2.create(newShape);
        if (newShape.length === 0 && this.mask?.some(([b, e2]) => b === e2)) return null;
        if (this.contiguous) return View2.create(newShape);
        const rStrides = [];
        const merge = mergeDims(this.shape, this.strides, this.mask);
        let rShapeIdx = newShape.length;
        for (let i = merge.length - 1; i >= 0; i--) {
          let [mergedSize, newStride, realSize] = merge[i];
          let acc = 1;
          while (acc < mergedSize && rShapeIdx > 0) {
            const newDim = newShape[--rShapeIdx];
            rStrides.push(newStride * acc);
            acc *= newDim;
            if (acc >= realSize) newStride = 0;
          }
          if (acc !== mergedSize) return null;
        }
        const newStrides = rep(newShape.length - rStrides.length, 0).concat(rStrides.reverse());
        if (!this.mask) return View2.create(newShape, newStrides, this.offset);
        const newMask = reshapeMask(this.mask, this.shape, newShape);
        if (!newMask) return null;
        let newOffset = this.offset;
        for (let i = 0; i < this.ndim; i++) newOffset += this.strides[i] * this.mask[i][0];
        for (let i = 0; i < newShape.length; i++) newOffset -= newStrides[i] * newMask[i][0];
        return View2.create(newShape, newStrides, newOffset, newMask);
      }
    };
    ShapeTracker = class ShapeTracker2 {
      views;
      constructor(views) {
        this.views = views;
      }
      /** Compose this shape tracker with another, applying it after this one. */
      compose(other) {
        if (this.contiguous) return other;
        let ret = this;
        for (const v of other.views) ret = new ShapeTracker2(ret.views.concat(v)).simplify();
        return ret;
      }
      static fromShape(shape2) {
        return new ShapeTracker2([View.create(shape2)]);
      }
      get contiguous() {
        return this.views.length === 1 && this.views[0].contiguous;
      }
      get consecutive() {
        return this.views.length === 1 && this.views[0].mask === null && deepEqual(this.views[0].strides, defaultStrides(this.views[0].shape));
      }
      get lastStrides() {
        return this.views[this.views.length - 1].strides;
      }
      get shape() {
        return this.views[this.views.length - 1].shape;
      }
      get size() {
        return this.views[this.views.length - 1].size;
      }
      toAluExp(idxs) {
        let [iexpr, vexpr] = this.views[this.views.length - 1].toAluExp(idxs);
        for (let i = this.views.length - 2; i >= 0; i--) {
          const view = this.views[i].minify();
          const exprs = view.toAluExp(unravelAlu(view.shape, iexpr));
          iexpr = exprs[0];
          vexpr = AluExp.mul(vexpr, exprs[1]);
        }
        return [iexpr.simplify(), vexpr.simplify()];
      }
      simplify() {
        const views = this.views.slice();
        while (views.length >= 2) {
          const newView = views[views.length - 2].compose(views[views.length - 1]);
          if (newView === null) break;
          views.splice(views.length - 2, 2, newView);
        }
        return new ShapeTracker2(views);
      }
      pad(arg) {
        return new ShapeTracker2(applyLast(this.views, (x) => x.pad(arg)));
      }
      shrink(arg) {
        return new ShapeTracker2(applyLast(this.views, (x) => x.shrink(arg)));
      }
      expand(newShape) {
        return new ShapeTracker2(applyLast(this.views, (x) => x.expand(newShape)));
      }
      permute(axis) {
        return new ShapeTracker2(applyLast(this.views, (x) => x.permute(axis)));
      }
      flip(arg) {
        return new ShapeTracker2(applyLast(this.views, (x) => x.flip(arg)));
      }
      reshape(newShape) {
        const newView = this.views[this.views.length - 1].reshape(newShape);
        return new ShapeTracker2(newView === null ? this.views.concat(View.create(newShape)) : this.views.toSpliced(this.views.length - 1, 1, newView));
      }
      /** Broadcast along the given new axes, then expand the shape. */
      broadcast(newShape, axis) {
        let st = this;
        if (axis.length > 0) {
          const unsqueezed = [...st.shape];
          for (const i of sorted(axis)) unsqueezed.splice(i, 0, 1);
          st = st.reshape(unsqueezed);
        }
        return st.expand(newShape);
      }
      /**
      * Repeat data in each axis by a positive number of repetitions.
      *
      * - If `tile` is true (default): [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
      * - If `tile` is false: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].
      */
      repeat(reps, tile2 = true) {
        if (reps.length > this.shape.length) throw new Error(`Too many repeats ${jstr(reps)} for shape ${jstr(this.shape)}`);
        if (reps.some((c) => c <= 0)) throw new Error(`Invalid repeats ${jstr(reps)}`);
        if (reps.length === 0) return this;
        const noop = this.shape.slice(0, -reps.length);
        const shape2 = this.shape.slice(-reps.length);
        return this.broadcast([...noop, ...shape2.flatMap((s, i) => tile2 ? [reps[i], s] : [s, reps[i]])], shape2.map((_, i) => noop.length + 2 * i + (tile2 ? 0 : 1))).reshape([...noop, ...shape2.map((s, i) => s * reps[i])]);
      }
      /** Move axis i to axis j. */
      moveaxis(i, j) {
        const perm = range(this.shape.length);
        perm.splice(i, 1);
        perm.splice(j, 0, i);
        return this.permute(perm);
      }
      /** Like pad(), but allows for negative values. */
      padOrShrink(arg) {
        const padArg = [];
        const shrinkArg = [];
        for (let i = 0; i < arg.length; i++) {
          const [b, e2] = arg[i];
          if (b < -this.shape[i] || e2 < -this.shape[i] || b + e2 < -this.shape[i]) throw new Error(`Invalid padOrShrink ${jstr(arg)} for ${jstr(this.shape)}`);
          padArg.push([Math.max(0, b), Math.max(0, e2)]);
          shrinkArg.push([Math.max(0, -b), this.shape[i] - Math.max(0, -e2)]);
        }
        return this.shrink(shrinkArg).pad(padArg);
      }
    };
    TuneDims = class {
      st;
      outputSt;
      groups;
      reduce;
      unroll;
      upcast;
      get end() {
        return this.st.shape.length;
      }
      constructor(shape2) {
        this.st = ShapeTracker.fromShape(shape2);
        this.outputSt = ShapeTracker.fromShape(shape2.slice(0, -1));
        this.groups = this.st.shape.length - 1;
        this.reduce = this.st.shape.length - 1;
        this.unroll = this.st.shape.length;
        this.upcast = this.st.shape.length;
      }
      applyLocal(axis, amount) {
        if (axis >= this.groups) throw new Error("Cannot localize reduction axis");
        const length = this.st.shape[axis];
        if (length % amount !== 0) throw new Error(`Localize by ${amount} on axis length ${length}`);
        if (length !== amount) {
          this.groups++, this.reduce++, this.unroll++, this.upcast++;
          this.st = this.st.reshape([
            ...this.st.shape.slice(0, axis),
            length / amount,
            amount,
            ...this.st.shape.slice(axis + 1)
          ]);
          this.outputSt = this.outputSt.reshape([
            ...this.outputSt.shape.slice(0, axis),
            length / amount,
            amount,
            ...this.outputSt.shape.slice(axis + 1)
          ]);
          axis++;
        }
        this.st = this.st.permute([
          ...range(axis),
          ...range(axis + 1, this.groups),
          axis,
          ...range(this.groups, this.st.shape.length)
        ]);
        this.outputSt = this.outputSt.permute([
          ...range(axis),
          ...range(axis + 1, this.groups),
          axis,
          ...range(this.groups, this.outputSt.shape.length)
        ]);
      }
      applyUpcast(axis, amount) {
        if (axis >= this.groups) throw new Error("Cannot upcast along reduction axis");
        const length = this.st.shape[axis];
        if (length % amount !== 0) throw new Error(`Upcast by ${amount} on axis length ${length}`);
        this.st = this.st.reshape([
          ...this.st.shape.slice(0, axis),
          length / amount,
          amount,
          ...this.st.shape.slice(axis + 1)
        ]).permute([
          ...range(axis + 1),
          ...range(axis + 2, this.st.shape.length + 1),
          axis + 1
        ]);
        this.outputSt = this.outputSt.reshape([
          ...this.outputSt.shape.slice(0, axis),
          length / amount,
          amount,
          ...this.outputSt.shape.slice(axis + 1)
        ]).permute([
          ...range(axis + 1),
          ...range(axis + 2, this.outputSt.shape.length + 1),
          axis + 1
        ]);
      }
      applyUnroll(axis, amount) {
        if (axis < this.groups) throw new Error("Cannot unroll non-reduce axis");
        if (axis >= this.unroll) throw new Error("Axis already unrolled");
        const length = this.st.shape[axis];
        if (length % amount !== 0) throw new Error(`Unroll by ${amount} on axis length ${length}`);
        if (length === amount) {
          this.st = this.st.permute([
            ...range(axis),
            ...range(axis + 1, this.upcast),
            axis,
            ...range(this.upcast, this.st.shape.length)
          ]);
          if (axis < this.reduce) this.reduce--;
          this.unroll--;
        } else {
          this.st = this.st.reshape([
            ...this.st.shape.slice(0, axis),
            length / amount,
            amount,
            ...this.st.shape.slice(axis + 1)
          ]).permute([
            ...range(axis + 1),
            ...range(axis + 2, this.upcast + 1),
            axis + 1,
            ...range(this.upcast + 1, this.st.shape.length + 1)
          ]);
          this.upcast++;
        }
      }
      applyGroup(axis, amount) {
        if (axis < this.reduce || axis >= this.unroll) throw new Error("Can only group reduction axes");
        const length = this.st.shape[axis];
        if (length % amount !== 0) throw new Error(`Group by ${amount} on axis length ${length}`);
        this.st = this.st.reshape([
          ...this.st.shape.slice(0, axis),
          length / amount,
          amount,
          ...this.st.shape.slice(axis + 1)
        ]).permute([
          ...range(this.reduce),
          axis + 1,
          ...range(this.reduce, axis + 1),
          ...range(axis + 2, this.st.shape.length + 1)
        ]);
        this.reduce++;
        this.unroll++;
        this.upcast++;
      }
    };
    CpuBackend = class {
      type = "cpu";
      maxArgs = Infinity;
      #buffers;
      #nextSlot;
      constructor() {
        this.#buffers = /* @__PURE__ */ new Map();
        this.#nextSlot = 1;
      }
      malloc(size2, initialData) {
        const buffer = new Uint8Array(size2);
        if (initialData) {
          if (initialData.byteLength !== size2) throw new Error("initialData size does not match buffer size");
          buffer.set(initialData);
        }
        const slot = this.#nextSlot++;
        this.#buffers.set(slot, {
          buffer,
          ref: 1
        });
        return slot;
      }
      incRef(slot) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        buffer.ref++;
      }
      decRef(slot) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        buffer.ref--;
        if (buffer.ref === 0) this.#buffers.delete(slot);
      }
      async read(slot, start, count) {
        return this.readSync(slot, start, count);
      }
      readSync(slot, start, count) {
        const buffer = this.#getBuffer(slot);
        if (start === void 0) start = 0;
        if (count === void 0) count = buffer.byteLength - start;
        return buffer.slice(start, start + count);
      }
      async prepareKernel(kernel) {
        return this.prepareKernelSync(kernel);
      }
      prepareKernelSync(kernel) {
        return new Executable(kernel, void 0);
      }
      async prepareRoutine(routine) {
        return this.prepareRoutineSync(routine);
      }
      prepareRoutineSync(routine) {
        return new Executable(routine, void 0);
      }
      dispatch(exe, inputs, outputs) {
        if (exe.source instanceof Routine) return runCpuRoutine(exe.source, inputs.map((slot) => this.#getBuffer(slot)), outputs.map((slot) => this.#getBuffer(slot)));
        const kernel = exe.source;
        const { exp: exp3, epilogue } = tuneNullopt(kernel);
        const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
        const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
        const usedArgs = new Map([...exp3.collect((exp4) => exp4.op === "GlobalIndex"), ...epilogue ? epilogue.collect((exp4) => exp4.op === "GlobalIndex") : []].map((exp4) => [exp4.arg[0], exp4.dtype]));
        const inputArrays = inputBuffers.map((buf, i) => {
          const dtype = usedArgs.get(i);
          if (!dtype) return null;
          return dtypedArray(dtype, buf);
        });
        const outputArray = dtypedArray(kernel.dtype, outputBuffers[0]);
        const globals = (gid, bufidx) => {
          if (gid < 0 || gid >= inputArrays.length) throw new Error("gid out of bounds: " + gid);
          if (bufidx < 0 || bufidx >= inputArrays[gid].length) throw new Error("bufidx out of bounds: " + bufidx);
          return inputArrays[gid][bufidx];
        };
        if (!kernel.reduction) for (let i = 0; i < kernel.size; i++) outputArray[i] = exp3.evaluate({ gidx: i }, globals);
        else for (let i = 0; i < kernel.size; i++) {
          let acc = kernel.reduction.identity;
          for (let j = 0; j < kernel.reduction.size; j++) {
            const item = exp3.evaluate({
              gidx: i,
              ridx: j
            }, globals);
            acc = kernel.reduction.evaluate(acc, item);
          }
          outputArray[i] = epilogue.evaluate({
            acc,
            gidx: i
          }, globals);
        }
      }
      #getBuffer(slot) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        return buffer.buffer;
      }
    };
    traceEnabled = false;
    flushCallbacks = [];
    ALLOCATION_ALIGNMENT = 64;
    WASM_PAGE_SIZE = 65536;
    MAX_MEMORY32_BYTES = 2 ** 32;
    WasmAllocator = class {
      #memory;
      #headPtr;
      #freeLists;
      #allocatedBuffers;
      constructor(memory) {
        this.#memory = memory;
        this.#headPtr = 64;
        this.#freeLists = /* @__PURE__ */ new Map();
        this.#allocatedBuffers = /* @__PURE__ */ new Map();
      }
      malloc(size2) {
        if (size2 === 0) return 0;
        const sizeClass = this.#findSizeClass(size2);
        const freeList = this.#freeLists.get(sizeClass);
        let ptr;
        if (freeList && freeList.length > 0) ptr = freeList.pop();
        else ptr = this.#bumpAlloc(sizeClass);
        this.#allocatedBuffers.set(ptr, sizeClass);
        return ptr;
      }
      free(ptr) {
        if (ptr === 0) return;
        const sizeClass = this.#allocatedBuffers.get(ptr);
        if (sizeClass === void 0) throw new Error(`Attempting to free unallocated pointer: ${ptr}`);
        const freeList = this.#freeLists.get(sizeClass);
        if (freeList) freeList.push(ptr);
        else this.#freeLists.set(sizeClass, [ptr]);
        this.#allocatedBuffers.delete(ptr);
      }
      #bumpAlloc(size2) {
        const ptr = this.#headPtr;
        const endPtr = ptr + alignTo(size2, ALLOCATION_ALIGNMENT);
        if (endPtr > MAX_MEMORY32_BYTES) throw new RangeError("Allocation exceeds the 4 GiB memory32 limit");
        const currentBytes = this.#memory.buffer.byteLength;
        if (endPtr > currentBytes) {
          const requiredPages = Math.ceil(endPtr / WASM_PAGE_SIZE);
          const currentPages = currentBytes / WASM_PAGE_SIZE;
          this.#memory.grow(requiredPages - currentPages);
        }
        this.#headPtr = endPtr;
        return ptr;
      }
      #findSizeClass(size2) {
        if (size2 <= 512) return alignTo(size2, 64);
        if (size2 <= 2048) return alignTo(size2, 512);
        if (size2 <= 65536) {
          let sizeClass = 4096;
          while (sizeClass < size2) sizeClass *= 2;
          return sizeClass;
        }
        return alignTo(size2, WASM_PAGE_SIZE);
      }
      getStats() {
        const freeListSizes = /* @__PURE__ */ new Map();
        for (const [sizeClass, freeList] of this.#freeLists) if (freeList.length > 0) freeListSizes.set(sizeClass, freeList.length);
        return {
          totalAllocated: this.#headPtr,
          freeListSizes
        };
      }
    };
    featureProbes = { "relaxed-madd": "0061736d0100000001080160037b7b7b017b030201000a0d010b00200020012002fd85020b" };
    featureSupportCache = /* @__PURE__ */ new Map();
    MIN_ELEMS_PER_THREAD = 512;
    WORKER_SOURCE = `
${cpuRoutineJSForWorkers()}

let memory = null;

const kernelLru = new Map();
function kernelFunc(kernelHash, module) {
  let func = kernelLru.get(kernelHash);
  if (func !== undefined) {
    kernelLru.delete(kernelHash);
    kernelLru.set(kernelHash, func);
    return func;
  }
  const instance = new WebAssembly.Instance(module, { env: { memory } });
  func = instance.exports.kernel;
  kernelLru.set(kernelHash, func);
  if (kernelLru.size > 256) {
    kernelLru.delete(kernelLru.keys().next().value);
  }
  return func;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    memory = msg.memory;
    postMessage({ type: "ready" });
    return;
  }
  try {
    if (msg.type === "kernel") {
      const { kernelHash, module, ptrs, begin, end } = msg;
      kernelFunc(kernelHash, module)(...ptrs, begin, end);
    } else if (msg.type === "routine") {
      const inputs = msg.inputs.map(({ ptr, size }) =>
        new Uint8Array(memory.buffer, ptr, size)
      );
      const outputs = msg.outputs.map(({ ptr, size }) =>
        new Uint8Array(memory.buffer, ptr, size)
      );
      __minify_safe_runCpuRoutine(msg.routine, inputs, outputs);
    } else {
      throw new Error("Unknown wasm worker message: " + msg.type);
    }
    postMessage({ type: "done", ok: true });
  } catch (err) {
    postMessage({ type: "done", ok: false, error: String(err) });
  }
};
`;
    WasmWorkerPool = class {
      #memory;
      #numWorkers;
      #workers = [];
      #ready = Promise.resolve();
      /** Serializes dispatches so concurrent read() calls don't clobber onmessage. */
      #queue = Promise.resolve();
      #epoch = 0n;
      #epochEnd = 0n;
      #hooks = /* @__PURE__ */ new Map();
      constructor(memory, numWorkers) {
        if (numWorkers <= 0) throw new Error("numWorkers must be positive");
        this.#memory = memory;
        this.#numWorkers = numWorkers;
      }
      get epoch() {
        return this.#epoch;
      }
      waitForEpoch(target) {
        if (target <= this.#epoch) return Promise.resolve();
        return new Promise((resolve) => {
          if (target <= this.#epoch) return resolve();
          const hooks = this.#hooks.get(target);
          if (hooks) hooks.push(resolve);
          else this.#hooks.set(target, [resolve]);
        });
      }
      #ensureInit() {
        if (this.#workers.length > 0) return;
        const blob = new Blob([WORKER_SOURCE], { type: "application/javascript" });
        const url = URL.createObjectURL(blob);
        this.#workers = [];
        const readyPromises = [];
        for (let i = 0; i < this.#numWorkers; i++) {
          const worker = new Worker(url, { type: "module" });
          this.#workers.push(worker);
          readyPromises.push(new Promise((resolve, reject) => {
            worker.onmessage = () => resolve();
            worker.onerror = (e2) => reject(new Error(e2.message || "Worker failed to load"));
          }));
          worker.postMessage({
            type: "init",
            memory: this.#memory
          });
        }
        this.#ready = Promise.all(readyPromises).then(() => {
          URL.revokeObjectURL(url);
        });
        this.#queue = this.#ready;
      }
      /**
      * Dispatch a kernel across multiple workers.
      *
      * Returns an epoch that can be used to wait for the ongoing work to complete,
      * which is guaranteed to be monotonically increasing.
      */
      dispatch(kernelHash, module, ptrs, size2, chunkAlignment = 16, minWorkPerWorker = MIN_ELEMS_PER_THREAD) {
        return this.#enqueue(() => this.#dispatchNow(kernelHash, module, ptrs, size2, chunkAlignment, minWorkPerWorker));
      }
      /**
      * Dispatch a CPU fallback routine on a worker.
      *
      * This uses the same serial queue as generated wasm kernels so routine reads
      * observe prior async writes and later kernels observe routine outputs.
      */
      dispatchRoutine(routine, inputs, outputs) {
        return this.#enqueue(() => this.#dispatchRoutineNow(routine, inputs, outputs));
      }
      #enqueue(job) {
        this.#ensureInit();
        this.#epochEnd++;
        const result = this.#queue.then(job);
        this.#queue = result.then(() => {
        }, () => {
        }).then(() => {
          this.#epoch++;
          const hooks = this.#hooks.get(this.#epoch);
          if (hooks) {
            for (const hook of hooks) hook();
            this.#hooks.delete(this.#epoch);
          }
        });
        return this.#epochEnd;
      }
      async #dispatchNow(kernelHash, module, ptrs, size2, chunkAlignment, minWorkPerWorker) {
        if (size2 === 0) return;
        const n = Math.min(this.#workers.length, Math.ceil(size2 / minWorkPerWorker));
        const chunkSize = Math.ceil(size2 / n / chunkAlignment) * chunkAlignment;
        const promises = [];
        for (let i = 0; i < n; i++) {
          const begin = i * chunkSize;
          const end = Math.min(begin + chunkSize, size2);
          if (begin >= size2) break;
          const worker = this.#workers[i];
          promises.push(new Promise((resolve, reject) => {
            worker.onmessage = (e2) => {
              if (e2.data.ok) resolve();
              else reject(/* @__PURE__ */ new Error(`Worker error: ${e2.data.error}`));
            };
            worker.postMessage({
              type: "kernel",
              kernelHash,
              module,
              ptrs,
              begin,
              end
            });
          }));
        }
        await Promise.all(promises);
      }
      async #dispatchRoutineNow(routine, inputs, outputs) {
        const worker = this.#workers[0];
        await new Promise((resolve, reject) => {
          worker.onmessage = (e2) => {
            if (e2.data.ok) resolve();
            else reject(/* @__PURE__ */ new Error(`Worker error: ${e2.data.error}`));
          };
          worker.postMessage({
            type: "routine",
            routine,
            inputs,
            outputs
          });
        });
      }
    };
    TILED_SIMD_ROWS = 128;
    TILED_SIMD_COLUMNS = 128;
    TILED_SIMD_K = 64;
    TILED_SIMD_MICRO_ROWS = 4;
    TILED_SIMD_MICRO_VECTORS = 4;
    K_SIMD_MICRO_ROWS = 4;
    K_SIMD_MICRO_COLS = 4;
    K_SIMD_MAX_UNROLL = 4;
    simdSupportedOps = /* @__PURE__ */ new Map();
    simdSupportedOps.set("float32", /* @__PURE__ */ new Set([
      "Add",
      "Sub",
      "Mul",
      "Floor",
      "Ceil",
      "Min",
      "Max",
      "Sqrt",
      "Cast",
      "Where",
      "Const",
      "GlobalIndex"
    ]));
    simdSupportedOps.set("int32", /* @__PURE__ */ new Set([
      "Add",
      "Sub",
      "Mul",
      "Min",
      "Max",
      "Cast",
      "Where",
      "Const",
      "GlobalIndex"
    ]));
    simdSupportedOps.set("uint32", simdSupportedOps.get("int32"));
    simdSupportedOps.set("bool", /* @__PURE__ */ new Set([
      "Add",
      "Mul",
      "Min",
      "Max",
      "Cmplt",
      "Cmpne",
      "Const",
      "GlobalIndex"
    ]));
    magicModuleHeader = [
      0,
      97,
      115,
      109
    ];
    moduleVersion = [
      1,
      0,
      0,
      0
    ];
    Function_ = class {
      inputTypes;
      outputTypes;
      body;
      locals = [];
      constructor(inputTypes, outputTypes, body) {
        this.inputTypes = inputTypes;
        this.outputTypes = outputTypes;
        this.body = body || (() => {
        });
      }
      emit() {
        this.locals = [];
        this.body();
      }
    };
    Memory = class {
      cg;
      min = 0;
      max = 0;
      isShared = false;
      aString = "";
      bString = "";
      constructor(cg) {
        this.cg = cg;
      }
      /** Declare the size of the memory. Each page is 64 KiB. */
      pages(min2, max2 = 0) {
        assert(this.min === 0 && this.max === 0);
        this.min = min2;
        this.max = max2;
        return this;
      }
      export(a) {
        assert(!this.isImport && !this.isExport, "already set");
        this.aString = a;
        return this;
      }
      shared(isShared) {
        this.isShared = isShared;
        return this;
      }
      import(a, b) {
        assert(!this.isImport && !this.isExport, "already set");
        this.aString = a;
        this.bString = b;
        return this;
      }
      size() {
        this.cg._emit(63);
        this.cg._emit(0);
      }
      grow() {
        this.cg._emit(64);
        this.cg._emit(0);
      }
      get isImport() {
        return this.aString.length > 0 && this.bString.length > 0;
      }
      get isExport() {
        return this.aString.length > 0 && this.bString.length === 0;
      }
    };
    CodeGenerator = class {
      local;
      i32;
      f32;
      f64;
      v128;
      i32x4;
      f32x4;
      memory;
      void = {
        typeId: 64,
        name: "void"
      };
      #functions = [];
      #importedFunctions = [];
      #exportedFunctions = /* @__PURE__ */ new Map();
      #curFunction = null;
      #curBytes = [];
      #typeStack = [];
      #blockFrames = [];
      constructor() {
        this.local = new Local(this);
        this.i32 = new I32(this);
        this.f32 = new F32(this);
        this.f64 = new F64(this);
        this.v128 = new V128(this);
        this.i32x4 = new I32x4(this);
        this.f32x4 = new F32x4(this);
        this.memory = new Memory(this);
      }
      unreachable() {
        this._emit(0);
      }
      nop() {
        this._emit(1);
      }
      block(...type) {
        this.#blockFrames.push({
          idx: this.#typeStack.length,
          ty: type
        });
        this._emit(2);
        this._emit(encodeBlocktype(type));
      }
      loop(...type) {
        this.#blockFrames.push({
          idx: this.#typeStack.length,
          ty: type
        });
        this._emit(3);
        this._emit(encodeBlocktype(type));
      }
      if(...type) {
        assert(this._pop().typeId === this.i32.typeId, "if_: expected i32");
        this.#blockFrames.push({
          idx: this.#typeStack.length,
          ty: type
        });
        this._emit(4);
        this._emit(encodeBlocktype(type));
      }
      else() {
        assert(this.#blockFrames.length > 0, "else: no block to else");
        const frame = this.#blockFrames[this.#blockFrames.length - 1];
        this.#typeStack = this.#typeStack.slice(0, frame.idx);
        this._emit(5);
      }
      /** End a block (`block`, `if`/`else`, `loop`, or function). */
      end() {
        const frame = this.#blockFrames.pop();
        assert(frame !== void 0, "end: no block to end");
        this.#typeStack = this.#typeStack.slice(0, frame.idx);
        for (const ty of frame.ty) if (ty.typeId !== this.void.typeId) this._push(ty);
        this._emit(11);
      }
      /** Branch to a block a certain depth outward on the stack. */
      br(depth) {
        this._emit(12);
        this._emit(encodeUnsigned(depth));
      }
      /** Conditional branch to a block a certain depth outward on the stack. */
      br_if(depth) {
        assert(this._pop().typeId === this.i32.typeId, "br_if: expected i32");
        this._emit(13);
        this._emit(encodeUnsigned(depth));
      }
      /** Jump table that indexes into a label vector (like switch). */
      br_table(...depths) {
        assert(this._pop().typeId === this.i32.typeId, "br_table: expected i32");
        assert(depths.length > 0, "br_table: expected at least one default depth");
        this._emit(14);
        this._emit(encodeUnsigned(depths.length - 1));
        for (const d of depths) this._emit(encodeUnsigned(d));
      }
      /** Return from a function, branching out of the outermost block. */
      return() {
        this._emit(15);
      }
      /** Call a function with the given ID. */
      call(fn) {
        assert(fn < this.#importedFunctions.length + this.#functions.length, "function index does not exist");
        const func = fn < this.#importedFunctions.length ? this.#importedFunctions[fn] : this.#functions[fn - this.#importedFunctions.length];
        for (let i = func.inputTypes.length - 1; i >= 0; i--) {
          const argType = this._pop();
          assert(argType.typeId === func.inputTypes[i].typeId, `call: argument ${i} type mismatch, expected ${func.inputTypes[i].name} got ${argType.name}`);
        }
        for (const outputType of func.outputTypes) this._push(outputType);
        this._emit(16);
        this._emit(encodeUnsigned(fn));
      }
      /** Throw away an operand on the stack. */
      drop() {
        this._pop();
        this._emit(26);
      }
      /** Select one of the first two operands (T, F) based on the third operand (i32)'s value. */
      select() {
        assert(this._pop().typeId === this.i32.typeId, "select: expected i32 condition");
        const [b, a] = [this._pop(), this._pop()];
        assert(a.typeId === b.typeId, "select: expected same type for both operands");
        this._push(a);
        this._emit(27);
      }
      /** Import a JavaScript function; returns its index. */
      importFunction(module, name, inputTypes, outputTypes) {
        if (this.#functions.length > 0) throw new Error("function imports must precede defining functions");
        const idx = this.#importedFunctions.length;
        this.#importedFunctions.push({
          module,
          name,
          inputTypes,
          outputTypes
        });
        return idx;
      }
      /** Export a function. */
      export(fn, name) {
        this.#exportedFunctions.set(fn, name);
      }
      /** Declare a new function; returns its index. */
      function(inputTypes, outputTypes, body) {
        const idx = this.#importedFunctions.length + this.#functions.length;
        this.#functions.push(new Function_(inputTypes, outputTypes, body));
        return idx;
      }
      _declareLocal(type) {
        assert(this.#curFunction !== null, "No current function");
        const idx = this.#curFunction.locals.length + this.#curFunction.inputTypes.length;
        this.#curFunction.locals.push(type);
        return idx;
      }
      _inputTypes() {
        assert(this.#curFunction !== null, "No current function");
        return this.#curFunction.inputTypes;
      }
      _locals() {
        assert(this.#curFunction !== null, "No current function");
        return this.#curFunction.locals;
      }
      _push(type) {
        if (!type) throw new Error(`pushing type ${type}`);
        this.#typeStack.push(type);
      }
      _pop() {
        assert(this.#typeStack.length > 0, "popping empty stack");
        return this.#typeStack.pop();
      }
      _emit(bytes) {
        if (typeof bytes === "number") this.#curBytes.push(bytes);
        else this.#curBytes.push(...bytes);
      }
      finish() {
        this.#curBytes = [];
        const emittedBytes = [];
        emittedBytes.push(...magicModuleHeader);
        emittedBytes.push(...moduleVersion);
        const typeSectionBytes = [];
        const totalFunctionTypes = this.#importedFunctions.length + this.#functions.length;
        typeSectionBytes.push(...encodeUnsigned(totalFunctionTypes));
        for (const f of [...this.#importedFunctions, ...this.#functions]) {
          typeSectionBytes.push(96);
          typeSectionBytes.push(...encodeUnsigned(f.inputTypes.length));
          for (const t of f.inputTypes) typeSectionBytes.push(t.typeId);
          typeSectionBytes.push(...encodeUnsigned(f.outputTypes.length));
          for (const t of f.outputTypes) typeSectionBytes.push(t.typeId);
        }
        emittedBytes.push(1);
        appendLengthEncodedBlock(emittedBytes, typeSectionBytes);
        const importSectionBytes = [];
        const numImports = this.#importedFunctions.length + (this.memory.isImport ? 1 : 0);
        if (numImports > 0) {
          importSectionBytes.push(...encodeUnsigned(numImports));
          for (let i = 0; i < this.#importedFunctions.length; i++) {
            const f = this.#importedFunctions[i];
            importSectionBytes.push(...encodeString(f.module));
            importSectionBytes.push(...encodeString(f.name));
            importSectionBytes.push(0);
            importSectionBytes.push(...encodeUnsigned(i));
          }
          if (this.memory.isImport) {
            importSectionBytes.push(...encodeString(this.memory.aString));
            importSectionBytes.push(...encodeString(this.memory.bString));
            importSectionBytes.push(2);
            if (this.memory.max) {
              if (this.memory.isShared) importSectionBytes.push(3);
              else importSectionBytes.push(1);
              importSectionBytes.push(...encodeUnsigned(this.memory.min));
              importSectionBytes.push(...encodeUnsigned(this.memory.max));
            } else {
              assert(!this.memory.isShared, "shared memory must have a max size");
              importSectionBytes.push(0);
              importSectionBytes.push(...encodeUnsigned(this.memory.min));
            }
          }
          emittedBytes.push(2);
          appendLengthEncodedBlock(emittedBytes, importSectionBytes);
        }
        const functionSectionBytes = [];
        functionSectionBytes.push(...encodeUnsigned(this.#functions.length));
        for (let i = 0; i < this.#functions.length; i++) {
          const typeIndex = this.#importedFunctions.length + i;
          functionSectionBytes.push(...encodeUnsigned(typeIndex));
        }
        emittedBytes.push(3);
        appendLengthEncodedBlock(emittedBytes, functionSectionBytes);
        const memorySectionBytes = [];
        if (!this.memory.isImport && (this.memory.min || this.memory.max)) {
          memorySectionBytes.push(1);
          if (this.memory.min && this.memory.max) {
            if (this.memory.isShared) memorySectionBytes.push(3);
            else memorySectionBytes.push(1);
            memorySectionBytes.push(...encodeUnsigned(this.memory.min));
            memorySectionBytes.push(...encodeUnsigned(this.memory.max));
          } else {
            assert(!this.memory.isShared, "shared memory must have a max size");
            memorySectionBytes.push(0);
            memorySectionBytes.push(...encodeUnsigned(this.memory.min));
          }
          emittedBytes.push(5);
          appendLengthEncodedBlock(emittedBytes, memorySectionBytes);
        }
        const exportSectionBytes = [];
        const numExports = this.#exportedFunctions.size + (this.memory.isExport ? 1 : 0);
        exportSectionBytes.push(...encodeUnsigned(numExports));
        if (this.memory.isExport) {
          exportSectionBytes.push(...encodeString(this.memory.aString));
          exportSectionBytes.push(2);
          exportSectionBytes.push(0);
        }
        for (const [key, name] of this.#exportedFunctions.entries()) {
          exportSectionBytes.push(...encodeString(name));
          exportSectionBytes.push(0);
          exportSectionBytes.push(...encodeUnsigned(key));
        }
        emittedBytes.push(7);
        appendLengthEncodedBlock(emittedBytes, exportSectionBytes);
        const codeSectionBytes = [];
        codeSectionBytes.push(...encodeUnsigned(this.#functions.length));
        for (const f of this.#functions) {
          this.#typeStack = [];
          this.#blockFrames = [{
            idx: 0,
            ty: f.outputTypes
          }];
          this.#curFunction = f;
          this.#curBytes = [];
          f.emit();
          this.end();
          const bodyBytes = this.#curBytes;
          this.#curBytes = [];
          this.#curBytes.push(...encodeUnsigned(f.locals.length));
          for (const l of f.locals) {
            this._emit(1);
            this._emit(l.typeId);
          }
          appendLengthEncodedBlock(codeSectionBytes, this.#curBytes.concat(bodyBytes));
        }
        this.#curFunction = null;
        emittedBytes.push(10);
        appendLengthEncodedBlock(emittedBytes, codeSectionBytes);
        return new Uint8Array(emittedBytes);
      }
    };
    Local = class {
      cg;
      constructor(cg) {
        this.cg = cg;
      }
      declare(type) {
        return this.cg._declareLocal(type);
      }
      get(idx) {
        assert(Number.isInteger(idx), "getting non-integer local");
        const inputTypes = this.cg._inputTypes();
        if (idx < inputTypes.length) this.cg._push(inputTypes[idx]);
        else this.cg._push(this.cg._locals()[idx - inputTypes.length]);
        this.cg._emit(32);
        this.cg._emit(encodeUnsigned(idx));
      }
      set(idx) {
        const t = this.cg._pop();
        const inputTypes = this.cg._inputTypes();
        assert((idx < inputTypes.length ? inputTypes[idx] : this.cg._locals()[idx - inputTypes.length]).typeId === t.typeId, "can't set local to this value (wrong type)");
        this.cg._emit(33);
        this.cg._emit(encodeUnsigned(idx));
      }
      tee(idx) {
        const t = this.cg._pop();
        const inputTypes = this.cg._inputTypes();
        const expectedType = idx < inputTypes.length ? inputTypes[idx] : this.cg._locals()[idx - inputTypes.length];
        assert(expectedType.typeId === t.typeId, "can't tee local to this value (wrong type)");
        this.cg._emit(34);
        this.cg._emit(encodeUnsigned(idx));
        this.cg._push(expectedType);
      }
    };
    I32 = class {
      cg;
      constructor(cg) {
        this.cg = cg;
      }
      get typeId() {
        return 127;
      }
      get name() {
        return "i32";
      }
      const(i) {
        this.cg._emit(65);
        this.cg._emit(encodeSigned(i));
        this.cg._push(this);
      }
      clz = UNARY_OP("clz", 103, "i32", "i32");
      ctz = UNARY_OP("ctz", 104, "i32", "i32");
      popcnt = UNARY_OP("popcnt", 105, "i32", "i32");
      lt_s = BINARY_OP("lt_s", 72, "i32", "i32", "i32");
      lt_u = BINARY_OP("lt_u", 73, "i32", "i32", "i32");
      gt_s = BINARY_OP("gt_s", 74, "i32", "i32", "i32");
      gt_u = BINARY_OP("gt_u", 75, "i32", "i32", "i32");
      le_s = BINARY_OP("le_s", 76, "i32", "i32", "i32");
      le_u = BINARY_OP("le_u", 77, "i32", "i32", "i32");
      ge_s = BINARY_OP("ge_s", 78, "i32", "i32", "i32");
      ge_u = BINARY_OP("ge_u", 79, "i32", "i32", "i32");
      add = BINARY_OP("add", 106, "i32", "i32", "i32");
      sub = BINARY_OP("sub", 107, "i32", "i32", "i32");
      mul = BINARY_OP("mul", 108, "i32", "i32", "i32");
      div_s = BINARY_OP("div_s", 109, "i32", "i32", "i32");
      div_u = BINARY_OP("div_u", 110, "i32", "i32", "i32");
      rem_s = BINARY_OP("rem_s", 111, "i32", "i32", "i32");
      rem_u = BINARY_OP("rem_u", 112, "i32", "i32", "i32");
      and = BINARY_OP("and", 113, "i32", "i32", "i32");
      or = BINARY_OP("or", 114, "i32", "i32", "i32");
      xor = BINARY_OP("xor", 115, "i32", "i32", "i32");
      shl = BINARY_OP("shl", 116, "i32", "i32", "i32");
      shr_s = BINARY_OP("shr_s", 117, "i32", "i32", "i32");
      shr_u = BINARY_OP("shr_u", 118, "i32", "i32", "i32");
      rotl = BINARY_OP("rotl", 119, "i32", "i32", "i32");
      rotr = BINARY_OP("rotr", 120, "i32", "i32", "i32");
      eqz = UNARY_OP("eqz", 69, "i32", "i32");
      eq = BINARY_OP("eq", 70, "i32", "i32", "i32");
      ne = BINARY_OP("ne", 71, "i32", "i32", "i32");
      trunc_f32_s = UNARY_OP("trunc_f32_s", 168, "f32", "i32");
      trunc_f32_u = UNARY_OP("trunc_f32_u", 169, "f32", "i32");
      trunc_f64_s = UNARY_OP("trunc_f64_s", 170, "f64", "i32");
      trunc_f64_u = UNARY_OP("trunc_f64_u", 171, "f64", "i32");
      load = LOAD_OP("load", 40, "i32");
      load8_s = LOAD_OP("load8_s", 44, "i32");
      load8_u = LOAD_OP("load8_u", 45, "i32");
      load16_s = LOAD_OP("load16_s", 46, "i32");
      load16_u = LOAD_OP("load16_u", 47, "i32");
      store = STORE_OP("store", 54, "i32");
      store8 = STORE_OP("store8", 58, "i32");
      store16 = STORE_OP("store16", 59, "i32");
      reinterpret_f32 = UNARY_OP("reinterpret_f32", 188, "f32", "i32");
      trunc_sat_f32_s = UNARY_OP("trunc_sat_f32_s", [252, 0], "f32", "i32");
      trunc_sat_f32_u = UNARY_OP("trunc_sat_f32_u", [252, 1], "f32", "i32");
      trunc_sat_f64_s = UNARY_OP("trunc_sat_f64_s", [252, 2], "f64", "i32");
      trunc_sat_f64_u = UNARY_OP("trunc_sat_f64_u", [252, 3], "f64", "i32");
    };
    F32 = class {
      cg;
      constructor(cg) {
        this.cg = cg;
      }
      get typeId() {
        return 125;
      }
      get name() {
        return "f32";
      }
      const(f) {
        this.cg._emit(67);
        const buffer = /* @__PURE__ */ new ArrayBuffer(4);
        new DataView(buffer).setFloat32(0, f, true);
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < 4; i++) this.cg._emit(bytes[i]);
        this.cg._push(this);
      }
      load = LOAD_OP("load", 42, "f32");
      store = STORE_OP("store", 56, "f32");
      eq = BINARY_OP("eq", 91, "f32", "f32", "i32");
      ne = BINARY_OP("ne", 92, "f32", "f32", "i32");
      lt = BINARY_OP("lt", 93, "f32", "f32", "i32");
      gt = BINARY_OP("gt", 94, "f32", "f32", "i32");
      le = BINARY_OP("le", 95, "f32", "f32", "i32");
      ge = BINARY_OP("ge", 96, "f32", "f32", "i32");
      abs = UNARY_OP("abs", 139, "f32", "f32");
      neg = UNARY_OP("neg", 140, "f32", "f32");
      ceil = UNARY_OP("ceil", 141, "f32", "f32");
      floor = UNARY_OP("floor", 142, "f32", "f32");
      trunc = UNARY_OP("trunc", 143, "f32", "f32");
      nearest = UNARY_OP("nearest", 144, "f32", "f32");
      sqrt = UNARY_OP("sqrt", 145, "f32", "f32");
      add = BINARY_OP("add", 146, "f32", "f32", "f32");
      sub = BINARY_OP("sub", 147, "f32", "f32", "f32");
      mul = BINARY_OP("mul", 148, "f32", "f32", "f32");
      div = BINARY_OP("div", 149, "f32", "f32", "f32");
      min = BINARY_OP("min", 150, "f32", "f32", "f32");
      max = BINARY_OP("max", 151, "f32", "f32", "f32");
      copysign = BINARY_OP("copysign", 152, "f32", "f32", "f32");
      convert_i32_s = UNARY_OP("convert_i32_s", 178, "i32", "f32");
      convert_i32_u = UNARY_OP("convert_i32_u", 179, "i32", "f32");
      demote_f64 = UNARY_OP("demote_f64", 182, "f64", "f32");
      reinterpret_i32 = UNARY_OP("reinterpret_i32", 190, "i32", "f32");
    };
    F64 = class {
      cg;
      constructor(cg) {
        this.cg = cg;
      }
      get typeId() {
        return 124;
      }
      get name() {
        return "f64";
      }
      const(f) {
        this.cg._emit(68);
        const buffer = /* @__PURE__ */ new ArrayBuffer(8);
        new DataView(buffer).setFloat64(0, f, true);
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < 8; i++) this.cg._emit(bytes[i]);
        this.cg._push(this);
      }
      load = LOAD_OP("load", 43, "f64");
      store = STORE_OP("store", 57, "f64");
      eq = BINARY_OP("eq", 97, "f64", "f64", "i32");
      ne = BINARY_OP("ne", 98, "f64", "f64", "i32");
      lt = BINARY_OP("lt", 99, "f64", "f64", "i32");
      gt = BINARY_OP("gt", 100, "f64", "f64", "i32");
      le = BINARY_OP("le", 101, "f64", "f64", "i32");
      ge = BINARY_OP("ge", 102, "f64", "f64", "i32");
      abs = UNARY_OP("abs", 153, "f64", "f64");
      neg = UNARY_OP("neg", 154, "f64", "f64");
      ceil = UNARY_OP("ceil", 155, "f64", "f64");
      floor = UNARY_OP("floor", 156, "f64", "f64");
      trunc = UNARY_OP("trunc", 157, "f64", "f64");
      nearest = UNARY_OP("nearest", 158, "f64", "f64");
      sqrt = UNARY_OP("sqrt", 159, "f64", "f64");
      add = BINARY_OP("add", 160, "f64", "f64", "f64");
      sub = BINARY_OP("sub", 161, "f64", "f64", "f64");
      mul = BINARY_OP("mul", 162, "f64", "f64", "f64");
      div = BINARY_OP("div", 163, "f64", "f64", "f64");
      min = BINARY_OP("min", 164, "f64", "f64", "f64");
      max = BINARY_OP("max", 165, "f64", "f64", "f64");
      copysign = BINARY_OP("copysign", 166, "f64", "f64", "f64");
      convert_i32_s = UNARY_OP("convert_i32_s", 183, "i32", "f64");
      convert_i32_u = UNARY_OP("convert_i32_u", 184, "i32", "f64");
      promote_f32 = UNARY_OP("promote_f32", 187, "f32", "f64");
    };
    V128 = class {
      cg;
      constructor(cg) {
        this.cg = cg;
      }
      get typeId() {
        return 123;
      }
      get name() {
        return "v128";
      }
      load = VECTOR_LOAD_OP("load", 0);
      load32x2_s = VECTOR_LOAD_OP("load32x2_s", 5);
      load32x2_u = VECTOR_LOAD_OP("load32x2_u", 6);
      load32_splat = VECTOR_LOAD_OP("load32_splat", 9);
      load32_zero = VECTOR_LOAD_OP("load32_zero", 92);
      store(align = 0, offset = 0) {
        assert(this.cg._pop().typeId === this.cg.v128.typeId, `invalid type for store`);
        assert(this.cg._pop().typeId === this.cg.i32.typeId, `invalid type for store`);
        this.cg._emit(253);
        this.cg._emit(encodeUnsigned(11));
        this.cg._emit(encodeUnsigned(align));
        this.cg._emit(encodeUnsigned(offset));
      }
      not = VECTOR_OP("not", 77, ["v128"], "v128");
      and = VECTOR_OP("and", 78, ["v128", "v128"], "v128");
      andnot = VECTOR_OP("andnot", 79, ["v128", "v128"], "v128");
      or = VECTOR_OP("or", 80, ["v128", "v128"], "v128");
      xor = VECTOR_OP("xor", 81, ["v128", "v128"], "v128");
      bitselect = VECTOR_OP("bitselect", 82, [
        "v128",
        "v128",
        "v128"
      ], "v128");
      any_true = VECTOR_OP("any_true", 83, ["v128"], "i32");
    };
    I32x4 = class extends V128 {
      splat = VECTOR_OP("splat", 17, ["i32"], "v128");
      extract_lane = VECTOR_OPL("extract_lane", 27, ["v128"], "i32");
      replace_lane = VECTOR_OPL("replace_lane", 28, ["v128", "i32"], "v128");
      eq = VECTOR_OP("eq", 55, ["v128", "v128"], "v128");
      ne = VECTOR_OP("ne", 56, ["v128", "v128"], "v128");
      lt_s = VECTOR_OP("lt_s", 57, ["v128", "v128"], "v128");
      lt_u = VECTOR_OP("lt_u", 58, ["v128", "v128"], "v128");
      gt_s = VECTOR_OP("gt_s", 59, ["v128", "v128"], "v128");
      gt_u = VECTOR_OP("gt_u", 60, ["v128", "v128"], "v128");
      le_s = VECTOR_OP("le_s", 61, ["v128", "v128"], "v128");
      le_u = VECTOR_OP("le_u", 62, ["v128", "v128"], "v128");
      ge_s = VECTOR_OP("ge_s", 63, ["v128", "v128"], "v128");
      ge_u = VECTOR_OP("ge_u", 64, ["v128", "v128"], "v128");
      abs = VECTOR_OP("abs", 160, ["v128"], "v128");
      neg = VECTOR_OP("neg", 161, ["v128"], "v128");
      all_true = VECTOR_OP("all_true", 163, ["v128"], "i32");
      bitmask = VECTOR_OP("bitmask", 164, ["v128"], "i32");
      shl = VECTOR_OP("shl", 171, ["v128", "i32"], "v128");
      shr_s = VECTOR_OP("shr_s", 172, ["v128", "i32"], "v128");
      shr_u = VECTOR_OP("shr_u", 173, ["v128", "i32"], "v128");
      add = VECTOR_OP("add", 174, ["v128", "v128"], "v128");
      sub = VECTOR_OP("sub", 177, ["v128", "v128"], "v128");
      mul = VECTOR_OP("mul", 181, ["v128", "v128"], "v128");
      min_s = VECTOR_OP("min_s", 182, ["v128", "v128"], "v128");
      min_u = VECTOR_OP("min_u", 183, ["v128", "v128"], "v128");
      max_s = VECTOR_OP("max_s", 184, ["v128", "v128"], "v128");
      max_u = VECTOR_OP("max_u", 185, ["v128", "v128"], "v128");
      trunc_sat_f32x4_s = VECTOR_OP("trunc_sat_f32x4_s", 248, ["v128"], "v128");
      trunc_sat_f32x4_u = VECTOR_OP("trunc_sat_f32x4_u", 249, ["v128"], "v128");
    };
    F32x4 = class extends V128 {
      splat = VECTOR_OP("splat", 19, ["f32"], "v128");
      extract_lane = VECTOR_OPL("extract_lane", 31, ["v128"], "f32");
      replace_lane = VECTOR_OPL("replace_lane", 32, ["v128", "f32"], "v128");
      eq = VECTOR_OP("eq", 65, ["v128", "v128"], "v128");
      ne = VECTOR_OP("ne", 66, ["v128", "v128"], "v128");
      lt = VECTOR_OP("lt", 67, ["v128", "v128"], "v128");
      gt = VECTOR_OP("gt", 68, ["v128", "v128"], "v128");
      le = VECTOR_OP("le", 69, ["v128", "v128"], "v128");
      ge = VECTOR_OP("ge", 70, ["v128", "v128"], "v128");
      ceil = VECTOR_OP("ceil", 103, ["v128"], "v128");
      floor = VECTOR_OP("floor", 104, ["v128"], "v128");
      trunc = VECTOR_OP("trunc", 105, ["v128"], "v128");
      nearest = VECTOR_OP("nearest", 106, ["v128"], "v128");
      abs = VECTOR_OP("abs", 224, ["v128"], "v128");
      neg = VECTOR_OP("neg", 225, ["v128"], "v128");
      sqrt = VECTOR_OP("sqrt", 227, ["v128"], "v128");
      add = VECTOR_OP("add", 228, ["v128", "v128"], "v128");
      sub = VECTOR_OP("sub", 229, ["v128", "v128"], "v128");
      mul = VECTOR_OP("mul", 230, ["v128", "v128"], "v128");
      div = VECTOR_OP("div", 231, ["v128", "v128"], "v128");
      min = VECTOR_OP("min", 232, ["v128", "v128"], "v128");
      max = VECTOR_OP("max", 233, ["v128", "v128"], "v128");
      pmin = VECTOR_OP("pmin", 234, ["v128", "v128"], "v128");
      pmax = VECTOR_OP("pmax", 235, ["v128", "v128"], "v128");
      convert_i32x4_s = VECTOR_OP("convert_i32x4_s", 250, ["v128"], "v128");
      convert_i32x4_u = VECTOR_OP("convert_i32x4_u", 251, ["v128"], "v128");
      relaxed_madd = VECTOR_OP("relaxed_madd", 261, [
        "v128",
        "v128",
        "v128"
      ], "v128");
      relaxed_nmadd = VECTOR_OP("relaxed_nmadd", 262, [
        "v128",
        "v128",
        "v128"
      ], "v128");
    };
    compiledProgramCache = /* @__PURE__ */ new Map();
    WasmBackend = class {
      type = "wasm";
      maxArgs = 64;
      #memory;
      #nextSlot;
      #allocator;
      #buffers;
      #workerPool;
      #pendingWork = /* @__PURE__ */ new Map();
      constructor() {
        this.#memory = hasSharedArrayBuffer() ? new WebAssembly.Memory({
          initial: 0,
          maximum: 65536,
          shared: true
        }) : new WebAssembly.Memory({ initial: 0 });
        this.#allocator = new WasmAllocator(this.#memory);
        this.#nextSlot = 1;
        this.#buffers = /* @__PURE__ */ new Map();
        this.#workerPool = createWorkerPool(this.#memory);
      }
      malloc(size2, initialData) {
        const ptr = this.#allocator.malloc(size2);
        if (initialData) {
          if (initialData.byteLength !== size2) throw new Error("initialData size does not match buffer size");
          new Uint8Array(this.#memory.buffer, ptr, size2).set(initialData);
        }
        const slot = this.#nextSlot++;
        this.#buffers.set(slot, {
          ptr,
          size: size2,
          ref: 1
        });
        return slot;
      }
      incRef(slot) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        buffer.ref++;
      }
      decRef(slot) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        buffer.ref--;
        if (buffer.ref === 0) {
          this.#allocator.free(buffer.ptr);
          this.#buffers.delete(slot);
          this.#pendingWork.delete(slot);
        }
      }
      async read(slot, start, count) {
        const epoch = this.#pendingWork.get(slot);
        if (epoch) await this.#workerPool.waitForEpoch(epoch);
        return this.#readData(slot, start, count);
      }
      readSync(slot, start, count) {
        const epoch = this.#pendingWork.get(slot);
        if (epoch && this.#workerPool.epoch < epoch) throw new Error("cannot read synchronously from a slot with async work");
        return this.#readData(slot, start, count);
      }
      #readData(slot, start, count) {
        const buffer = this.#getBuffer(slot);
        if (start === void 0) start = 0;
        if (count === void 0) count = buffer.byteLength - start;
        if (hasSharedArrayBuffer() && buffer.buffer instanceof SharedArrayBuffer) return new Uint8Array(buffer.slice(start, start + count));
        else return buffer.slice(start, start + count);
      }
      async prepareKernel(kernel) {
        const kernelHash = FpHash.hash(kernel).toString();
        return new Executable(kernel, {
          program: await runWithCacheAsync(compiledProgramCache, kernelHash, async () => {
            const { bytes, ...metadata } = codegenWasm(kernel);
            return {
              kernelHash,
              module: await WebAssembly.compile(bytes),
              ...metadata
            };
          }),
          sync: false
        });
      }
      prepareKernelSync(kernel) {
        const kernelHash = FpHash.hash(kernel).toString();
        return new Executable(kernel, {
          program: runWithCache(compiledProgramCache, kernelHash, () => {
            const { bytes, ...metadata } = codegenWasm(kernel);
            return {
              kernelHash,
              module: new WebAssembly.Module(bytes),
              ...metadata
            };
          }),
          sync: true
        });
      }
      async prepareRoutine(routine) {
        return new Executable(routine, {
          program: void 0,
          sync: false
        });
      }
      prepareRoutineSync(routine) {
        return new Executable(routine, {
          program: void 0,
          sync: true
        });
      }
      dispatch(exe, inputs, outputs) {
        const tracing = isTracing();
        const start = tracing ? performance.now() : 0;
        const { sync } = exe.data;
        if (exe.source instanceof Routine) if (this.#workerPool && !sync) {
          const routine = {
            name: exe.source.name,
            type: exe.source.type,
            params: exe.source.params
          };
          const retainedSlots = [...inputs, ...outputs];
          for (const slot of retainedSlots) this.incRef(slot);
          const epoch = this.#workerPool.dispatchRoutine(routine, inputs.map((slot) => {
            const { ptr, size: size2 } = this.#buffers.get(slot);
            return {
              ptr,
              size: size2
            };
          }), outputs.map((slot) => {
            const { ptr, size: size2 } = this.#buffers.get(slot);
            return {
              ptr,
              size: size2
            };
          }));
          for (const slot of outputs) this.#pendingWork.set(slot, epoch);
          this.#workerPool.waitForEpoch(epoch).then(() => {
            for (const slot of outputs) if (this.#pendingWork.get(slot) === epoch) this.#pendingWork.delete(slot);
            for (const slot of retainedSlots) this.decRef(slot);
          });
        } else {
          if (inputs.some((slot) => {
            const epoch = this.#pendingWork.get(slot);
            return epoch && this.#workerPool.epoch < epoch;
          })) throw new Error("cannot dispatch wasm routine synchronously with pending async work");
          runCpuRoutine(exe.source, inputs.map((slot) => this.#getBuffer(slot)), outputs.map((slot) => this.#getBuffer(slot)));
        }
        else {
          const { program } = exe.data;
          const ptrs = [...inputs, ...outputs].map((slot) => this.#buffers.get(slot).ptr);
          if (this.#workerPool && !sync) {
            const retainedSlots = [...inputs, ...outputs];
            for (const slot of retainedSlots) this.incRef(slot);
            const epoch = this.#workerPool.dispatch(program.kernelHash, program.module, ptrs, program.workSize, program.chunkAlignment, program.minWorkPerWorker);
            for (const slot of outputs) this.#pendingWork.set(slot, epoch);
            this.#workerPool.waitForEpoch(epoch).then(() => {
              for (const slot of outputs) if (this.#pendingWork.get(slot) === epoch) this.#pendingWork.delete(slot);
              for (const slot of retainedSlots) this.decRef(slot);
            });
          } else {
            if (inputs.some((slot) => {
              const epoch = this.#pendingWork.get(slot);
              return epoch && this.#workerPool.epoch < epoch;
            })) throw new Error("cannot dispatch synchronously with pending async work");
            const func = new WebAssembly.Instance(program.module, { env: { memory: this.#memory } }).exports.kernel;
            func(...ptrs, 0, program.workSize);
          }
        }
        if (tracing) emitTrace("wasm", traceSourceInfo(exe.source), start, performance.now());
      }
      #getBuffer(slot) {
        const buffer = this.#buffers.get(slot);
        if (!buffer) throw new SlotError(slot);
        return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
      }
    };
    devices = [
      "cpu",
      "wasm",
      "webgpu",
      "webgl"
    ];
    initializedBackends = /* @__PURE__ */ new Map();
    initializedBackends.set("cpu", new CpuBackend());
    if (typeof WebAssembly !== "undefined") initializedBackends.set("wasm", new WasmBackend());
    defaultBackend = initializedBackends.has("wasm") ? "wasm" : "cpu";
    Executable = class {
      source;
      data;
      constructor(source, data) {
        this.source = source;
        this.data = data;
      }
    };
    SlotError = class extends Error {
      constructor(slot) {
        super(`Used a buffer that is invalid or already freed: ${slot}`);
      }
    };
    UnsupportedOpError = class extends Error {
      constructor(op, dtype, device, arg) {
        let msg = `${op || ""}<${dtype}> not supported in ${device} backend`;
        if (arg !== void 0) msg += ` with arg ${JSON.stringify(arg)}`;
        super(msg);
      }
    };
    UnsupportedRoutineError = class extends Error {
      constructor(name, device) {
        super(`routine '${name}' is not supported in ${device} backend`);
      }
    };
  }
});

// node_modules/@jax-js/jax/dist/rolldown-runtime-D7D4PA-g.js
var __defProp2 = Object.defineProperty;
var __exportAll = (all2, no_symbols) => {
  let target = {};
  for (var name in all2) __defProp2(target, name, {
    get: all2[name],
    enumerable: true
  });
  if (!no_symbols) __defProp2(target, Symbol.toStringTag, { value: "Module" });
  return target;
};

// node_modules/@jax-js/jax/dist/index.js
init_backend_D_Uwkp6d();
function checkConvShape(lhsShape, rhsShape, { vmapDims, strides, padding, lhsDilation, rhsDilation }) {
  if (lhsShape.length !== rhsShape.length) throw new Error(`conv() requires inputs with the same number of dimensions, got ${lhsShape.length} and ${rhsShape.length}`);
  const n = lhsShape.length - 2 - vmapDims;
  if (n < 0) throw new Error("conv() requires at least 2D inputs");
  if (strides.length !== n) throw new Error("conv() strides != spatial dims");
  if (padding.length !== n) throw new Error("conv() padding != spatial dims");
  if (lhsDilation.length !== n) throw new Error("conv() lhsDilation != spatial dimensions");
  if (rhsDilation.length !== n) throw new Error("conv() rhsDilation != spatial dimensions");
  if (lhsShape[vmapDims + 1] !== rhsShape[vmapDims + 1]) throw new Error(`conv() input channels: ${lhsShape[1]} != ${rhsShape[1]}`);
  const outShape = [
    ...generalBroadcast(lhsShape.slice(0, vmapDims), rhsShape.slice(0, vmapDims)),
    lhsShape[vmapDims],
    rhsShape[vmapDims]
  ];
  for (let i = 0; i < n; i++) {
    if (strides[i] <= 0 || !Number.isInteger(strides[i])) throw new Error(`conv() strides[${i}] must be a positive integer`);
    if (padding[i].length !== 2 || !padding[i].every(Number.isInteger)) throw new Error(`conv() padding[${i}] must be a 2-tuple of integers`);
    if (lhsDilation[i] <= 0 || !Number.isInteger(lhsDilation[i])) throw new Error(`conv() lhsDilation[${i}] must be a positive integer`);
    if (rhsDilation[i] <= 0 || !Number.isInteger(rhsDilation[i])) throw new Error(`conv() rhsDilation[${i}] must be a positive integer`);
    const [x, k] = [lhsShape[i + vmapDims + 2], rhsShape[i + vmapDims + 2]];
    if (k <= 0) throw new Error("conv() kernel size must be positive");
    const [pl, pr] = padding[i];
    if (pl < -x || pr < -x || pl + pr < -x) throw new Error(`conv() padding[${i}]=(${pl},${pr}) is too negative for input size ${x}`);
    const kernelSize = (k - 1) * rhsDilation[i] + 1;
    const inSize = Math.max((x - 1) * lhsDilation[i] + 1, 0) + pl + pr;
    if (kernelSize > inSize) throw new Error(`conv() kernel size ${kernelSize} > input size ${inSize} in dimension ${i}`);
    outShape.push(Math.ceil((inSize - kernelSize + 1) / strides[i]));
  }
  return outShape;
}
function checkPoolShape(inShape, window, strides) {
  if (strides.length !== window.length) throw new Error("pool() strides != window dims");
  if (window.length > inShape.length) throw new Error("pool() window has more dimensions than input");
  const outShape = inShape.slice(0, inShape.length - window.length);
  for (let i = 0; i < window.length; i++) {
    const k = window[i];
    const s = strides[i];
    const size2 = inShape[inShape.length - window.length + i];
    if (k <= 0 || !Number.isInteger(k)) throw new Error(`pool() window[${i}] must be a positive integer`);
    if (k > size2) throw new Error(`pool() window[${i}]=${k} > input size ${size2}`);
    if (s <= 0 || !Number.isInteger(s)) throw new Error(`pool() strides[${i}] must be a positive integer`);
    outShape.push(Math.ceil((size2 - k + 1) / s));
  }
  return outShape.concat(window);
}
function pool(st, ks, strides = 1, dilation = 1) {
  if (ks.length === 0) return st;
  if (st.shape.length < ks.length) throw new Error("pool() called with too many dimensions");
  if (typeof strides === "number") strides = rep(ks.length, strides);
  if (typeof dilation === "number") dilation = rep(ks.length, dilation);
  if (strides.some((s) => s <= 0 || !Number.isInteger(s))) throw new Error("pool() strides must be positive integers");
  if (dilation.some((d) => d <= 0 || !Number.isInteger(d))) throw new Error("pool() dilation must be positive integers");
  const noop = st.shape.slice(0, -ks.length);
  const i_ = st.shape.slice(-ks.length);
  const s_ = strides;
  const d_ = dilation;
  const o_ = zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
  if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
    st = st.padOrShrink([...noop.map(() => [0, 0]), ...zipn(i_, o_, s_).map(([i, o, s]) => [0, o * s - i])]);
    st = st.reshape([...noop, ...zip(o_, s_).flatMap(([o, s]) => [o, s])]).shrink([...noop.map((x) => [0, x]), ...zip(o_, ks).flatMap(([o, k]) => [[0, o], [0, k]])]);
    st = st.permute([
      ...range(noop.length),
      ...ks.map((_, j) => noop.length + 2 * j),
      ...ks.map((_, j) => noop.length + 2 * j + 1)
    ]);
    return st;
  }
  const kidf = zipn(ks, i_, d_, zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1))));
  st = st.repeat([...rep(noop.length, 1), ...kidf.map(([k, i, d, f]) => Math.ceil(k * (i * f + d) / i))]);
  st = st.shrink([...noop.map((x) => [0, x]), ...kidf.map(([k, i, d, f]) => [0, k * (i * f + d)])]).reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [k, i * f + d])]);
  const kos = zipn(ks, o_, s_);
  st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o, s]) => [[0, k], [0, o * s]])]).reshape([...noop, ...kos.flat(1)]);
  st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o]) => [
    [0, k],
    [0, o],
    [0, 1]
  ])]).reshape([...noop, ...kos.flatMap(([k, o]) => [k, o])]);
  st = st.permute([
    ...range(noop.length),
    ...ks.map((_, j) => noop.length + 2 * j + 1),
    ...ks.map((_, j) => noop.length + 2 * j)
  ]);
  return st;
}
function poolTranspose(st, inShape, ks, strides = 1, dilation = 1) {
  if (ks.length === 0) return st;
  if (typeof strides === "number") strides = rep(ks.length, strides);
  if (typeof dilation === "number") dilation = rep(ks.length, dilation);
  const noop = inShape.slice(0, -ks.length);
  const i_ = inShape.slice(-ks.length);
  const s_ = strides;
  const d_ = dilation;
  const o_ = zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
  if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
    st = st.permute([...range(noop.length), ...ks.flatMap((_, j) => [noop.length + j, noop.length + o_.length + j])]);
    st = st.pad([...noop.map(() => [0, 0]), ...zip(s_, ks).flatMap(([s, k]) => [[0, 0], [0, s - k]])]).reshape([...noop, ...zip(o_, s_).map(([o, s]) => o * s)]);
    st = st.padOrShrink([...noop.map(() => [0, 0]), ...zipn(i_, o_, s_).map(([i, o, s]) => [0, i - o * s])]);
    return st.reshape(st.shape.concat(rep(ks.length, 1)));
  }
  if (!deepEqual(o_, st.shape.slice(noop.length, noop.length + ks.length))) throw new Error("poolTranspose() called with mismatched output shape");
  const kidf = zipn(ks, i_, d_, zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1))));
  const kos = zipn(ks, o_, s_);
  st = st.permute([...range(noop.length), ...ks.flatMap((_, j) => [noop.length + ks.length + j, noop.length + j])]);
  st = st.reshape([...noop, ...kos.flatMap(([k, o]) => [
    k,
    o,
    1
  ])]).pad([...noop.map(() => [0, 0]), ...s_.flatMap((s) => [
    [0, 0],
    [0, 0],
    [0, s - 1]
  ])]);
  st = st.reshape([...noop, ...kos.flatMap(([k, o, s]) => [k, o * s])]).pad([...noop.map(() => [0, 0]), ...kidf.flatMap(([_k, i, d, f], j) => [[0, 0], [0, i * f + d - o_[j] * s_[j]]])]);
  st = st.reshape([...noop, ...kidf.map(([k, i, d, f]) => k * (i * f + d))]).pad([...noop.map(() => [0, 0]), ...kidf.map(([k, i, d, f]) => [0, Math.ceil(k * (i * f + d) / i) * i - k * (i * f + d)])]);
  st = st.reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [Math.ceil(k * (i * f + d) / i), i])]).permute([
    ...range(noop.length),
    ...ks.map((_, j) => noop.length + 2 * j + 1),
    ...ks.map((_, j) => noop.length + 2 * j)
  ]);
  return st;
}
function applyDilation(st, dilation) {
  if (dilation.every((s) => s === 1)) return st;
  const s_ = dilation;
  const n = s_.length;
  const prefix = st.shape.slice(0, -n);
  const k_ = st.shape.slice(-n);
  st = st.reshape([...prefix, ...k_.flatMap((k) => [k, 1])]);
  st = st.pad([...prefix.map(() => [0, 0]), ...s_.flatMap((s) => [[0, 0], [0, s - 1]])]);
  st = st.reshape([...prefix, ...k_.map((k, i) => k * s_[i])]);
  st = st.shrink([...prefix.map((p) => [0, p]), ...k_.map((k, i) => [0, (k - 1) * s_[i] + 1])]);
  return st;
}
function prepareConv(stX, stY, params) {
  const v = params.vmapDims;
  const n = stX.shape.length - 2 - v;
  const vmapShape = stX.shape.slice(0, v);
  stX = applyDilation(stX, params.lhsDilation);
  const ks = stY.shape.slice(v + 2);
  stX = stX.padOrShrink([...rep(v + 2, [0, 0]), ...params.padding]);
  stX = pool(stX, ks, params.strides, params.rhsDilation);
  stX = stX.moveaxis(v + 1, v + n + 1).reshape([
    ...vmapShape,
    stX.shape[v],
    1,
    ...stX.shape.slice(v + 2, v + n + 2),
    stX.shape[v + 1] * prod(ks)
  ]);
  stY = stY.reshape([
    ...vmapShape,
    1,
    stY.shape[v],
    ...rep(n, 1),
    stY.shape[v + 1] * prod(ks)
  ]);
  return [stX, stY];
}
var JsArray$4 = globalThis.Array;
var JsTreeDef = class JsTreeDef2 {
  nodeType;
  nodeMetadata;
  childTreedefs;
  static leaf = new JsTreeDef2("Leaf", null, []);
  constructor(nodeType, nodeMetadata, childTreedefs) {
    this.nodeType = nodeType;
    this.nodeMetadata = nodeMetadata;
    this.childTreedefs = childTreedefs;
  }
  /** Get the total number of leaves in the tree. */
  get size() {
    return this.nodeType === "Leaf" ? 1 : this.childTreedefs.reduce((a, b) => a + b.size, 0);
  }
  /** Returns a string representation of this tree definition. */
  toString(root = true) {
    if (root) return "JsTreeDef(" + this.toString(false) + ")";
    switch (this.nodeType) {
      case "Leaf":
        return "*";
      case "Array":
        return `[${this.childTreedefs.map((x) => x.toString(false)).join(", ")}]`;
      case "Object": {
        const parts = [];
        for (let i = 0; i < this.childTreedefs.length; i++) parts.push(`${quoteObjectKey(this.nodeMetadata[i])}: ${this.childTreedefs[i].toString(false)}`);
        return `{${parts.join(", ")}}`;
      }
    }
  }
  /** Compare this tree definition with another. */
  equals(other) {
    return this.nodeType === other.nodeType && deepEqual(this.nodeMetadata, other.nodeMetadata) && this.childTreedefs.length === other.childTreedefs.length && this.childTreedefs.every((x, i) => x.equals(other.childTreedefs[i]));
  }
};
function quoteObjectKey(key) {
  if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key)) return key;
  return JSON.stringify(key);
}
function flatten(tree) {
  const leaves = [];
  return [leaves, _flatten(tree, leaves)];
}
function _flatten(tree, leaves) {
  if (JsArray$4.isArray(tree)) return new JsTreeDef("Array", null, tree.map((c) => _flatten(c, leaves)));
  else if (typeof tree === "object" && tree !== null && tree.constructor === Object) {
    const [keys, values] = unzip2(Object.entries(tree));
    return new JsTreeDef("Object", keys, values.map((c) => _flatten(c, leaves)));
  } else {
    leaves.push(tree);
    return JsTreeDef.leaf;
  }
}
function unflatten(treedef, leaves) {
  return _unflatten(treedef, leaves[Symbol.iterator]());
}
function _unflatten(treedef, leaves) {
  switch (treedef.nodeType) {
    case "Leaf": {
      const { value, done } = leaves.next();
      if (done) throw new TypeError("Ran out of leaves while unflattening JsTree");
      return value;
    }
    case "Array":
      return treedef.childTreedefs.map((c) => _unflatten(c, leaves));
    case "Object": {
      const obj = {};
      for (let i = 0; i < treedef.childTreedefs.length; i++) obj[treedef.nodeMetadata[i]] = _unflatten(treedef.childTreedefs[i], leaves);
      return obj;
    }
  }
}
function map(fn, tree, ...rest) {
  const [leaves, treedef] = flatten(tree);
  const restLeaves = rest.map((x) => flatten(x)[0]);
  const resultLeaves = [];
  for (let i = 0; i < leaves.length; i++) resultLeaves.push(fn(leaves[i], ...restLeaves.map((x) => x[i])));
  return unflatten(treedef, resultLeaves);
}
var routinePrimitives = /* @__PURE__ */ new Map([
  ["sort", "Sort"],
  ["argsort", "Argsort"],
  ["scatter", "Scatter"],
  ["triangular_solve", "TriangularSolve"],
  ["cholesky", "Cholesky"],
  ["lu", "LU"],
  ["jacobi_eigh", "JacobiEigh"],
  ["fft", "Fft"]
]);
function add$1(x, y) {
  return bind1("add", [x, y]);
}
function mul(x, y) {
  return bind1("mul", [x, y]);
}
function idiv(x, y) {
  return bind1("idiv", [x, y]);
}
function mod(x, y) {
  return bind1("mod", [x, y]);
}
function min$1(x, y) {
  return bind1("min", [x, y]);
}
function max$1(x, y) {
  return bind1("max", [x, y]);
}
function bitCombine(x, y, op) {
  return bind1("bit_combine", [x, y], { op });
}
function bitShift(x, y, op) {
  return bind1("bit_shift", [x, y], { op });
}
function neg(x) {
  return bind1("neg", [x]);
}
function reciprocal$1(x) {
  return bind1("reciprocal", [x]);
}
function floor$1(x) {
  return bind1("floor", [x]);
}
function ceil$1(x) {
  return bind1("ceil", [x]);
}
function cast(x, dtype) {
  return bind1("cast", [x], { dtype });
}
function bitcast(x, dtype) {
  return bind1("bitcast", [x], { dtype });
}
function sin$1(x) {
  return bind1("sin", [x]);
}
function cos$1(x) {
  return bind1("cos", [x]);
}
function asin$1(x) {
  return bind1("asin", [x]);
}
function atan$1(x) {
  return bind1("atan", [x]);
}
function exp$1(x) {
  return bind1("exp", [x]);
}
function log$1(x) {
  return bind1("log", [x]);
}
function erf$1(x) {
  return bind1("erf", [x]);
}
function erfc$1(x) {
  return bind1("erfc", [x]);
}
function sqrt$1(x) {
  return bind1("sqrt", [x]);
}
function reduce(x, op, axis = null, opts) {
  if (!AluGroup.Reduce.has(op)) throw new TypeError(`Invalid reduce operation: ${op}`);
  axis = normalizeAxis(axis, ndim$1(x));
  const originalShape = getShape(x);
  let result = bind1("reduce", [x], {
    op,
    axis
  });
  if (opts?.keepdims) result = result.reshape(originalShape.map((dim, i) => axis.includes(i) ? 1 : dim));
  return result;
}
function dot$2(x, y) {
  return bind1("dot", [x, y]);
}
function conv$1(x, y, params = {}) {
  if (x.ndim !== y.ndim) throw new Error(`conv() requires inputs with the same number of dimensions, got ${x.ndim} and ${y.ndim}`);
  const vmapDims = params.vmapDims ?? 0;
  const n = x.ndim - 2 - vmapDims;
  if (n < 0) throw new Error("conv() requires at least 2D inputs");
  return bind1("conv", [x, y], {
    vmapDims,
    strides: params.strides ?? rep(n, 1),
    padding: params.padding ?? rep(n, [0, 0]),
    lhsDilation: params.lhsDilation ?? rep(n, 1),
    rhsDilation: params.rhsDilation ?? rep(n, 1)
  });
}
function compare(x, y, op) {
  return bind1("compare", [x, y], { op });
}
function greater$1(x, y) {
  return compare(y, x, "less");
}
function less$1(x, y) {
  return compare(x, y, "less");
}
function equal$1(x, y) {
  return compare(x, y, "equal");
}
function notEqual$1(x, y) {
  return compare(x, y, "not_equal");
}
function greaterEqual$1(x, y) {
  return compare(y, x, "less_equal");
}
function lessEqual$1(x, y) {
  return compare(x, y, "less_equal");
}
function where$1(cond, x, y) {
  return bind1("where", [
    cond,
    x,
    y
  ]);
}
function concatenate$1(xs, axis) {
  if (xs.length === 0) throw new Error("concatenate requires at least one input");
  const avals = xs.map((x) => ShapedArray.fromAval(getAval(x)));
  axis = checkAxis(axis, avals[0].ndim);
  for (const x of avals) if (x.ndim !== avals[0].ndim || !x.shape.every((s, i) => i === axis || s === avals[0].shape[i])) throw new Error(`Concatenate: inputs ${avals[0]} and ${x} must match shapes except on axis ${axis}`);
  return bind1("concatenate", xs, { axis });
}
function split$2(x, axis, sizes) {
  axis = checkAxis(axis, ndim$1(x));
  if (sizes.some((s) => s < 0 || !Number.isInteger(s))) throw new Error(`split: sizes must be nonnegative integers, got ${JSON.stringify(sizes)}`);
  const totalSize = sizes.reduce((a, b) => a + b, 0);
  if (totalSize !== getShape(x)[axis]) throw new Error(`split: sizes must sum to the size of the axis ${axis}, got ${totalSize}`);
  return bind("split", [x], {
    axis,
    sizes
  });
}
function randomBits(k0, k1, shape2, mode = "xor") {
  if (!deepEqual(k0.shape, k1.shape) || k0.dtype !== "uint32" || k1.dtype !== "uint32") throw new Error(`randomBits: key parts must be uint32 with the same shape, got ${ShapedArray.fromAval(k0.aval)} and ${ShapedArray.fromAval(k1.aval)}`);
  return bind1("random_bits", [k0, k1], {
    shape: shape2,
    mode
  });
}
function normalizeGatherIndexing(operandNdim, indexCount, axis, outDim) {
  if (indexCount === 0) throw new Error("gather() requires at least one index");
  if (!Array.isArray(axis) || axis.length !== indexCount) throw new Error(`Invalid gather() axis: expected ${indexCount} axes, got ${JSON.stringify(axis)}`);
  axis = axis.map((a) => checkAxis(a, operandNdim));
  if (new Set(axis).size !== axis.length) throw new Error(`Invalid gather() axis: duplicate axes ${JSON.stringify(axis)}`);
  return [axis, checkAxis(outDim, operandNdim - axis.length + 1)];
}
function gather(x, indices2, axis, outDim, uniqueIndices = false) {
  [axis, outDim] = normalizeGatherIndexing(ndim$1(x), indices2.length, axis, outDim);
  return bind1("gather", [x, ...indices2], {
    axis,
    outDim,
    uniqueIndices
  });
}
function transpose$1(x, perm) {
  perm = perm ? perm.map((a) => checkAxis(a, ndim$1(x))) : range(ndim$1(x)).reverse();
  if (!isPermutation(perm, ndim$1(x))) throw new Error(`Invalid transpose permutation for ${ndim$1(x)} axes: ${JSON.stringify(perm)}`);
  return bind1("transpose", [x], { perm });
}
function broadcast(x, shape2, axis) {
  axis = normalizeAxis(axis, shape2.length);
  return bind1("broadcast", [x], {
    shape: shape2,
    axis
  });
}
function reshape$1(x, shape2) {
  if (typeof shape2 === "number") shape2 = [shape2];
  const originalShape = getShape(x);
  const autoIdx = shape2.indexOf(-1);
  if (autoIdx !== -1) {
    const remaining = prod(originalShape) / -prod(shape2);
    if (!Number.isInteger(remaining) || remaining < 0) throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape2)}`);
    shape2 = shape2.toSpliced(autoIdx, 1, remaining);
  }
  if (prod(originalShape) !== prod(shape2)) throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape2)}`);
  return bind1("reshape", [x], { shape: shape2 });
}
function flip$1(x, axis) {
  axis = normalizeAxis(axis, ndim$1(x));
  return bind1("flip", [x], { axis });
}
function shrink(x, slice) {
  const shape2 = getShape(x);
  if (!Array.isArray(slice) || !slice.every(isNumberPair)) throw new Error(`Invalid shrink() type: ${JSON.stringify(slice)}`);
  if (slice.length !== shape2.length) throw new Error(`Invalid shrink(): expected ${shape2.length} axes, got ${slice.length}`);
  for (let i = 0; i < shape2.length; i++) {
    const [start, end] = slice[i];
    if (start > end || start < 0 || end > shape2[i]) throw new Error(`Invalid shrink() slice for axis ${i}: [${start}, ${end}] on shape ${shape2[i]}`);
  }
  return bind1("shrink", [x], { slice });
}
function pad$1(x, width) {
  const nd = ndim$1(x);
  let w;
  if (typeof width === "number") w = [[width, width]];
  else if (isNumberPair(width)) w = [width];
  else if (!Array.isArray(width)) {
    const indicesAndPairs = Object.entries(width);
    w = rep(nd, [0, 0]);
    for (const [k, v] of indicesAndPairs) w[checkAxis(parseInt(k), nd)] = v;
  } else if (!width.every(isNumberPair)) throw new TypeError(`Invalid pad() type: ${JSON.stringify(width)}`);
  else w = width;
  if (w.length === 1) {
    const [w0, w1] = w[0];
    w = rep(nd, () => [w0, w1]);
  } else if (w.length !== nd) throw new Error(`Invalid pad(): expected ${nd} axes, got ${w.length}`);
  return bind1("pad", [x], { width: w });
}
function triangularSolve$1(a, b, { lower = false, unitDiagonal = false } = {}) {
  const as = getShape(a);
  const bs = getShape(b);
  if (as.length < 2 || bs.length < 2) throw new Error(`triangular_solve: must be >=2D, got a=${as}, b=${bs}`);
  const n = as[as.length - 2];
  if (n !== as[as.length - 1] || n !== bs[bs.length - 1]) throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
  if (lower) {
    a = flip$1(a, [-2, -1]);
    b = flip$1(b, [-1]);
  }
  let x = bind1("triangular_solve", [a, b], { unitDiagonal });
  if (lower) x = flip$1(x, [-1]);
  return x;
}
function cholesky$2(x) {
  const aval = ShapedArray.fromAval(getAval(x));
  if (aval.ndim < 2 || aval.shape[aval.ndim - 1] !== aval.shape[aval.ndim - 2]) throw new Error(`cholesky: expected batch of square matrices, got ${aval}`);
  return bind1("cholesky", [x]);
}
function lu$1(x) {
  const aval = ShapedArray.fromAval(getAval(x));
  if (aval.ndim < 2) throw new Error(`lu: expected batch of matrices, got ${aval}`);
  return bind("lu", [x]);
}
function jacobiEigh(x, { maxSweeps, tolerance }) {
  const aval = ShapedArray.fromAval(getAval(x));
  if (aval.ndim < 2 || aval.shape[aval.ndim - 1] !== aval.shape[aval.ndim - 2]) throw new Error(`jacobi_eigh: expected batch of square matrices, got ${aval}`);
  if (!isFloatDtype(aval.dtype)) throw new TypeError(`jacobi_eigh: expected floating-point input, got ${aval}`);
  return bind("jacobi_eigh", [x], {
    maxSweeps,
    tolerance
  });
}
function fft$1(real, imag, params) {
  return bind("fft", [real, imag], params);
}
function sort$1(x) {
  if (ndim$1(x) === 0) throw new Error("sort: requires at least 1D input");
  return bind1("sort", [x]);
}
function argsort$1(x) {
  if (ndim$1(x) === 0) throw new Error("argsort: requires at least 1D input");
  return bind("argsort", [x]);
}
function scatter(updates, indices2, params) {
  return bind1("scatter", [updates, ...indices2], params);
}
function bind1(prim, args, params = {}) {
  const [results] = bind(prim, args, params);
  return results;
}
var traceStack = [];
var dynamicTrace = null;
function newMain(traceType, globalData = null) {
  const main2 = {
    level: traceStack.length,
    traceType,
    globalData
  };
  traceStack.push(main2);
  return Object.assign(main2, { [Symbol.dispose]() {
    traceStack.pop();
  } });
}
function newDynamic(main2) {
  const prevDynamicTrace = dynamicTrace;
  dynamicTrace = main2;
  return { [Symbol.dispose]() {
    dynamicTrace = prevDynamicTrace;
  } };
}
var Trace = class {
  main;
  constructor(main2) {
    this.main = main2;
  }
};
function promoteAvals(a, b) {
  const shape2 = generalBroadcast(a.shape, b.shape);
  const weakType = a.weakType && b.weakType;
  let dtype;
  if (a.weakType === b.weakType) dtype = promoteTypes(a.dtype, b.dtype);
  else if (a.weakType) dtype = promoteTypes(b.dtype, "uint32");
  else dtype = promoteTypes(a.dtype, "uint32");
  return new ShapedArray(shape2, dtype, weakType);
}
var Tracer = class Tracer2 {
  /** @ignore */
  _trace;
  constructor(trace2) {
    this._trace = trace2;
  }
  /** The shape of the array. */
  get shape() {
    return this.aval.shape;
  }
  /** The total number of elements in the array. */
  get size() {
    return prod(this.shape);
  }
  /** The dtype of elements stored in the array. */
  get dtype() {
    return this.aval.dtype;
  }
  /**
  * Whether the array is weakly typed.
  *
  * Weakly typed arrays will cast to the dtype of the other operand. See
  * `promoteTypes()` for details.
  */
  get weakType() {
    return this.aval.weakType;
  }
  /** The number of dimensions of the array. */
  get ndim() {
    return this.shape.length;
  }
  /** @ignore */
  fullLower() {
    return this;
  }
  neg() {
    return neg(this);
  }
  add(other) {
    return add$1(this, other);
  }
  mul(other) {
    return mul(this, other);
  }
  mod(other) {
    return mod(this, other);
  }
  greater(other) {
    return greater$1(this, other);
  }
  less(other) {
    return less$1(this, other);
  }
  equal(other) {
    return equal$1(this, other);
  }
  notEqual(other) {
    return notEqual$1(this, other);
  }
  greaterEqual(other) {
    return greaterEqual$1(this, other);
  }
  lessEqual(other) {
    return lessEqual$1(this, other);
  }
  /** Sum of the elements of the array over a given axis, or axes. */
  sum(axis = null, opts) {
    return reduce(this, "Add", axis, opts);
  }
  /** Product of the array elements over a given axis. */
  prod(axis = null, opts) {
    return reduce(this, "Mul", axis, opts);
  }
  /** Compute the average of the array elements along the specified axis. */
  mean(axis = null, opts) {
    axis = normalizeAxis(axis, this.ndim);
    const n = axis.reduce((acc, a) => acc * this.shape[a], 1);
    if (n === 0) throw new Error("mean: cannot compute mean over zero-length axis");
    const originalDtype = this.dtype;
    const castDtype = promoteTypes(originalDtype, "float32");
    const out = reduce(this.astype(castDtype), "Add", axis, opts).mul(1 / n);
    return isFloatDtype(originalDtype) ? out.astype(originalDtype) : out;
  }
  /** Minimum of the elements of the array along a given axis. */
  min(axis = null, opts) {
    return reduce(this, "Min", axis, opts);
  }
  /** Maximum of the elements of the array along a given axis. */
  max(axis = null, opts) {
    return reduce(this, "Max", axis, opts);
  }
  /** Test whether all array elements along a given axis evaluate to true. */
  all(axis = null, opts) {
    return this.astype("bool").min(axis, opts);
  }
  /** Test whether any array element along a given axis evaluates to true. */
  any(axis = null, opts) {
    return this.astype("bool").max(axis, opts);
  }
  /** Permute the dimensions of an array. Defaults to reversing the axis order. */
  transpose(perm) {
    return transpose$1(this, perm);
  }
  /**
  * Give a new shape to an array without changing its data.
  *
  * One shape dimension can be -1. In this case, the value is inferred from the
  * length of the array and remaining dimensions.
  */
  reshape(shape2) {
    return reshape$1(this, shape2);
  }
  /** Copy the array and cast to a specified dtype. */
  astype(dtype) {
    if (this.dtype === dtype) return this;
    return cast(this, dtype);
  }
  /** Return a bitwise cast of the array, viewed as a new dtype. */
  view(dtype) {
    if (!dtype || dtype === this.dtype) return this;
    return bitcast(this, dtype);
  }
  /** Subtract an array from this one. */
  sub(other) {
    return this.add(neg(other));
  }
  /** Divide an array by this one. */
  div(other) {
    if (isFloatDtype(this.dtype)) return this.mul(reciprocal$1(other));
    return idiv(this, other);
  }
  /** Return specified diagonals. See `jax.numpy.diagonal` for full docs. */
  diagonal(offset = 0, axis1 = 0, axis2 = 1) {
    if (!Number.isInteger(offset)) throw new TypeError(`offset must be an integer, got ${offset}`);
    if (offset < 0) return this.diagonal(-offset, axis2, axis1);
    axis1 = checkAxis(axis1, this.ndim);
    axis2 = checkAxis(axis2, this.ndim);
    if (axis1 === axis2) throw new Error("axis1 and axis2 must not be equal");
    if (offset >= this.shape[axis2]) throw new Error("offset exceeds axis size");
    let ar = this;
    if (axis1 !== ar.ndim - 2 || axis2 !== ar.ndim - 1) {
      const perm = range(ar.ndim).filter((i) => i !== axis1 && i !== axis2).concat(axis1, axis2);
      ar = ar.transpose(perm);
    }
    const [n, m] = ar.shape.slice(-2);
    const diagSize = Math.min(n, m - offset);
    ar = ar.reshape([...ar.shape.slice(0, -2), n * m]);
    const npad = diagSize * (m + 1) - n * m;
    if (npad > 0) ar = pad$1(ar, [...rep(ar.ndim - 1, [0, 0]), [0, npad]]);
    else if (npad < 0) ar = shrink(ar, [...ar.shape.slice(0, -1), n * m + npad].map((x) => [0, x]));
    ar = ar.reshape([
      ...ar.shape.slice(0, -1),
      diagSize,
      m + 1
    ]);
    ar = shrink(ar, [...ar.shape.slice(0, -1).map((x) => [0, x]), [offset, offset + 1]]).reshape(ar.shape.slice(0, -1));
    return ar;
  }
  /** Flatten the array without changing its data. */
  flatten() {
    return this.reshape(-1);
  }
  /** Flatten the array without changing its data. */
  ravel() {
    return this.reshape(-1);
  }
  /**
  * Iterate over the first dimension of this array, returning slices.
  *
  * This can be used to destructure arrays. For example:
  *
  * ```js
  * let x = np.array([[1, 2], [3, 4]]);
  * let [a, b] = x;
  * console.log(a.js()); // [1, 2]
  * console.log(b.js()); // [3, 4]
  * ```
  */
  *[Symbol.iterator]() {
    if (this.ndim === 0) throw new Error("Cannot iterate over a scalar array");
    let residual = this;
    const subarrayShape = this.shape.slice(1);
    for (let i = 0; i < this.shape[0]; i++) {
      const lr = split$2(residual, 0, [1, residual.shape[0] - 1]);
      yield lr[0].reshape(subarrayShape);
      residual = lr[1];
    }
    residual.dispose();
  }
  /**
  * Return a sorted copy of an array in ascending order.
  *
  * See `jax.numpy.sort` for full docs.
  */
  sort(axis = -1) {
    axis = checkAxis(axis, this.ndim);
    if (this.shape[axis] <= 1) return this;
    if (axis === this.ndim - 1) return sort$1(this);
    const perm = range(this.ndim);
    perm.splice(axis, 1);
    perm.push(axis);
    return sort$1(this.transpose(perm)).transpose(invertPermutation(perm));
  }
  /**
  * Return the indices that would sort an array. Unlike `sort`, this is
  * guaranteed to be a stable sorting algorithm; it always returns the smaller
  * index first in event of ties.
  *
  * See `jax.numpy.argsort` for full docs.
  */
  argsort(axis = -1) {
    axis = checkAxis(axis, this.ndim);
    if (axis === this.ndim - 1) {
      const [y2, yi2] = argsort$1(this);
      y2.dispose();
      return yi2;
    }
    const perm = range(this.ndim);
    perm.splice(axis, 1);
    perm.push(axis);
    const [y, yi] = argsort$1(this.transpose(perm));
    y.dispose();
    return yi.transpose(invertPermutation(perm));
  }
  /**
  * Slice an array along one or more axes.
  *
  * This is the equivalent of slicing in Python, e.g. `x[1:3, 2, :, None]`. To
  * mimic this in JavaScript, we would write:
  *
  * ```js
  * x.slice([1, 3], 2, [], null);
  * ```
  *
  * The `slice` method accepts a variable number of arguments, each of which
  * can be a number, an empty array, a single-element array, a two-element
  * array, or `null`. The arguments are interpreted as follows:
  *
  * - A number `n` means to access the `n`-th element along that axis, removing
  *   that axis from the resulting shape.
  * - An empty array `[]` means to keep that axis as-is, like `:` in Python.
  * - A single-element array `[i]` means to start slicing from index `i`
  *   (inclusive) to the end of the axis, like `x[i:]`.
  * - A two-element array `[i, j]` means to slice from index `i` (inclusive)
  *   to index `j` (exclusive), like `x[i:j]`.
  * - `null` means to add a new axis at that position, like `np.newaxis`.
  *
  * Like in Python, negative indices are supported, which count from the end of
  * the axis. For example, `-1` means the last element.
  *
  * Strided slices are not yet implemented, so you cannot write `x[::2]` or
  * similar.
  *
  * Advanced indexing by integer arrays is also supported. This translates to
  * the "gather" primitive, and it allows you to access specific elements of
  * the array by integer indices stored in another array.
  */
  slice(...index) {
    const checkBounds = (n, i) => {
      if (i > n || i < -n) throw new RangeError(`Index ${i} out of bounds for axis of size ${n}`);
      return i < 0 ? n + i : i;
    };
    const hasAdvancedIdx = index.some((value) => value instanceof Tracer2);
    const axesForGather = [];
    let outDim = -1;
    if (hasAdvancedIdx) {
      const advancedAxes = [];
      let currentAxisForGather = 0;
      for (let i = 0; i < index.length; i++) {
        const value = index[i];
        if (value instanceof Tracer2) {
          advancedAxes.push(i);
          axesForGather.push(currentAxisForGather++);
        } else if (typeof value === "number") advancedAxes.push(i);
        else currentAxisForGather++;
      }
      if (advancedAxes[advancedAxes.length - 1] - advancedAxes[0] !== advancedAxes.length - 1) outDim = 0;
      else outDim = axesForGather[0];
    }
    const slice = [];
    const basicShape = [];
    let needsReshape = false;
    let axis = 0;
    for (const value of index) if (value === null) {
      basicShape.push(1);
      needsReshape = true;
    } else if (typeof value === "number") {
      if (axis >= this.shape.length) throw new RangeError("Too many indices");
      const i = checkBounds(this.shape[axis++], value);
      slice.push([i, i + 1]);
      needsReshape = true;
    } else if (Array.isArray(value)) {
      if (axis >= this.shape.length) throw new RangeError("Too many indices");
      const n = this.shape[axis++];
      if (value.length === 0) {
        basicShape.push(n);
        slice.push([0, n]);
      } else if (value.length === 1) {
        const i = checkBounds(n, value[0]);
        basicShape.push(n - i);
        slice.push([i, n]);
      } else if (value.length === 2) {
        const [i, j] = value.map((v) => checkBounds(n, v));
        if (i > j) throw new RangeError(`Slice start at ${i} > end at ${j}`);
        basicShape.push(j - i);
        slice.push([i, j]);
      }
    } else if (value instanceof Tracer2) {
      const n = this.shape[axis++];
      basicShape.push(n);
      slice.push([0, n]);
    } else throw new TypeError(`Invalid slice argument: ${JSON.stringify(value)}`);
    while (axis < this.shape.length) {
      slice.push([0, this.shape[axis]]);
      basicShape.push(this.shape[axis++]);
    }
    let result = shrink(this, slice);
    result = needsReshape ? reshape$1(result, basicShape) : result;
    if (hasAdvancedIdx) result = gather(result, index.filter((a) => a instanceof Tracer2), axesForGather, outDim);
    return result;
  }
};
function ndim$1(x) {
  if (x instanceof Tracer) return x.shape.length;
  else return 0;
}
function getShape(x) {
  return x instanceof Tracer ? x.shape : [];
}
var ShapedArray = class ShapedArray2 {
  shape;
  dtype;
  weakType;
  constructor(shape2, dtype, weakType) {
    this.shape = shape2;
    this.dtype = dtype;
    this.weakType = weakType;
  }
  static fromAval(aval) {
    return new ShapedArray2(aval.shape, aval.dtype, aval.weakType);
  }
  get ndim() {
    return this.shape.length;
  }
  get size() {
    return prod(this.shape);
  }
  scalar() {
    return new ShapedArray2([], this.dtype, this.weakType);
  }
  toString() {
    return `${this.dtype}[${this.shape.join(",")}]`;
  }
  equals(other) {
    return this === other || this.constructor === other.constructor && this.ndim === other.ndim && this.shape.every((d, i) => d === other.shape[i]);
  }
};
function getAval(x) {
  if (x instanceof Tracer) return x.aval;
  else if (typeof x === "boolean" || typeof x === "number") return new ShapedArray([], typeof x === "boolean" ? "bool" : "float32", typeof x === "boolean" ? false : true);
  else throw new TypeError(`Unknown value: ${x}`);
}
function bind(prim, args, params = {}) {
  const topTrace = findTopTrace(args);
  const tracers = args.map((arg) => fullRaise(topTrace, arg));
  const outs = topTrace.processPrimitive(prim, tracers, params);
  if (DEBUG >= 5) console.info(`processing rule for ${prim} on ${tracers.map((x) => x.toString())} and got ${outs.map((x) => x.toString())}`);
  return outs.map((out) => out.fullLower());
}
function findTopTrace(xs) {
  let topMain = traceStack[0];
  for (const x of xs) if (x instanceof Tracer && x._trace.main.level > topMain.level) topMain = x._trace.main;
  if (dynamicTrace && dynamicTrace.level > topMain.level) topMain = dynamicTrace;
  return new topMain.traceType(topMain);
}
function fullRaise(trace2, val) {
  if (!(val instanceof Tracer)) return trace2.pure(val);
  const level = trace2.main.level;
  if (Object.is(val._trace.main, trace2.main)) return val;
  else if (val._trace.main.level < level) return trace2.lift(val);
  else if (val._trace.main.level > level) throw new Error(`Can't lift Tracer level ${val._trace.main.level} to level ${level}`);
  else throw new Error(`Different traces at same level: ${val._trace.constructor}, ${trace2.constructor}.`);
}
var TreeMismatchError = class extends TypeError {
  constructor(where2, left, right) {
    super(`Mismatched tree structures in ${where2}: ${left} != ${right}`);
  }
};
function flattenFun(f, inTree) {
  const store = { value: void 0 };
  const flatFun = (...argsFlat) => {
    const [outFlat, outTree] = flatten(f(...unflatten(inTree, argsFlat)));
    store.value = outTree;
    return outFlat;
  };
  return [flatFun, store];
}
var UseAfterFreeError = class extends ReferenceError {
  constructor(tracer) {
    super(`Referenced tracer ${tracer.toString()} freed, please use .ref move semantics`);
  }
};
function _usingCtx() {
  var r = "function" == typeof SuppressedError ? SuppressedError : function(r2, e3) {
    var n2 = Error();
    return n2.name = "SuppressedError", n2.error = r2, n2.suppressed = e3, n2;
  }, e2 = {}, n = [];
  function using(r2, e3) {
    if (null != e3) {
      if (Object(e3) !== e3) throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
      if (r2) var o = e3[Symbol.asyncDispose || Symbol["for"]("Symbol.asyncDispose")];
      if (void 0 === o && (o = e3[Symbol.dispose || Symbol["for"]("Symbol.dispose")], r2)) var t = o;
      if ("function" != typeof o) throw new TypeError("Object is not disposable.");
      t && (o = function o2() {
        try {
          t.call(e3);
        } catch (r3) {
          return Promise.reject(r3);
        }
      }), n.push({
        v: e3,
        d: o,
        a: r2
      });
    } else r2 && n.push({
      d: e3,
      a: r2
    });
    return e3;
  }
  return {
    e: e2,
    u: using.bind(null, false),
    a: using.bind(null, true),
    d: function d() {
      var o, t = this.e, s = 0;
      function next() {
        for (; o = n.pop(); ) try {
          if (!o.a && 1 === s) return s = 0, n.push(o), Promise.resolve().then(next);
          if (o.d) {
            var r2 = o.d.call(o.v);
            if (o.a) return s |= 2, Promise.resolve(r2).then(next, err);
          } else s |= 1;
        } catch (r3) {
          return err(r3);
        }
        if (1 === s) return t !== e2 ? Promise.reject(t) : Promise.resolve();
        if (t !== e2) throw t;
      }
      function err(n2) {
        return t = t !== e2 ? new r(n2, t) : n2, next();
      }
      return next();
    }
  };
}
var Var = class Var2 {
  static #nextId = 1;
  id;
  aval;
  constructor(aval) {
    this.id = Var2.#nextId++;
    this.aval = aval;
  }
  toString() {
    return `Var(${this.id}):${this.aval.toString()}`;
  }
};
var Lit = class {
  value;
  aval;
  get dtype() {
    return this.aval.dtype;
  }
  constructor(aval, value) {
    if (aval.shape.length !== 0) throw new Error(`internal: Lit must be a scalar`);
    this.value = value;
    this.aval = ShapedArray.fromAval(aval);
  }
};
function atomIsLit(atom, literal) {
  return atom instanceof Lit && (literal === void 0 || atom.value === literal);
}
var VarPrinter = class {
  names = /* @__PURE__ */ new Map();
  #next = "a";
  #advance() {
    const ret = this.#next;
    let lastNonz = this.#next.length - 1;
    while (lastNonz >= 0 && this.#next[lastNonz] === "z") lastNonz--;
    if (lastNonz < 0) this.#next = "a".repeat(this.#next.length + 1);
    else {
      let result = this.#next.slice(0, lastNonz);
      result += String.fromCharCode(this.#next.charCodeAt(lastNonz) + 1);
      result += "a".repeat(this.#next.length - 1 - lastNonz);
      this.#next = result;
    }
    return ret;
  }
  name(v) {
    if (this.names.has(v)) return this.names.get(v);
    const name = this.#advance();
    this.names.set(v, name);
    return name;
  }
  nameType(v) {
    return `${this.name(v)}:${v.aval.toString()}`;
  }
};
var JaxprEqn = class {
  primitive;
  inputs;
  params;
  outBinders;
  constructor(primitive, inputs, params, outBinders) {
    this.primitive = primitive;
    this.inputs = inputs;
    this.params = params;
    this.outBinders = outBinders;
  }
  pprint(usedVars, vp = new VarPrinter()) {
    const lhs = PPrint.pp(this.outBinders.map((v) => !usedVars || usedVars.has(v) ? vp.nameType(v) : "_").join(" "));
    let rhs = PPrint.pp(this.primitive);
    const paramsList = Object.entries(this.params).map(([k, v]) => PPrint.pp(`${k}=${v}`));
    if (paramsList.length > 0) rhs = rhs.stack(PPrint.pp(" [ ")).stack(PPrint.prototype.concat(...paramsList)).stack(PPrint.pp(" ] "));
    else rhs = rhs.stack(PPrint.pp(" "));
    rhs = rhs.stack(PPrint.pp(this.inputs.map((x) => x instanceof Var ? vp.name(x) : String(x.value)).join(" ")));
    return lhs.stack(PPrint.pp(" = ")).stack(rhs);
  }
  toString() {
    return this.pprint().toString();
  }
};
var Jaxpr = class Jaxpr2 {
  inBinders;
  eqns;
  outs;
  #hash;
  constructor(inBinders, eqns, outs) {
    this.inBinders = inBinders;
    this.eqns = eqns;
    this.outs = outs;
  }
  pprint() {
    const vp = new VarPrinter();
    const usedVars = new Set([...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter((x) => x instanceof Var));
    const inBinders = this.inBinders.map((v) => vp.nameType(v)).join(", ");
    const eqns = PPrint.prototype.concat(...this.eqns.map((e2) => e2.pprint(usedVars, vp)));
    const outs = this.outs.map((x) => x instanceof Var ? vp.name(x) : x.value).join(", ");
    return PPrint.pp(`{ lambda ${inBinders} .`).concat((this.eqns.length ? PPrint.pp("let ").stack(eqns).concat(PPrint.pp(`in ( ${outs} ) }`)) : PPrint.pp(`( ${outs} ) }`)).indent(2));
  }
  toString() {
    return this.pprint().toString();
  }
  /**
  * Gets a hash of this Jaxpr.
  *
  * Var identity is not considered in the hash, so two Jaxprs with the same
  * order of assignments and operators but different variable IDs will resolve
  * to the same hash (and toString representation).
  */
  getHash() {
    if (this.#hash !== void 0) return this.#hash;
    const hasher = new FpHash();
    const varIds = /* @__PURE__ */ new Map();
    const vi = (v) => {
      if (varIds.has(v)) return varIds.get(v);
      const id = varIds.size + 1;
      varIds.set(v, FpHash.hash(id, v.aval.dtype, ...v.aval.shape));
      return id;
    };
    hasher.update(this.inBinders.length);
    for (const x of this.inBinders) hasher.update(vi(x));
    hasher.update(this.eqns.length);
    for (const eqn of this.eqns) {
      hasher.update(eqn.primitive);
      hasher.update(eqn.inputs.length);
      for (const x of eqn.inputs) hasher.update(x instanceof Var ? vi(x) : x.value);
      hasher.update(JSON.stringify(eqn.params));
      hasher.update(eqn.outBinders.length);
      for (const x of eqn.outBinders) hasher.update(vi(x));
    }
    hasher.update(this.outs.length);
    for (const x of this.outs) hasher.update(x instanceof Var ? vi(x) : x.value);
    return this.#hash = hasher.value;
  }
  hash(state) {
    state.update(this.getHash());
  }
  /**
  * Produce a simplified Jaxpr with basic optimizations applied.
  *  - Trim away unused variables.
  *  - Fold away *1, *0, or +0 operations against literals.
  *  - Remove no-op movement operations.
  */
  simplify() {
    const context = /* @__PURE__ */ new Map();
    const newEqns = [];
    for (const e2 of this.eqns) {
      const inputs = e2.inputs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
      const eqn = new JaxprEqn(e2.primitive, inputs, e2.params, e2.outBinders);
      if (eqn.primitive === "add") {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a, 0)) context.set(c, b);
        else if (atomIsLit(b, 0)) context.set(c, a);
        else if (atomIsLit(a) && atomIsLit(b)) context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.dtype === "bool" ? Math.min(a.value + b.value, 1) : a.value + b.value));
        else newEqns.push(eqn);
      } else if (eqn.primitive === "neg") {
        const [a] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a)) context.set(c, new Lit(a.aval, -a.value));
        else newEqns.push(eqn);
      } else if (eqn.primitive === "mul") {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a, 1)) context.set(c, b);
        else if (atomIsLit(b, 1)) context.set(c, a);
        else if (atomIsLit(a) && atomIsLit(b)) context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.value * b.value));
        else newEqns.push(eqn);
      } else if (eqn.primitive === "idiv") {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(b, 1) && !isFloatDtype(a.aval.dtype)) context.set(c, a);
        else newEqns.push(eqn);
      } else if ((eqn.primitive === "broadcast" || eqn.primitive === "reshape") && deepEqual(eqn.params.shape, eqn.inputs[0].aval.shape) || eqn.primitive === "transpose" && eqn.params.perm.every((p, i) => p === i) || eqn.primitive === "flip" && eqn.params.axis.length === 0 || eqn.primitive === "shrink" && eqn.params.slice.every(([s, e3], i) => s === 0 && e3 === eqn.inputs[0].aval.shape[i]) || eqn.primitive === "pad" && eqn.params.width.every(([w0, w1]) => w0 === 0 && w1 === 0)) context.set(eqn.outBinders[0], eqn.inputs[0]);
      else newEqns.push(eqn);
    }
    const outs = this.outs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
    const usedVars = new Set(outs.filter((x) => x instanceof Var));
    const liveEqns = [];
    for (let i = newEqns.length - 1; i >= 0; i--) {
      const eqn = newEqns[i];
      if (eqn.outBinders.some((v) => usedVars.has(v))) {
        liveEqns.push(eqn);
        for (const v of eqn.inputs) if (v instanceof Var) usedVars.add(v);
      }
    }
    return new Jaxpr2(this.inBinders, liveEqns.reverse(), outs);
  }
  /** Flattens nested Jit in a Jaxpr. Useful for handling jit-of-jit. */
  flatten() {
    if (!this.eqns.some((eqn) => eqn.primitive === "jit")) return this;
    const newEqns = [];
    const varMap = /* @__PURE__ */ new Map();
    const varMapF = (x) => x instanceof Var ? varMap.get(x) ?? x : x;
    for (const eqn of this.eqns) if (eqn.primitive === "jit") {
      const jaxpr = eqn.params.jaxpr.flatten();
      const translation = /* @__PURE__ */ new Map();
      const translationF = (x) => x instanceof Var ? translation.get(x) : x;
      for (const [v, x] of zip(jaxpr.inBinders, eqn.inputs)) translation.set(v, varMapF(x));
      for (const ieqn of jaxpr.eqns) {
        const inputs = ieqn.inputs.map(translationF);
        const outBinders = [];
        for (const v of ieqn.outBinders) {
          const u = new Var(v.aval);
          outBinders.push(u);
          translation.set(v, u);
        }
        newEqns.push(new JaxprEqn(ieqn.primitive, inputs, ieqn.params, outBinders));
      }
      for (const [v, x] of zip(eqn.outBinders, jaxpr.outs)) varMap.set(v, translationF(x));
    } else if (eqn.inputs.some((x) => x instanceof Var && varMap.has(x))) newEqns.push(new JaxprEqn(eqn.primitive, eqn.inputs.map(varMapF), eqn.params, eqn.outBinders));
    else newEqns.push(eqn);
    const newOuts = this.outs.map(varMapF);
    return new Jaxpr2(this.inBinders, newEqns, newOuts);
  }
};
var JaxprType = class {
  inTypes;
  outTypes;
  constructor(inTypes, outTypes) {
    this.inTypes = inTypes;
    this.outTypes = outTypes;
  }
  toString() {
    return `(${this.inTypes.map((aval) => aval.toString()).join(", ")}) -> (${this.outTypes.map((aval) => aval.toString()).join(", ")})`;
  }
};
function typecheckJaxpr(jaxpr) {
  const env = /* @__PURE__ */ new Set();
  for (const v of jaxpr.inBinders) {
    if (env.has(v)) throw new TypeError(`Duplicate variable binding: ${v}`);
    env.add(v);
  }
  for (const eqn of jaxpr.eqns) {
    const inTypes = eqn.inputs.map((x) => typecheckAtom(env, x));
    const rule = abstractEvalRules[eqn.primitive];
    const outTypes = rule(inTypes, eqn.params);
    for (const [outBinder, outType] of zip(eqn.outBinders, outTypes)) {
      if (!outType.equals(outBinder.aval)) throw new TypeError(`Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`);
      if (env.has(outBinder)) throw new TypeError(`Duplicate variable binding: ${outBinder}`);
      env.add(outBinder);
    }
  }
  return new JaxprType(jaxpr.inBinders.map((v) => v.aval), jaxpr.outs.map((x) => typecheckAtom(env, x)));
}
function typecheckAtom(env, x) {
  if (x instanceof Var) {
    if (!env.has(x)) throw new Error(`Unknown variable: ${x}`);
    return x.aval;
  } else if (x instanceof Lit) return x.aval;
  else throw new TypeError(`Invalid atom type: ${x}`);
}
function evalJaxpr(jaxpr, args) {
  const env = /* @__PURE__ */ new Map();
  const usageCount = /* @__PURE__ */ new Map();
  for (const x of jaxpr.eqns.flatMap((eqn) => eqn.inputs).concat(jaxpr.outs)) if (x instanceof Var) usageCount.set(x, (usageCount.get(x) ?? 0) + 1);
  const remainingRefs = /* @__PURE__ */ new Map();
  const read = (x) => {
    if (x instanceof Var) {
      remainingRefs.set(x, (remainingRefs.get(x) ?? 0) - 1);
      return env.get(x);
    } else return array(x.value, { dtype: x.dtype });
  };
  const write = (v, val) => {
    if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
    let refCount = usageCount.get(v) ?? 0;
    if (refCount) {
      env.set(v, val);
      remainingRefs.set(v, refCount);
      while (refCount-- > 1) val.ref;
    } else val.dispose();
  };
  try {
    for (const [v, arg] of zip(jaxpr.inBinders, args)) write(v, arg);
    for (const eqn of jaxpr.eqns) {
      const inVals = eqn.inputs.map(read);
      const outVals = bind(eqn.primitive, inVals, eqn.params);
      for (const [v, val] of zip(eqn.outBinders, outVals)) write(v, val);
    }
    return jaxpr.outs.map(read);
  } catch (error) {
    for (let [v, refCount] of remainingRefs.entries()) if (refCount > 0) {
      const tracer = env.get(v);
      while (refCount--) tracer.dispose();
    }
    throw error;
  }
}
function jaxprAsFun(jaxpr) {
  return (...args) => evalJaxpr(jaxpr, args);
}
var ClosedJaxpr = class ClosedJaxpr2 {
  jaxpr;
  consts;
  constructor(jaxpr, consts) {
    this.jaxpr = jaxpr;
    this.consts = consts;
  }
  /** String representation of this Jaxpr. */
  toString() {
    return this.jaxpr.toString();
  }
  /** Apply a function to the underlying Jaxpr. */
  mapJaxpr(f) {
    return new ClosedJaxpr2(f(this.jaxpr), this.consts);
  }
  /** Dispose of the constants in this Jaxpr. */
  dispose() {
    for (const c of this.consts) c.dispose();
  }
};
var JaxprTracer = class extends Tracer {
  aval;
  #rc;
  constructor(trace2, aval) {
    super(trace2);
    this.aval = aval;
    this.#rc = 1;
  }
  toString() {
    return `JaxprTracer(${this.aval.toString()})`;
  }
  get ref() {
    if (this.#rc <= 0) throw new UseAfterFreeError(this);
    this.#rc++;
    return this;
  }
  dispose() {
    if (this.#rc <= 0) throw new UseAfterFreeError(this);
    this.#rc--;
  }
  trackLiftedConstant() {
    this.#rc++;
  }
};
var JaxprTrace = class extends Trace {
  /** Register a Jaxpr argument with a given shape and return the tracer. */
  newArg(aval) {
    aval = ShapedArray.fromAval(aval);
    const tracer = this.builder.newTracer(this, aval);
    this.builder.addVar(tracer);
    return tracer;
  }
  /** Register a constant / literal in this Jaxpr. */
  getOrMakeConstTracer(val) {
    if (!(val instanceof Tracer)) val = pureArray(val);
    let tracer = this.builder.constTracers.get(val);
    if (tracer === void 0) {
      tracer = this.builder.newTracer(this, ShapedArray.fromAval(getAval(val)));
      this.builder.addConst(tracer, val);
    } else {
      val.dispose();
      tracer.trackLiftedConstant();
    }
    return tracer;
  }
  pure = this.getOrMakeConstTracer;
  lift = this.getOrMakeConstTracer;
  processPrimitive(primitive, tracers, params) {
    const avalsIn = tracers.map((t) => {
      t.dispose();
      return t.aval;
    });
    const outTracers = abstractEvalRules[primitive](avalsIn, params).map((aval) => this.builder.newTracer(this, aval));
    this.builder.addEqn(new JaxprEqn(primitive, tracers.map((t) => this.builder.getVar(t)), params, outTracers.map((t) => this.builder.addVar(t))));
    return outTracers;
  }
  get builder() {
    return this.main.globalData;
  }
};
var JaxprBuilder = class {
  eqns = [];
  tracerToVar = /* @__PURE__ */ new Map();
  constTracers = /* @__PURE__ */ new Map();
  constVals = /* @__PURE__ */ new Map();
  tracers = [];
  newTracer(trace2, aval) {
    const tracer = new JaxprTracer(trace2, aval);
    this.tracers.push(tracer);
    return tracer;
  }
  addEqn(eqn) {
    this.eqns.push(eqn);
  }
  addVar(tracer) {
    if (this.tracerToVar.has(tracer)) throw new Error(`Tracer was added as variable twice: ${tracer}`);
    const v = new Var(tracer.aval);
    this.tracerToVar.set(tracer, v);
    return v;
  }
  getVar(tracer) {
    const v = this.tracerToVar.get(tracer);
    if (v === void 0) throw new Error(`Could not find variable for tracer: ${tracer}`);
    return v;
  }
  addConst(tracer, val) {
    const v = this.addVar(tracer);
    this.constTracers.set(val, tracer);
    this.constVals.set(v, val);
    return v;
  }
  build(inTracers, outTracers) {
    const [constVars, consts] = unzip2(this.constVals.entries());
    const t2v = this.getVar.bind(this);
    const inBinders = [...constVars, ...inTracers.map(t2v)];
    const outVars = outTracers.map(t2v);
    const jaxpr = new Jaxpr(inBinders, this.eqns, outVars);
    typecheckJaxpr(jaxpr);
    return _inlineLiterals(new ClosedJaxpr(jaxpr, consts));
  }
};
function _inlineLiterals({ jaxpr, consts }) {
  const literals = /* @__PURE__ */ new Map();
  const constBinders = [];
  const newConsts = [];
  for (let i = 0; i < consts.length; i++) if (ndim$1(consts[i]) === 0 && consts[i] instanceof Array$1) {
    const ar = consts[i];
    literals.set(jaxpr.inBinders[i], new Lit(ar.aval, ar.dataSync()[0]));
  } else {
    constBinders.push(jaxpr.inBinders[i]);
    newConsts.push(consts[i]);
  }
  const newEqns = jaxpr.eqns.map((eqn) => new JaxprEqn(eqn.primitive, eqn.inputs.map((x) => literals.get(x) ?? x), eqn.params, eqn.outBinders));
  const newOuts = jaxpr.outs.map((x) => literals.get(x) ?? x);
  const newJaxpr = new Jaxpr([...constBinders, ...jaxpr.inBinders.slice(consts.length)], newEqns, newOuts);
  typecheckJaxpr(newJaxpr);
  return new ClosedJaxpr(newJaxpr, newConsts);
}
function binopAbstractEval([x, y]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  return [promoteAvals(x, y)];
}
function compareAbstractEval([x, y]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) throw new TypeError("compareAbstractEval expects ShapedArray inputs");
  return [new ShapedArray(promoteAvals(x, y).shape, "bool", false)];
}
function vectorizedUnopAbstractEval([x]) {
  return [ShapedArray.fromAval(x)];
}
function gatherShape(operandShape, indices2, { axis, outDim }) {
  for (const a of indices2) if (a.dtype !== "int32" && a.dtype !== "uint32") throw new TypeError(`indices must be Int32 or Uint32, got ${a.dtype}`);
  if (axis.length !== indices2.length) throw new TypeError(`got ${axis} axes but ${indices2.length} indices`);
  if (indices2.length === 0) throw new TypeError(`must have at least one index`);
  if (axis.some((a) => a < 0 || a >= operandShape.length)) throw new TypeError(`axis out of bounds`);
  if (outDim < 0 || outDim > operandShape.length - axis.length) throw new TypeError(`outDim out of bounds`);
  const axisSet = new Set(axis);
  if (axisSet.size !== axis.length) throw new TypeError(`axes are not unique`);
  const indexShape = indices2.reduce((result2, a) => generalBroadcast(result2, a.shape), []);
  const result = operandShape.filter((_, i) => !axisSet.has(i));
  result.splice(outDim, 0, ...indexShape);
  return result;
}
var abstractEvalRules = {
  ["add"]: binopAbstractEval,
  ["mul"]: binopAbstractEval,
  ["idiv"]: binopAbstractEval,
  ["mod"]: binopAbstractEval,
  ["min"]: binopAbstractEval,
  ["max"]: binopAbstractEval,
  ["bit_combine"]([x, y]) {
    const aval = promoteAvals(x, y);
    if (isFloatDtype(aval.dtype)) throw new TypeError(`bitwise operations require integer or boolean inputs, got ${aval.dtype}`);
    return [aval];
  },
  ["bit_shift"]([x, y]) {
    const shape2 = generalBroadcast(x.shape, y.shape);
    if (isFloatDtype(x.dtype) || isFloatDtype(y.dtype) || x.dtype === "bool" || y.dtype === "bool") throw new TypeError(`bit shift operations require integer inputs, got ${x} and ${y}`);
    return [new ShapedArray(shape2, x.dtype, x.weakType)];
  },
  ["neg"]: vectorizedUnopAbstractEval,
  ["reciprocal"]: vectorizedUnopAbstractEval,
  ["floor"]: vectorizedUnopAbstractEval,
  ["ceil"]: vectorizedUnopAbstractEval,
  ["stop_gradient"]: vectorizedUnopAbstractEval,
  ["cast"]([x], { dtype }) {
    return [new ShapedArray(x.shape, dtype, false)];
  },
  ["bitcast"]([x], { dtype }) {
    if (x.dtype !== dtype && (x.dtype === "bool" || dtype === "bool")) throw new TypeError("Bitcast to/from bool is not allowed");
    if (byteWidth(x.dtype) !== byteWidth(dtype)) throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
    return [new ShapedArray(x.shape, dtype, false)];
  },
  ["sin"]: vectorizedUnopAbstractEval,
  ["cos"]: vectorizedUnopAbstractEval,
  ["asin"]: vectorizedUnopAbstractEval,
  ["atan"]: vectorizedUnopAbstractEval,
  ["exp"]: vectorizedUnopAbstractEval,
  ["log"]: vectorizedUnopAbstractEval,
  ["erf"]: vectorizedUnopAbstractEval,
  ["erfc"]: vectorizedUnopAbstractEval,
  ["sqrt"]: vectorizedUnopAbstractEval,
  ["reduce"]([x], { axis }) {
    const axisSet = new Set(axis);
    return [new ShapedArray(x.shape.filter((_, i) => !axisSet.has(i)), x.dtype, x.weakType)];
  },
  ["pool"]([x], { window, strides }) {
    return [new ShapedArray(checkPoolShape(x.shape, window, strides), x.dtype, x.weakType)];
  },
  ["pool_transpose"]([x], { inShape, window, strides }) {
    const shape2 = checkPoolShape(inShape, window, strides);
    if (!deepEqual(shape2, x.shape)) throw new TypeError(`PoolTranspose shape mismatch: expected ${JSON.stringify(shape2)}, got ${JSON.stringify(x.shape)}`);
    return [new ShapedArray(inShape, x.dtype, x.weakType)];
  },
  ["dot"]([x, y]) {
    if (x.ndim === 0 && y.ndim === 0) throw new TypeError("Dot requires at least 1D inputs");
    const { shape: shape2, dtype, weakType } = promoteAvals(x, y);
    shape2.splice(-1, 1);
    return [new ShapedArray(shape2, dtype, weakType)];
  },
  ["conv"]([lhs, rhs], params) {
    const { dtype, weakType } = promoteAvals(lhs.scalar(), rhs.scalar());
    return [new ShapedArray(checkConvShape(lhs.shape, rhs.shape, params), dtype, weakType)];
  },
  ["compare"]: compareAbstractEval,
  ["where"]([cond, x, y]) {
    if (cond.dtype !== "bool") throw new TypeError(`Condition must be boolean, got ${cond.dtype}`);
    const xy = promoteAvals(x, y);
    return [new ShapedArray(generalBroadcast(cond.shape, xy.shape), xy.dtype, xy.weakType)];
  },
  ["concatenate"](xs, { axis }) {
    if (xs.length === 0) throw new TypeError("Concatenate requires at least one input");
    for (const x of xs) if (x.ndim !== xs[0].ndim || !x.shape.every((s, i) => i === axis || s === xs[0].shape[i])) throw new TypeError(`Concatenate: inputs ${xs[0]} and ${x} must match shapes except on axis ${axis}`);
    const shape2 = xs[0].shape.slice();
    shape2[axis] = xs.reduce((sum2, x) => sum2 + x.shape[axis], 0);
    const { dtype, weakType } = xs.map((x) => x.scalar()).reduce(promoteAvals);
    return [new ShapedArray(shape2, dtype, weakType)];
  },
  ["split"]([x], { axis, sizes }) {
    const totalSize = sizes.reduce((a, b) => a + b, 0);
    if (x.shape[axis] !== totalSize) throw new TypeError(`Split: sizes ${sizes} do not sum to dimension ${x.shape[axis]} on axis ${axis}`);
    return sizes.map((size2) => {
      return new ShapedArray(x.shape.toSpliced(axis, 1, size2), x.dtype, x.weakType);
    });
  },
  ["random_bits"]([k0, k1], { shape: shape2 }) {
    if (k0.dtype !== "uint32" || k1.dtype !== "uint32") throw new TypeError(`RandomBits requires uint32 keys, got ${k0.dtype} and ${k1.dtype}`);
    if (!deepEqual(k0.shape, k1.shape)) throw new TypeError(`RandomBits: Keys have different shapes ${k0.shape} and ${k1.shape}`);
    if (!deepEqual(shape2.slice(0, k0.ndim), k0.shape)) throw new TypeError(`RandomBits: generated shape ${shape2} must match key shape ${k0.shape}`);
    return [new ShapedArray(shape2, "uint32", false)];
  },
  ["gather"]([x, ...indices2], { axis, outDim }) {
    return [new ShapedArray(gatherShape(x.shape, indices2, {
      axis,
      outDim
    }), x.dtype, x.weakType)];
  },
  ["transpose"]([x], { perm }) {
    return [new ShapedArray(perm.map((i) => x.shape[i]), x.dtype, x.weakType)];
  },
  ["broadcast"]([x], { shape: shape2 }) {
    return [new ShapedArray(shape2, x.dtype, x.weakType)];
  },
  ["reshape"]([x], { shape: shape2 }) {
    return [new ShapedArray(shape2, x.dtype, x.weakType)];
  },
  ["flip"]([x], _) {
    return [ShapedArray.fromAval(x)];
  },
  ["shrink"]([x], { slice }) {
    return [new ShapedArray(slice.map((s) => s[1] - s[0]), x.dtype, x.weakType)];
  },
  ["pad"]([x], { width }) {
    return [new ShapedArray(x.shape.map((dim, i) => dim + width[i][0] + width[i][1]), x.dtype, x.weakType)];
  },
  ["sort"]([x]) {
    if (x.ndim === 0) throw new TypeError("sort: requires at least 1D input");
    return [ShapedArray.fromAval(x)];
  },
  ["argsort"]([x]) {
    if (x.ndim === 0) throw new TypeError("argsort: requires at least 1D input");
    return [ShapedArray.fromAval(x), new ShapedArray(x.shape, "int32", false)];
  },
  ["scatter"]([updates, ...indices2], { shape: shape2, axis, outDim }) {
    const expectedUpdatesShape = gatherShape(shape2, indices2, {
      axis,
      outDim
    });
    if (!deepEqual(updates.shape, expectedUpdatesShape)) throw new TypeError(`Scatter updates shape ${updates.shape} does not match expected shape ${expectedUpdatesShape}`);
    return [new ShapedArray(shape2, updates.dtype, updates.weakType)];
  },
  ["triangular_solve"]([a, b]) {
    if (a.ndim < 2) throw new TypeError(`triangular_solve: a must be at least 2D, got ${a}`);
    if (b.ndim < 2) throw new TypeError(`triangular_solve: b must be at least 2D, got ${b}`);
    const [m, n] = a.shape.slice(-2);
    const [_batch, q] = b.shape.slice(-2);
    if (!deepEqual(a.shape.slice(0, -2), b.shape.slice(0, -2)) || a.dtype !== b.dtype || m !== n || n !== q) throw new TypeError(`triangular_solve: mismatch ${a} vs ${b}`);
    return [new ShapedArray(b.shape, b.dtype, a.weakType && b.weakType)];
  },
  ["cholesky"]([a]) {
    if (a.ndim < 2) throw new TypeError(`cholesky: requires at least 2D input, got ${a}`);
    if (a.shape[a.ndim - 2] !== a.shape[a.ndim - 1]) throw new TypeError(`cholesky: must be square, got ${a}`);
    return [ShapedArray.fromAval(a)];
  },
  ["lu"]([a]) {
    if (a.ndim < 2) throw new TypeError(`lu: requires at least 2D input, got ${a}`);
    const batch = a.shape.slice(0, -2);
    const [m, n] = a.shape.slice(-2);
    return [
      ShapedArray.fromAval(a),
      new ShapedArray([...batch, Math.min(m, n)], "int32", false),
      new ShapedArray([...batch, m], "int32", false)
    ];
  },
  ["jacobi_eigh"]([a]) {
    if (a.ndim < 2) throw new TypeError(`jacobi_eigh: requires at least 2D input, got ${a}`);
    if (a.shape[a.ndim - 2] !== a.shape[a.ndim - 1]) throw new TypeError(`jacobi_eigh: must be square, got ${a}`);
    if (!isFloatDtype(a.dtype)) throw new TypeError(`jacobi_eigh: requires floating-point input, got ${a}`);
    return [ShapedArray.fromAval(a), ShapedArray.fromAval(a)];
  },
  ["fft"]([real, imag], { factors }) {
    if (!real.equals(imag)) throw new TypeError(`fft: real and imag arrays must match, got ${real} and ${imag}`);
    if (real.ndim < 1) throw new TypeError(`fft: requires at least 1D input, got ${real}`);
    if (!isFloatDtype(real.dtype)) throw new TypeError(`fft: requires floating-point input, got ${real}`);
    const n = real.shape[real.ndim - 1];
    if (prod(factors) !== n) throw new TypeError(`fft: factorization ${factors} does not match size ${n}`);
    return [ShapedArray.fromAval(real), ShapedArray.fromAval(real)];
  },
  ["jit"](args, { jaxpr }) {
    const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
    if (args.length !== inTypes.length) throw new TypeError(`jit expected ${inTypes.length} arguments, got ${args.length}`);
    for (let i = 0; i < inTypes.length; i++) if (!args[i].equals(inTypes[i])) throw new TypeError(`jit argument ${i} has type ${args[i]}, expected ${inTypes[i]}`);
    return outTypes;
  }
};
function splitIdx(values, argnums) {
  const a = [];
  const b = [];
  for (let i = 0; i < values.length; i++) if (argnums.has(i)) a.push(values[i]);
  else b.push(values[i]);
  return [a, b];
}
function joinIdx(n, a, b, argnums) {
  const result = [];
  let ai = 0;
  let bi = 0;
  for (let i = 0; i < n; i++) if (argnums.has(i)) result.push(a[ai++]);
  else result.push(b[bi++]);
  return result;
}
function makeJaxpr$1(f, opts) {
  return (...argsIn) => {
    try {
      var _usingCtx$3 = _usingCtx();
      const staticArgnums = new Set(opts?.staticArgnums ?? []);
      const [staticArgs, shapedArgs] = splitIdx(argsIn, staticArgnums);
      const [avalsIn, inTree] = flatten(shapedArgs);
      const [fFlat, outTree] = flattenFun((...dynamicArgs) => {
        return f(...joinIdx(argsIn.length, staticArgs, dynamicArgs, staticArgnums));
      }, inTree);
      const builder = new JaxprBuilder();
      const main2 = _usingCtx$3.u(newMain(JaxprTrace, builder));
      _usingCtx$3.u(newDynamic(main2));
      const trace2 = new JaxprTrace(main2);
      const tracersIn = avalsIn.map((aval) => trace2.newArg(typeof aval === "object" ? aval : pureArray(aval)));
      const tracersOut = fFlat(...tracersIn).map((out) => fullRaise(trace2, out));
      const jaxpr = builder.build(tracersIn, tracersOut);
      if (outTree.value === void 0) throw new Error("outTree was not set in makeJaxpr");
      return {
        jaxpr: jaxpr.mapJaxpr((j) => j.simplify()),
        treedef: outTree.value
      };
    } catch (_) {
      _usingCtx$3.e = _;
    } finally {
      _usingCtx$3.d();
    }
  };
}
function jit$1(f, opts) {
  const cache = /* @__PURE__ */ new Map();
  const staticArgnums = new Set(opts?.staticArgnums ?? []);
  const result = ((...args) => {
    const [staticArgs, dynamicArgs] = splitIdx(args, staticArgnums);
    const [argsFlat, inTree] = flatten(dynamicArgs);
    const avalsIn = unflatten(inTree, argsFlat.map((x) => ShapedArray.fromAval(getAval(x))));
    const jaxprArgs = joinIdx(args.length, staticArgs, avalsIn, staticArgnums);
    const { jaxpr, treedef: outTree } = runWithCache(cache, jaxprArgs, () => makeJaxpr$1(f, opts)(...jaxprArgs));
    return unflatten(outTree, bind("jit", [...jaxpr.consts.map((c) => c.ref), ...argsFlat], {
      name: f.name || "closure",
      jaxpr: jaxpr.jaxpr,
      numConsts: jaxpr.consts.length
    }));
  });
  result.dispose = () => {
    for (const { jaxpr } of cache.values()) jaxpr.dispose();
  };
  return result;
}
var JitProgram = class {
  backend;
  steps;
  inputs;
  outputs;
  constructor(backend, steps, inputs, outputs) {
    this.backend = backend;
    this.steps = steps;
    this.inputs = inputs;
    this.outputs = outputs;
  }
  pprint() {
    const steps = this.steps.map((step) => {
      switch (step.type) {
        case "execute": {
          const executeText = `execute (${step.inputs.map((id, i) => `${i}: %${id}`).join(", ")}) -> ${step.outputs.map((id) => `%${id}`).join(", ")}`;
          if (step.source instanceof Kernel) return PPrint.pp(`${executeText}, kernel`).concat(step.source.pprint().indent(2));
          else if (step.source instanceof Routine) return PPrint.pp(`${executeText}, routine ${step.source.name}`);
          else {
            step.source;
            return PPrint.pp(executeText);
          }
        }
        case "malloc":
          return PPrint.pp(`%${step.output} = malloc <${step.size} bytes>`);
        case "incref":
          return PPrint.pp(`incref ${step.input}`);
        case "free":
          return PPrint.pp(`free ${step.input}`);
      }
    });
    const display = PPrint.prototype.concat(PPrint.pp(`device = ${this.backend.type}`), PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"), PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"), PPrint.pp("steps ="), PPrint.prototype.concat(...steps).indent(2));
    return PPrint.pp("{ ").stack(display.stack(PPrint.pp(" }")));
  }
  toString() {
    return this.pprint().toString();
  }
  /** Execute the JitProgram with the given inputs. */
  execute(inputs) {
    const scope = /* @__PURE__ */ new Map();
    if (inputs.length !== this.inputs.length) throw new TypeError(`Expected ${this.inputs.length} inputs, got ${inputs.length}`);
    for (const [i, id] of this.inputs.entries()) scope.set(id, inputs[i]);
    const pending = [];
    for (const step of this.steps) switch (step.type) {
      case "execute": {
        const inputs2 = step.inputs.map((id) => scope.get(id));
        const outputs = step.outputs.map((id) => scope.get(id));
        if (inputs2.some((s) => s === void 0) || outputs.some((s) => s === void 0)) throw new Error(`internal: JitProgram scope undefined`);
        pending.push(new PendingExecute(this.backend, step.source, inputs2, outputs));
        break;
      }
      case "malloc": {
        const slot = this.backend.malloc(step.size);
        scope.set(step.output, slot);
        break;
      }
      case "incref": {
        const slot = scope.get(step.input);
        this.backend.incRef(slot);
        break;
      }
      case "free": {
        const slot = scope.get(step.input);
        this.backend.decRef(slot);
        scope.delete(step.input);
        break;
      }
      default:
    }
    return {
      outputs: this.outputs.map((id) => scope.get(id)),
      pending
    };
  }
};
var JitProgramBuilder = class {
  backend;
  #nextId;
  steps;
  constructor(backend, nargs) {
    this.backend = backend;
    this.#nextId = nargs;
    this.steps = [];
  }
  pushLit(lit) {
    const kernel = new Kernel(0, lit.aval.size, AluExp.const(lit.dtype, lit.value));
    return this.pushKernel(kernel, []);
  }
  pushBuffer(size2) {
    const id = this.#nextId++;
    this.steps.push({
      type: "malloc",
      size: size2,
      output: id
    });
    return id;
  }
  pushKernel(kernel, inputs) {
    const id = this.pushBuffer(kernel.bytes);
    this.steps.push({
      type: "execute",
      source: kernel,
      inputs,
      outputs: [id]
    });
    return id;
  }
  pushRoutine(routine, inputs, outputs) {
    this.steps.push({
      type: "execute",
      source: routine,
      inputs,
      outputs
    });
  }
  pushIncref(id) {
    this.steps.push({
      type: "incref",
      input: id
    });
  }
  insertFreeSteps(outputIds) {
    const ids = this.steps.filter((s) => s.type === "malloc").map((s) => s.output);
    for (const id of ids) {
      if (outputIds.includes(id)) continue;
      const lastUsage = this.steps.findLastIndex((s) => s.type === "execute" && (s.outputs.includes(id) || s.inputs.includes(id)) || s.type === "malloc" && s.output === id);
      this.steps.splice(lastUsage + 1, 0, {
        type: "free",
        input: id
      });
    }
  }
  pushFree(id) {
    this.steps.push({
      type: "free",
      input: id
    });
  }
};
var jitCompileCache = /* @__PURE__ */ new Map();
function jitCompile(backend, jaxpr) {
  const cacheKey = backend.type + "," + FpHash.hash(jaxpr);
  const cached = jitCompileCache.get(cacheKey);
  if (cached) return cached;
  if (DEBUG >= 1) console.info("=========== JIT Compile ===========\n" + jaxpr.toString());
  jaxpr = jaxpr.flatten().simplify();
  const nargs = jaxpr.inBinders.length;
  const builder = new JitProgramBuilder(backend, nargs);
  const blackNodes = splitGraphDataflow(backend, jaxpr);
  const ctx = /* @__PURE__ */ new Map();
  for (let i = 0; i < nargs; i++) {
    const v = jaxpr.inBinders[i];
    ctx.set(v, {
      type: "imm",
      arg: i
    });
  }
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    if (routinePrimitives.has(eqn.primitive)) {
      const routine = new Routine(routinePrimitives.get(eqn.primitive), {
        inputShapes: eqn.inputs.map((x) => x.aval.shape),
        inputDtypes: eqn.inputs.map((x) => x.aval.dtype),
        outputShapes: eqn.outBinders.map((x) => x.aval.shape),
        outputDtypes: eqn.outBinders.map((x) => x.aval.dtype)
      }, eqn.params);
      const inputs = [];
      for (const input of eqn.inputs) if (input instanceof Var) {
        const jv = ctx.get(input);
        if (jv.type !== "imm") throw new Error(`jit: routine primitive ${eqn.primitive} input is not imm`);
        inputs.push(jv.arg);
      } else if (input instanceof Lit) inputs.push(builder.pushLit(input));
      const outputs = [];
      for (const outVar of eqn.outBinders) {
        const outId = builder.pushBuffer(outVar.aval.size * byteWidth(outVar.aval.dtype));
        outputs.push(outId);
        ctx.set(outVar, {
          type: "imm",
          arg: outId
        });
      }
      builder.pushRoutine(routine, inputs, outputs);
      continue;
    }
    const inputExps = [];
    const inputAvals = [];
    const inputArgs = [];
    let inputReduction = null;
    const addArgs = (args) => {
      const newGids = [];
      for (const jitId of args) {
        let newGid = inputArgs.indexOf(jitId);
        if (newGid === -1) {
          newGid = inputArgs.length;
          inputArgs.push(jitId);
        }
        newGids.push(newGid);
      }
      return newGids;
    };
    for (const input of eqn.inputs) if (input instanceof Var) {
      const jv = ctx.get(input);
      if (jv.type === "exp") {
        const newGids = addArgs(jv.args);
        inputExps.push(jv.exp.reindexGids(newGids));
      } else if (jv.type === "imm") {
        const [gid] = addArgs([jv.arg]);
        const st = ShapeTracker.fromShape(input.aval.shape);
        const indices2 = unravelAlu(st.shape, AluVar.gidx);
        inputExps.push(AluExp.globalView(input.aval.dtype, gid, st, indices2));
      } else if (jv.type === "red") {
        if (inputReduction) throw new Error("jit: unexpected, multiple red inputs");
        const newGids = addArgs(jv.args);
        inputExps.push(jv.reduction.epilogue.reindexGids(newGids));
        inputReduction = jv;
      }
      inputAvals.push(input.aval);
    } else if (input instanceof Lit) {
      inputExps.push(AluExp.const(input.dtype, input.value));
      inputAvals.push(input.aval);
    } else throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
    const rule = jitRules[eqn.primitive];
    if (!rule) throw new TypeError(`JIT not implemented for primitive ${eqn.primitive}`);
    let exp3;
    let reduction;
    if (inputReduction) {
      const jv = inputReduction;
      const newEpilogue = rule(inputExps, inputAvals, eqn.params).exp[0];
      exp3 = [jv.exp.reindexGids(addArgs(jv.args))];
      reduction = new Reduction(jv.reduction.dtype, jv.reduction.op, jv.reduction.size, newEpilogue);
    } else {
      const ruleOutput = rule(inputExps, inputAvals, eqn.params);
      exp3 = ruleOutput.exp;
      reduction = ruleOutput.reduction;
    }
    for (let i2 = 0; i2 < eqn.outBinders.length; i2++) {
      const outVar = eqn.outBinders[i2];
      if (blackNodes.has(outVar)) {
        const nargs2 = inputArgs.length;
        const size2 = outVar.aval.size;
        const kernel = new Kernel(nargs2, size2, exp3[i2], reduction);
        const outId = builder.pushKernel(kernel, inputArgs);
        ctx.set(outVar, {
          type: "imm",
          arg: outId
        });
      } else if (reduction) ctx.set(outVar, {
        type: "red",
        exp: exp3[i2],
        reduction,
        args: inputArgs
      });
      else ctx.set(outVar, {
        type: "exp",
        exp: exp3[i2],
        args: inputArgs
      });
    }
  }
  const outputIds = [];
  for (const out of jaxpr.outs) if (out instanceof Var) {
    const jitValue = ctx.get(out);
    if (jitValue.type !== "imm") throw new Error("internal: Expected imm, since outs are black nodes");
    outputIds.push(jitValue.arg);
  } else if (out instanceof Lit) outputIds.push(builder.pushLit(out));
  const outputNeedsRef = new Set(range(nargs));
  for (const outputId of outputIds) if (outputNeedsRef.has(outputId)) builder.pushIncref(outputId);
  else outputNeedsRef.add(outputId);
  builder.insertFreeSteps(outputIds);
  const jp = new JitProgram(backend, builder.steps, range(0, nargs), outputIds);
  if (DEBUG >= 4) console.info(jp.toString());
  jitCompileCache.set(cacheKey, jp);
  return jp;
}
function reshapeViews(exp3, mapping, reduceAxis = false) {
  return exp3.rewrite((exp4) => {
    if (exp4.op === "GlobalView") {
      const [gid, st] = exp4.arg;
      const newSt = mapping(st);
      if (newSt) {
        const indices2 = reduceAxis ? unravelAlu(newSt.shape.slice(0, -1), AluVar.gidx).concat(AluVar.ridx) : unravelAlu(newSt.shape, AluVar.gidx);
        return AluExp.globalView(exp4.dtype, gid, newSt, indices2);
      }
    } else if (exp4.op === "GlobalIndex") throw new Error("internal: reshapeViews() called with GlobalIndex op");
  });
}
function broadcastedJit(fn, opts) {
  return (exps, avals, params) => {
    let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);
    const skipCastIdx = opts?.skipCastIdx ?? [];
    if (skipCastIdx.length) newDtype = avals.filter((_, i) => !skipCastIdx.includes(i)).reduce(promoteAvals).dtype;
    exps = exps.map((exp3, i) => {
      exp3 = reshapeViews(exp3, (st) => {
        if (!deepEqual(st.shape, newShape)) return st.broadcast(newShape, range(newShape.length - st.shape.length));
      });
      if (exp3.dtype !== newDtype && !skipCastIdx.includes(i)) exp3 = AluExp.cast(newDtype, exp3);
      return exp3;
    });
    return { exp: [fn(exps, params)] };
  };
}
function unopJit(fn) {
  return ([a], [_as], params) => {
    return { exp: [fn(a, params)] };
  };
}
function reshapeJit(fn) {
  return ([a], [_as], params) => {
    return { exp: [reshapeViews(a, (st) => fn(st, params))] };
  };
}
function routineNoJit() {
  return () => {
    throw new Error("jit: rule is not implemented for routines");
  };
}
var jitRules = {
  ["add"]: broadcastedJit(([a, b]) => AluExp.add(a, b)),
  ["mul"]: broadcastedJit(([a, b]) => AluExp.mul(a, b)),
  ["idiv"]: broadcastedJit(([a, b]) => AluExp.idiv(a, b)),
  ["mod"]: broadcastedJit(([a, b]) => AluExp.mod(a, b)),
  ["min"]: broadcastedJit(([a, b]) => AluExp.min(a, b)),
  ["max"]: broadcastedJit(([a, b]) => AluExp.max(a, b)),
  ["bit_combine"]: broadcastedJit(([a, b], { op }) => AluExp.bitCombine(a, b, op)),
  ["bit_shift"]: broadcastedJit(([a, b], { op }) => AluExp.bitShift(a, b, op)),
  ["neg"]: unopJit((a) => AluExp.sub(AluExp.const(a.dtype, 0), a)),
  ["reciprocal"]: unopJit(AluExp.reciprocal),
  ["floor"]: unopJit(AluExp.floor),
  ["ceil"]: unopJit(AluExp.ceil),
  ["stop_gradient"]: unopJit((a) => a),
  ["cast"]: unopJit((a, { dtype }) => AluExp.cast(dtype, a)),
  ["bitcast"]: unopJit((a, { dtype }) => AluExp.bitcast(dtype, a)),
  ["sin"]: unopJit(AluExp.sin),
  ["cos"]: unopJit(AluExp.cos),
  ["asin"]: unopJit(AluExp.asin),
  ["atan"]: unopJit(AluExp.atan),
  ["exp"]: unopJit(AluExp.exp),
  ["log"]: unopJit(AluExp.log),
  ["erf"]: unopJit(AluExp.erf),
  ["erfc"]: unopJit(AluExp.erfc),
  ["sqrt"]: unopJit(AluExp.sqrt),
  ["reduce"]([a], [as], { op, axis }) {
    const keptAxes = [];
    const shiftedAxes = [];
    const newShape = [];
    for (let i = 0; i < as.shape.length; i++) if (axis.includes(i)) shiftedAxes.push(i);
    else {
      keptAxes.push(i);
      newShape.push(as.shape[i]);
    }
    const reductionSize = prod(shiftedAxes.map((ax) => as.shape[ax]));
    newShape.push(reductionSize);
    const perm = keptAxes.concat(shiftedAxes);
    a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
    const reduction = new Reduction(a.dtype, op, reductionSize);
    return {
      exp: [a],
      reduction
    };
  },
  ["pool"]: reshapeJit((st, { window, strides }) => pool(st, window, strides)),
  ["pool_transpose"]([a], [as], { inShape, window, strides }) {
    let stX = poolTranspose(ShapeTracker.fromShape(as.shape), inShape, window, strides);
    stX = stX.reshape([...inShape, prod(stX.shape.slice(inShape.length))]);
    a = reshapeViews(a, (st) => st.compose(stX), true);
    const reduction = new Reduction(a.dtype, "Add", stX.shape[stX.shape.length - 1]);
    return {
      exp: [a],
      reduction
    };
  },
  ["dot"]([a, b], [as, bs]) {
    const [c] = jitRules["mul"]([a, b], [as, bs], {}).exp;
    const cs = promoteAvals(as, bs);
    return jitRules["reduce"]([c], [cs], {
      op: "Add",
      axis: [cs.ndim - 1]
    });
  },
  ["conv"]([a, b], [as, bs], params) {
    const [stX, stY] = prepareConv(ShapeTracker.fromShape(as.shape), ShapeTracker.fromShape(bs.shape), params);
    a = reshapeViews(a, (st) => st.compose(stX));
    b = reshapeViews(b, (st) => st.compose(stY));
    as = new ShapedArray(stX.shape, as.dtype, as.weakType);
    bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
    return jitRules["dot"]([a, b], [as, bs], {});
  },
  ["compare"]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
  ["where"]: broadcastedJit(([cond, a, b]) => AluExp.where(cond, a, b), { skipCastIdx: [0] }),
  ["concatenate"](exps, avals, { axis }) {
    const ndim2 = avals[0].ndim;
    const sizes = avals.map((x) => x.shape[axis]);
    const finalSize = sizes.reduce((a, b) => a + b, 0);
    const { dtype: dtypeOut } = avals.map((x) => x.scalar()).reduce(promoteAvals);
    const makePadAxis = (start, end) => range(ndim2).map((i) => i === axis ? [start, end] : [0, 0]);
    let cum = 0;
    const src = [];
    for (let i = 0; i < exps.length; i++) {
      const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
      src.push(reshapeViews(AluExp.cast(dtypeOut, exps[i]), (st) => st.pad(padding)));
      cum += sizes[i];
    }
    return { exp: [src.reduce(AluExp.add)] };
  },
  ["split"]([a], [as], { axis, sizes }) {
    const exp3 = [];
    let start = 0;
    for (const size2 of sizes) {
      const slice = range(as.ndim).map((d) => d === axis ? [start, start + size2] : [0, as.shape[d]]);
      exp3.push(reshapeViews(a, (st) => st.shrink(slice)));
      start += size2;
    }
    return { exp: exp3 };
  },
  ["random_bits"]: (keys, keyShapes, { shape: shape2, mode }) => {
    const keyShape = keyShapes[0].shape;
    const mapping = (st) => {
      if (!deepEqual(st.shape, shape2)) return st.broadcast(shape2, range(st.shape.length, shape2.length));
    };
    const k0 = reshapeViews(keys[0], mapping);
    const k1 = reshapeViews(keys[1], mapping);
    const c0 = AluExp.u32(0);
    const c1 = AluExp.mod(AluExp.cast("uint32", AluVar.gidx), AluExp.u32(Math.max(prod(shape2.slice(keyShape.length)), 1)));
    return { exp: [AluExp.threefry2x32(k0, k1, c0, c1, mode)] };
  },
  ["gather"]([x, ...indices2], [xs, ...indicesShapes], { axis, outDim }) {
    const axisSet = new Set(axis);
    const indexShape = indicesShapes.map((c) => c.shape).reduce(generalBroadcast);
    const finalShape = xs.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);
    const idxNonaxis = [...unravelAlu(finalShape, AluVar.gidx)];
    idxNonaxis.splice(outDim, indexShape.length);
    const src = [...idxNonaxis];
    for (let i = 0; i < xs.shape.length; i++) if (axisSet.has(i)) src.splice(i, 0, null);
    for (const [i, iexp] of indices2.entries()) src[axis[i]] = AluExp.cast("int32", reshapeViews(iexp, (st) => st.broadcast(finalShape, [...range(outDim + indexShape.length - st.shape.length), ...range(outDim + indexShape.length, finalShape.length)])));
    const [index, valid] = ShapeTracker.fromShape(xs.shape).toAluExp(src);
    if (!valid.resolve()) throw new Error("internal: expected full validity mask in Gather");
    return { exp: [x.substitute({ gidx: index })] };
  },
  ["transpose"]: reshapeJit((st, { perm }) => st.permute(perm)),
  ["broadcast"]: reshapeJit((st, { shape: shape2, axis }) => st.broadcast(shape2, axis)),
  ["reshape"]: reshapeJit((st, { shape: shape2 }) => st.reshape(shape2)),
  ["flip"]: reshapeJit((st, { axis }) => {
    const arg = rep(st.shape.length, false);
    for (const ax of axis) arg[ax] = true;
    return st.flip(arg);
  }),
  ["shrink"]: reshapeJit((st, { slice }) => st.shrink(slice)),
  ["pad"]: reshapeJit((st, { width }) => st.pad(width)),
  ["sort"]: routineNoJit(),
  ["argsort"]: routineNoJit(),
  ["scatter"]: routineNoJit(),
  ["triangular_solve"]: routineNoJit(),
  ["cholesky"]: routineNoJit(),
  ["lu"]: routineNoJit(),
  ["jacobi_eigh"]: routineNoJit(),
  ["fft"]: routineNoJit(),
  ["jit"]() {
    throw new Error("internal: Jit should have been flattened before JIT compilation");
  }
};
function splitGraphDataflow(backend, jaxpr) {
  const varToDefn = /* @__PURE__ */ new Map();
  const varToUsages = /* @__PURE__ */ new Map();
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    for (const v of eqn.outBinders) if (v instanceof Var) varToDefn.set(v, i);
    for (const input of eqn.inputs) if (input instanceof Var) {
      const usages = varToUsages.get(input);
      if (usages) usages.push(i);
      else varToUsages.set(input, [i]);
    }
  }
  const reducePrimitives = [
    "reduce",
    "dot",
    "conv",
    "pool_transpose"
  ];
  const reductionEpilogueEqns = /* @__PURE__ */ new Set();
  const reductionEndpointEqns = /* @__PURE__ */ new Set();
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    if (reducePrimitives.includes(eqn.primitive)) {
      let head = i;
      while (true) {
        reductionEpilogueEqns.add(head);
        const outVar = jaxpr.eqns[head].outBinders[0];
        const usages = varToUsages.get(outVar) ?? [];
        if (jaxpr.outs.includes(outVar) || usages.length !== 1) break;
        if (reductionEpilogueEqns.has(usages[0])) break;
        const nextEqn = jaxpr.eqns[usages[0]];
        switch (nextEqn.primitive) {
          case "neg":
          case "reciprocal":
          case "floor":
          case "ceil":
          case "stop_gradient":
          case "cast":
          case "bitcast":
          case "sin":
          case "cos":
          case "asin":
          case "atan":
          case "exp":
          case "log":
          case "erf":
          case "erfc":
          case "sqrt":
            head = usages[0];
            continue;
          case "add":
          case "mul":
          case "idiv":
          case "mod":
          case "min":
          case "max":
          case "bit_combine":
          case "bit_shift": {
            const otherInput = nextEqn.inputs.find((v) => v !== outVar);
            if (otherInput instanceof Lit || deepEqual(generalBroadcast(otherInput.aval.shape, outVar.aval.shape), outVar.aval.shape)) {
              head = usages[0];
              continue;
            }
            break;
          }
        }
        break;
      }
      reductionEndpointEqns.add(head);
    }
  }
  const viewPrimitives = [
    "transpose",
    "broadcast",
    "reshape",
    "flip",
    "shrink",
    "pad"
  ];
  const materializedProducers = /* @__PURE__ */ new Set();
  const reductionInputHasReuse = (shape2, promoted) => {
    const offset = promoted.length - shape2.length;
    for (let axis = 0; axis < promoted.length - 1; axis++) if ((axis < offset ? 1 : shape2[axis - offset]) === 1 && promoted[axis] > 1) return true;
    return false;
  };
  const markNonViewProducer = (v) => {
    let current = v;
    while (true) {
      const defn = varToDefn.get(current);
      if (defn === void 0) return;
      const eqn = jaxpr.eqns[defn];
      if (!viewPrimitives.includes(eqn.primitive)) break;
      const input = eqn.inputs.find((input2) => input2 instanceof Var);
      if (!input) return;
      current = input;
    }
    materializedProducers.add(current);
  };
  for (const eqn of jaxpr.eqns) {
    let inputShapes = null;
    if (eqn.primitive === "dot") inputShapes = [eqn.inputs[0].aval.shape, eqn.inputs[1].aval.shape];
    else if (eqn.primitive === "conv") {
      const [lhs, rhs] = prepareConv(ShapeTracker.fromShape(eqn.inputs[0].aval.shape), ShapeTracker.fromShape(eqn.inputs[1].aval.shape), eqn.params);
      inputShapes = [lhs.shape, rhs.shape];
    }
    if (!inputShapes) continue;
    const promoted = generalBroadcast(inputShapes[0], inputShapes[1]);
    for (const [i, input] of eqn.inputs.entries()) if (input instanceof Var && reductionInputHasReuse(inputShapes[i], promoted)) markNonViewProducer(input);
  }
  const blackNodes = /* @__PURE__ */ new Set();
  const p1NextBlack = /* @__PURE__ */ new Map();
  for (const v of jaxpr.outs) if (v instanceof Var) {
    blackNodes.add(v);
    p1NextBlack.set(v, v);
  }
  const heterogeneousViewPrimitives = ["random_bits", "gather"];
  const needsCleanShapePrimitives = ["concatenate", "pad"];
  for (const eqn of jaxpr.eqns) if (needsCleanShapePrimitives.includes(eqn.primitive) || routinePrimitives.has(eqn.primitive)) {
    for (const input of eqn.inputs) if (input instanceof Var) markNonViewProducer(input);
  }
  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    if (reductionEndpointEqns.has(i) || heterogeneousViewPrimitives.includes(eqn.primitive) || routinePrimitives.has(eqn.primitive) || eqn.outBinders.some((v) => materializedProducers.has(v)) || eqn.outBinders.some((v) => blackNodes.has(v))) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
      continue;
    }
    const reach = /* @__PURE__ */ new Set();
    for (const v of eqn.outBinders) for (const j of varToUsages.get(v) ?? []) for (const o of jaxpr.eqns[j].outBinders) {
      const u = p1NextBlack.get(o);
      if (u) reach.add(u);
    }
    if (reach.size > 1) for (const v of eqn.outBinders) {
      blackNodes.add(v);
      p1NextBlack.set(v, v);
    }
    else if (reach.size === 1) {
      const b = reach.values().next().value;
      for (const v of eqn.outBinders) p1NextBlack.set(v, b);
    }
  }
  const p2Deps = /* @__PURE__ */ new Map();
  for (const v of jaxpr.inBinders) p2Deps.set(v, /* @__PURE__ */ new Set([v]));
  let p2idx = 0;
  while (p2idx < jaxpr.eqns.length) {
    const eqn = jaxpr.eqns[p2idx++];
    const deps = [];
    for (const input of eqn.inputs) if (input instanceof Var) if (blackNodes.has(input)) deps.push(/* @__PURE__ */ new Set([input]));
    else deps.push(p2Deps.get(input));
    else deps.push(/* @__PURE__ */ new Set());
    const depCounter = /* @__PURE__ */ new Map();
    for (const depSet of deps) for (const dep of depSet) depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
    if (depCounter.size > backend.maxArgs) {
      let maxUniqueDeps = 0;
      let assocInput = -1;
      for (let i = 0; i < eqn.inputs.length; i++) {
        const input = eqn.inputs[i];
        if (input instanceof Var && varToDefn.has(input)) {
          let uniqueDeps = 0;
          for (const dep of deps[i]) if (depCounter.get(dep) === 1) uniqueDeps++;
          if (uniqueDeps > maxUniqueDeps) {
            maxUniqueDeps = uniqueDeps;
            assocInput = i;
          }
        }
      }
      if (assocInput === -1) throw new Error(`internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`);
      const assocVar = eqn.inputs[assocInput];
      p2idx = varToDefn.get(assocVar);
      for (const out of jaxpr.eqns[p2idx++].outBinders) blackNodes.add(out);
    } else {
      const s = new Set(depCounter.keys());
      for (const out of eqn.outBinders) p2Deps.set(out, s);
    }
  }
  return blackNodes;
}
var JsArray$3 = globalThis.Array;
var inlineArrayLimit = 128;
var fudgeArray = pureArray;
var PendingExecute = class {
  backend;
  source;
  inputs;
  outputs;
  prepared = null;
  submitted = false;
  #promise = null;
  #rc = 1;
  constructor(backend, source, inputs, outputs) {
    this.backend = backend;
    this.source = source;
    this.inputs = inputs;
    this.outputs = outputs;
    for (const slot of inputs) this.backend.incRef(slot);
    for (const slot of outputs) this.backend.incRef(slot);
  }
  updateRc(delta) {
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    this.#rc += delta;
    if (this.#rc <= 0 && !this.submitted) {
      for (const slot of this.inputs) this.backend.decRef(slot);
      for (const slot of this.outputs) this.backend.decRef(slot);
    }
  }
  async prepare() {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      if (this.source instanceof Kernel) this.prepared = await this.backend.prepareKernel(this.source);
      else this.prepared = await this.backend.prepareRoutine(this.source);
    })();
    await this.#promise;
  }
  prepareSync() {
    if (this.prepared) return;
    if (this.source instanceof Kernel) this.prepared = this.backend.prepareKernelSync(this.source);
    else this.prepared = this.backend.prepareRoutineSync(this.source);
  }
  submit() {
    if (this.submitted) return;
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    if (!this.prepared) throw new Error("internal: Not prepared yet");
    this.submitted = true;
    this.backend.dispatch(this.prepared, this.inputs, this.outputs);
    for (const slot of this.inputs) this.backend.decRef(slot);
    for (const slot of this.outputs) this.backend.decRef(slot);
  }
};
var Array$1 = class Array$12 extends Tracer {
  #dtype;
  #weakType;
  #source;
  #st;
  #backend;
  #committed;
  #rc;
  #pendingSet;
  /**
  * @ignore
  * Constructs an array from source, shape and backend. Note that if the source
  * is a backend `Slot`, this constructor _takes ownership_ of the slot. It
  * will be freed when the array is disposed.
  */
  constructor(args) {
    super(baseArrayTrace);
    this.#dtype = args.dtype;
    this.#weakType = args.weakType;
    this.#source = args.source;
    this.#st = args.st;
    this.#backend = args.backend;
    this.#committed = args.committed;
    this.#rc = 1;
    this.#pendingSet = new Set(args.pending);
    if (this.#pendingSet.size === 0) this.#pendingSet = null;
    else if (this.#source instanceof AluExp) throw new Error("internal: AluExp source cannot have pending executes");
  }
  /** @ignore */
  get aval() {
    return new ShapedArray(this.#st.shape, this.#dtype, this.#weakType);
  }
  /** Return a simple string representation of the array's dimensions. */
  toString() {
    return `Array:${this.#dtype}[${this.shape.join(",")}]`;
  }
  get device() {
    return this.#backend.type;
  }
  #check() {
    if (this.#rc <= 0) throw new UseAfterFreeError(this);
  }
  /** Construct an array, copying fields from `this`. */
  #newArrayFrom(args) {
    return new Array$12({
      source: args.source ?? this.#source,
      st: args.st ?? this.#st,
      dtype: args.dtype ?? this.#dtype,
      weakType: this.#weakType,
      backend: args.backend ?? this.#backend,
      committed: args.committed ?? this.#committed,
      pending: args.pending ?? this.#pending ?? void 0
    });
  }
  get ref() {
    this.#check();
    this.#rc++;
    return this;
  }
  /** Get the current reference count (for debugging memory management). */
  get refCount() {
    return this.#rc;
  }
  dispose() {
    this.#check();
    if (--this.#rc === 0) {
      for (const exe of this.#pending) exe.updateRc(-1);
      if (typeof this.#source === "number") this.#backend.decRef(this.#source);
    }
  }
  /** Get the pending executes as a list, trimming if already submitted. */
  get #pending() {
    if (!this.#pendingSet) return [];
    for (const p of this.#pendingSet) if (p.submitted) this.#pendingSet.delete(p);
    if (this.#pendingSet.size === 0) {
      this.#pendingSet = null;
      return [];
    } else return [...this.#pendingSet];
  }
  /**
  * Convert this array into a primitive value.
  *
  * This only works for scalars (0-dimensional arrays). It lets you get values
  * "out" of the JAX system. For instance, if `x = np.array(5)`, then you can
  * evaluate `x + 1` and `x ** 2` to get `6` and `25`, respectively.
  *
  * This method is also called for `==` equality.
  */
  [Symbol.toPrimitive]() {
    if (this.ndim === 0) return this.dataSync()[0];
    else throw new Error(`Cannot convert non-scalar array to primitive: ${this.toString()}`);
  }
  #reshape(st) {
    this.#check();
    const pending = this.#pending;
    for (const exe of pending) exe.updateRc(1);
    if (typeof this.#source === "number") this.#backend.incRef(this.#source);
    const ar = this.#newArrayFrom({
      st,
      pending
    });
    this.dispose();
    return ar;
  }
  /**
  * Underlying implementation of the Gather primitive. This indexes an array
  * and extracts slices based on indices in other integer arrays.
  */
  #gather(indices2, axis, outDim) {
    this.#check();
    const axisSet = new Set(axis);
    if (axisSet.size !== axis.length) throw new TypeError("Gather axis must not have duplicates");
    if (indices2.some((a) => a.#committed && a.#backend !== this.#backend)) throw new TypeError(`Gather indices must have the same backend: ${this.#backend.type}`);
    indices2 = indices2.map((ar) => ar._putSync(this.#backend));
    indices2 = Array$12.#broadcastArrays(indices2);
    const indexShape = indices2[0].shape;
    const finalShape = this.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);
    const idxNonaxis = [...unravelAlu(finalShape, AluVar.gidx)];
    const idxAxis = idxNonaxis.splice(outDim, indexShape.length);
    const inputs = [];
    const src = [...idxNonaxis];
    for (let i = 0; i < this.shape.length; i++) if (axisSet.has(i)) src.splice(i, 0, null);
    for (const [i, ar] of indices2.entries()) if (ar.#source instanceof AluExp) src[axis[i]] = AluExp.cast("int32", accessorAluExp(ar.#source, ar.#st, idxAxis));
    else {
      let gid = inputs.indexOf(ar.#source);
      if (gid === -1) {
        gid = inputs.length;
        inputs.push(ar.#source);
      }
      src[axis[i]] = AluExp.cast("int32", AluExp.globalView(ar.#dtype, gid, ar.#st, idxAxis));
    }
    let exp3;
    if (this.#source instanceof AluExp) exp3 = accessorAluExp(this.#source, this.#st, src);
    else {
      let gid = inputs.indexOf(this.#source);
      if (gid === -1) {
        gid = inputs.length;
        inputs.push(this.#source);
      }
      exp3 = accessorGlobal(this.#dtype, gid, this.#st, src);
    }
    const kernel = new Kernel(inputs.length, prod(finalShape), exp3);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending, ...indices2.flatMap((ar) => ar.#pending)];
    for (const exe of pending) exe.updateRc(1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
    this.dispose();
    for (const ar of indices2) ar.dispose();
    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(finalShape),
      pending
    });
  }
  /** Move axes to the rightmost dimension of the shape. */
  #moveAxesDown(axis) {
    this.#check();
    if (axis.length === 0) return this.reshape(this.shape.concat(1));
    const newShape = [];
    const keptAxes = [];
    const shiftedAxes = [];
    for (let i = 0; i < this.#st.shape.length; i++) if (axis.includes(i)) shiftedAxes.push(i);
    else {
      keptAxes.push(i);
      newShape.push(this.#st.shape[i]);
    }
    const collapsedAxisSize = prod(shiftedAxes.map((axis2) => this.#st.shape[axis2]));
    newShape.push(collapsedAxisSize);
    return this.#transpose(keptAxes.concat(shiftedAxes)).reshape(newShape);
  }
  #transpose(perm) {
    this.#check();
    if (!isPermutation(perm, this.ndim)) throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
    return this.#reshape(this.#st.permute(perm));
  }
  #unary(op, dtypeOutput) {
    const weakType = !dtypeOutput && this.#weakType;
    dtypeOutput ??= this.#dtype;
    this.#check();
    if (this.#source instanceof AluExp) {
      const exp4 = new AluExp(op, dtypeOutput, [this.#source]);
      this.dispose();
      return this.#newArrayFrom({
        source: exp4.simplify(),
        dtype: dtypeOutput,
        weakType
      });
    }
    const indices2 = unravelAlu(this.#st.shape, AluVar.gidx);
    const exp3 = new AluExp(op, dtypeOutput, [AluExp.globalView(this.#dtype, 0, this.#st, indices2)]);
    const kernel = new Kernel(1, this.#st.size, exp3);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(1);
    pending.push(new PendingExecute(this.#backend, kernel, [this.#source], [output]));
    this.dispose();
    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(this.shape),
      dtype: dtypeOutput,
      weakType,
      pending
    });
  }
  #binary(op, other) {
    const custom = (src) => new AluExp(op, src[0].dtype, src);
    return Array$12.#naryCustom(op, custom, [this, other]);
  }
  static #naryCustom(name, custom, arrays, { dtypeOverride, strongTypeOutput, reduceAxis } = {}) {
    const n = arrays.length;
    if (n === 0) throw new TypeError(`No inputs for ${name}`);
    for (const ar of arrays) ar.#check();
    let castDtype;
    let castWeakType = true;
    for (let i = 0; i < n; i++) if (dtypeOverride?.[i]) {
      if (arrays[i].#dtype !== dtypeOverride[i]) throw new TypeError(`Wrong dtype in ${name}: expected ${dtypeOverride[i]}, got ${arrays[i].#dtype}`);
    } else if (castDtype === void 0) {
      castDtype = arrays[i].#dtype;
      castWeakType = arrays[i].#weakType;
    } else ({ dtype: castDtype, weakType: castWeakType } = promoteAvals(new ShapedArray([], castDtype, castWeakType), arrays[i].aval.scalar()));
    const weakType = castWeakType && !strongTypeOutput;
    const { backend, committed } = Array$12.#computeBackend(name, arrays);
    arrays = arrays.map((ar) => ar._putSync(backend));
    arrays = Array$12.#broadcastArrays(arrays);
    const newShape = [...arrays[0].shape];
    if (arrays.every((ar) => ar.#source instanceof AluExp) && !reduceAxis) {
      const sources = arrays.map((ar, i) => {
        if (!dtypeOverride?.[i]) return AluExp.cast(castDtype, ar.#source);
        else return ar.#source;
      });
      if (arrays.every((ar) => deepEqual(ar.#st, arrays[0].#st))) {
        const exp5 = custom(sources);
        arrays.forEach((ar) => ar.dispose());
        return new Array$12({
          source: exp5.simplify(),
          st: arrays[0].#st,
          dtype: exp5.dtype,
          weakType,
          backend,
          committed
        });
      }
      const exp4 = custom(arrays.map((ar, i) => {
        const src2 = sources[i];
        if (ar.#st.contiguous) return src2;
        return accessorAluExp(src2, ar.#st, unravelAlu(newShape, AluVar.idx));
      }));
      const st = ShapeTracker.fromShape(newShape);
      arrays.forEach((ar) => ar.dispose());
      return new Array$12({
        source: exp4.simplify(),
        st,
        dtype: exp4.dtype,
        weakType,
        backend,
        committed
      });
    }
    let indices2;
    if (!reduceAxis) indices2 = unravelAlu(newShape, AluVar.gidx);
    else indices2 = [...unravelAlu(newShape.slice(0, -1), AluVar.gidx), AluVar.ridx];
    const inputs = [];
    const src = [];
    for (const [i, ar] of arrays.entries()) {
      let nextSrc;
      if (ar.#source instanceof AluExp) nextSrc = accessorAluExp(ar.#source, ar.#st, indices2);
      else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        nextSrc = AluExp.globalView(ar.#dtype, gid, ar.#st, indices2);
      }
      if (!dtypeOverride?.[i]) nextSrc = AluExp.cast(castDtype, nextSrc);
      src.push(nextSrc);
    }
    const exp3 = custom(src);
    let re = void 0;
    if (reduceAxis) {
      const [axisSize] = newShape.splice(-1, 1);
      re = new Reduction(exp3.dtype, "Add", axisSize);
    }
    const kernel = new Kernel(inputs.length, prod(newShape), exp3, re);
    const output = backend.malloc(kernel.bytes);
    const pending = /* @__PURE__ */ new Set([...arrays.flatMap((ar) => ar.#pending)]);
    for (const exe of pending) exe.updateRc(1);
    pending.add(new PendingExecute(backend, kernel, inputs, [output]));
    arrays.forEach((ar) => ar.dispose());
    return new Array$12({
      source: output,
      st: ShapeTracker.fromShape(newShape),
      dtype: kernel.dtype,
      weakType,
      backend,
      committed,
      pending
    });
  }
  /** Reduce the last dimension of the array by an operation. */
  #reduce(op) {
    const shape2 = this.shape;
    const reduction = new Reduction(this.#dtype, op, shape2[shape2.length - 1]);
    const newShape = shape2.slice(0, -1);
    const newSize = prod(newShape);
    const indices2 = [...unravelAlu(newShape, AluVar.gidx), AluVar.ridx];
    let exp3;
    const inputs = [];
    if (this.#source instanceof AluExp) exp3 = accessorAluExp(this.#source, this.#st, indices2);
    else {
      inputs.push(this.#source);
      exp3 = accessorGlobal(this.#dtype, 0, this.#st, indices2);
    }
    const kernel = new Kernel(inputs.length, newSize, exp3, reduction);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
    this.dispose();
    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(newShape),
      pending
    });
  }
  /** Apply an operation with custom lowering to this array. */
  static #routine(prim) {
    return (arrays, params) => {
      const { backend, committed } = Array$12.#computeBackend(prim, arrays);
      for (const ar of arrays) ar.#realize();
      const avals = arrays.map((ar) => ar.aval);
      const avalsOut = abstractEvalRules[prim](avals, params);
      const routine = new Routine(routinePrimitives.get(prim), {
        inputShapes: avals.map((a) => a.shape),
        inputDtypes: avals.map((a) => a.dtype),
        outputShapes: avalsOut.map((a) => a.shape),
        outputDtypes: avalsOut.map((a) => a.dtype)
      }, params);
      const inputs = arrays.map((ar) => ar.#source);
      const outputs = avalsOut.map((x) => backend.malloc(byteWidth(x.dtype) * x.size));
      const pending = arrays.flatMap((ar) => ar.#pending);
      for (const exe of pending) exe.updateRc(+outputs.length);
      pending.push(new PendingExecute(backend, routine, inputs, outputs));
      pending[pending.length - 1].updateRc(+outputs.length - 1);
      arrays.forEach((ar) => ar.dispose());
      return outputs.map((output, i) => new Array$12({
        source: output,
        st: ShapeTracker.fromShape(avalsOut[i].shape),
        dtype: avalsOut[i].dtype,
        weakType: avalsOut[i].weakType,
        backend,
        committed,
        pending
      }));
    };
  }
  /**
  * Normalizes this array into one backed by a `Slot`.
  *
  * This mutates the array in-place, turning it into an equivalent array whose
  * source is actual, contiguous data on device.
  *
  * Calling this twice is a no-op.
  */
  #realize() {
    this.#check();
    const indices2 = unravelAlu(this.#st.shape, AluVar.gidx);
    if (this.#source instanceof AluExp) {
      let resolvedSource;
      if (this.#st.contiguous && this.#st.size < inlineArrayLimit && (resolvedSource = this.#source.resolve()) !== void 0) {
        const byteLength = this.#st.size * byteWidth(this.#dtype);
        const initialData = new Uint8Array(byteLength);
        dtypedArray(this.#dtype, initialData).fill(resolvedSource);
        this.#source = this.#backend.malloc(byteLength, initialData);
        this.#st = ShapeTracker.fromShape(this.shape);
        return;
      }
      const exp3 = accessorAluExp(this.#source, this.#st, indices2);
      const kernel = new Kernel(0, this.#st.size, exp3);
      const output = this.#backend.malloc(kernel.bytes);
      const pendingItem = new PendingExecute(this.#backend, kernel, [], [output]);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet = /* @__PURE__ */ new Set([pendingItem]);
    } else {
      if (this.#st.contiguous) return;
      const exp3 = accessorGlobal(this.#dtype, 0, this.#st, indices2);
      const kernel = new Kernel(1, this.#st.size, exp3);
      const output = this.#backend.malloc(kernel.bytes);
      const pendingItem = new PendingExecute(this.#backend, kernel, [this.#source], [output]);
      this.#backend.decRef(this.#source);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet ??= /* @__PURE__ */ new Set();
      this.#pendingSet.add(pendingItem);
    }
  }
  #dataInline() {
    this.#check();
    if (!(this.#source instanceof AluExp)) throw new Error("internal: #dataInline called on non-AluExp source");
    const ar = this.#newArrayFrom({ backend: getBackend("cpu") });
    this.dispose();
    return ar.dataSync();
  }
  static #broadcastArrays(arrays) {
    if (arrays.length === 0) throw new Error("Need at least one array to broadcast");
    if (arrays.length === 1) return arrays;
    const newShape = arrays.map((a) => a.shape).reduce(generalBroadcast);
    return arrays.map((ar) => {
      if (deepEqual(ar.shape, newShape)) return ar;
      return ar.#reshape(ar.#st.broadcast(newShape, range(newShape.length - ar.ndim)));
    });
  }
  static #computeBackend(name, arrays) {
    const committed = arrays.filter((ar) => ar.#committed);
    if (committed.length > 0) {
      const backend = committed[0].#backend;
      for (const ar of committed) if (ar.#backend !== backend) throw new Error(`Device mismatch in ${name} between committed arrays on (${backend.type}, ${ar.#backend.type}), please move to the same device with devicePut()`);
      return {
        backend,
        committed: true
      };
    } else return {
      backend: arrays.length > 0 ? arrays[0].#backend : getBackend(),
      committed: false
    };
  }
  /** Realize the array and return it as data. */
  async data() {
    if (this.#source instanceof AluExp && this.size < inlineArrayLimit && this.device !== "cpu") return this.#dataInline();
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending) p.submit();
    }
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = await this.#backend.read(this.#source, 0, byteCount);
    this.dispose();
    return dtypedArray(this.dtype, buf);
  }
  /**
  * Wait for this array to finish evaluation.
  *
  * Operations and data loading in jax-js are lazy, so this function ensures
  * that pending operations are dispatched and fully executed before it
  * returns.
  *
  * If you are mapping from `data()` or `dataSync()`, it will also trigger
  * dispatch of operations as well.
  *
  * **Note:** `jax.blockUntilReady()` is a higher-level API, it calls this
  * asynchronously for multiple arrays.
  */
  async blockUntilReady() {
    this.#check();
    if (this.#source instanceof AluExp) return this;
    const pending = this.#pending;
    if (pending) {
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending) p.submit();
    }
    await this.#backend.read(this.#source, 0, 0);
    return this;
  }
  /**
  * Realize the array and return it as data. This is a sync variant and not
  * recommended for performance reasons, as it will block rendering.
  */
  dataSync() {
    if (this.#source instanceof AluExp && this.size < inlineArrayLimit && this.device !== "cpu") return this.#dataInline();
    this.#realize();
    for (const p of this.#pending) {
      p.prepareSync();
      p.submit();
    }
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = this.#backend.readSync(this.#source, 0, byteCount);
    this.dispose();
    return dtypedArray(this.dtype, buf);
  }
  /**
  * Return this array as a WebGPU buffer (with `STORAGE | COPY_SRC`).
  *
  * Only available on the WebGPU backend. The array's memory is still managed
  * by jax-js, and it will be freed when the buffer is no longer in use. You
  * _should not_ mutate the buffer's contents.
  *
  * Note that the GPU buffer may be slightly larger than the array's size; it
  * will always be aligned to 4 bytes.
  */
  async gpuBuffer() {
    if (this.device !== "webgpu") throw new Error(`gpuBuffer() is only available on WebGPU backend`);
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending) p.submit();
    }
    const { buffer } = this.#backend.buffers.get(this.#source);
    this.dispose();
    return buffer;
  }
  /** Synchronous version of `Array.gpuBuffer()`. */
  gpuBufferSync() {
    if (this.device !== "webgpu") throw new Error(`gpuBufferSync() is only available on WebGPU backend`);
    this.#realize();
    for (const p of this.#pending) {
      p.prepareSync();
      p.submit();
    }
    const { buffer } = this.#backend.buffers.get(this.#source);
    this.dispose();
    return buffer;
  }
  /**
  * Convert this array into a JavaScript object.
  *
  * This is a blocking operation that will compile all of the shaders and wait
  * for execution to complete, synchronously. No other JavaScript code on the
  * site will be run during shader execution.
  *
  * To avoid blocking, prefer `jsAsync()` when possible.
  */
  js() {
    return dataToJs(this.dtype, this.dataSync(), this.shape);
  }
  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return dataToJs(this.dtype, await this.data(), this.shape);
  }
  /**
  * Copy an element of an array to a numeric scalar and return it.
  *
  * Throws an error if the array does not have a single element. The array must
  * either be rank-0, or all dimensions of the shape are 1.
  */
  item() {
    if (this.size !== 1) throw new Error(`item() can only be called on arrays of size 1`);
    return this.dataSync()[0];
  }
  /** @private Internal plumbing method for Array / Tracer ops. */
  static _implRules() {
    return {
      ["add"]([x, y]) {
        return [x.#binary("Add", y)];
      },
      ["mul"]([x, y]) {
        return [x.#binary("Mul", y)];
      },
      ["idiv"]([x, y]) {
        return [x.#binary("Idiv", y)];
      },
      ["mod"]([x, y]) {
        return [x.#binary("Mod", y)];
      },
      ["min"]([x, y]) {
        return [x.#binary("Min", y)];
      },
      ["max"]([x, y]) {
        return [x.#binary("Max", y)];
      },
      ["bit_combine"]([x, y], { op }) {
        const custom = (src) => AluExp.bitCombine(src[0], src[1], op);
        return [Array$12.#naryCustom("bit_combine", custom, [x, y])];
      },
      ["bit_shift"]([x, y], { op }) {
        const custom = (src) => AluExp.bitShift(src[0], src[1], op);
        return [Array$12.#naryCustom("bit_shift", custom, [x, y], { dtypeOverride: [void 0, y.dtype] })];
      },
      ["neg"]([x]) {
        return [zerosLike$1(x.ref).#binary("Sub", x)];
      },
      ["reciprocal"]([x]) {
        return [x.#unary("Reciprocal")];
      },
      ["floor"]([x]) {
        return [x.#unary("Floor")];
      },
      ["ceil"]([x]) {
        return [x.#unary("Ceil")];
      },
      ["stop_gradient"]([x]) {
        return [x];
      },
      ["cast"]([x], { dtype }) {
        return [x.#unary("Cast", dtype)];
      },
      ["bitcast"]([x], { dtype }) {
        if (x.dtype === dtype) return [x];
        if (x.dtype === "bool" || dtype === "bool") throw new TypeError("Bitcast to/from bool is not allowed");
        if (byteWidth(x.dtype) !== byteWidth(dtype)) throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
        if (x.#source instanceof AluExp) return [x.#unary("Bitcast", dtype)];
        else {
          x.#backend.incRef(x.#source);
          const pending = x.#pending;
          for (const exe of pending) exe.updateRc(1);
          const y = x.#newArrayFrom({
            dtype,
            weakType: false,
            pending
          });
          x.dispose();
          return [y];
        }
      },
      ["sin"]([x]) {
        return [x.#unary("Sin")];
      },
      ["cos"]([x]) {
        return [x.#unary("Cos")];
      },
      ["asin"]([x]) {
        return [x.#unary("Asin")];
      },
      ["atan"]([x]) {
        return [x.#unary("Atan")];
      },
      ["exp"]([x]) {
        return [x.#unary("Exp")];
      },
      ["log"]([x]) {
        return [x.#unary("Log")];
      },
      ["erf"]([x]) {
        return [x.#unary("Erf")];
      },
      ["erfc"]([x]) {
        return [x.#unary("Erfc")];
      },
      ["sqrt"]([x]) {
        return [x.#unary("Sqrt")];
      },
      ["reduce"]([x], { op, axis }) {
        if (axis.length === 0) return [x];
        return [x.#moveAxesDown(axis).#reduce(op)];
      },
      ["pool"]([x], { window, strides }) {
        const st = pool(x.#st, window, strides);
        return [x.#reshape(st)];
      },
      ["pool_transpose"]([x], { inShape, window, strides }) {
        const n = inShape.length;
        let st = poolTranspose(x.#st, inShape, window, strides);
        st = st.reshape([...st.shape.slice(0, n), prod(st.shape.slice(n))]);
        return [x.#reshape(st).#reduce("Add")];
      },
      ["dot"]([x, y]) {
        return [Array$12.#naryCustom("dot", ([x2, y2]) => AluExp.mul(x2, y2), [x, y], { reduceAxis: true })];
      },
      ["conv"]([x, y], params) {
        checkConvShape(x.shape, y.shape, params);
        if (x.#backend.type === "wasm" && params.lhsDilation.every((d) => d === 1) && params.padding.some(([left, right]) => left > 0 || right > 0)) {
          x = x.#reshape(x.#st.padOrShrink([...rep(params.vmapDims + 2, [0, 0]), ...params.padding]));
          x.#realize();
          params = {
            ...params,
            padding: rep(params.padding.length, [0, 0])
          };
        }
        const [stX, stY] = prepareConv(x.#st, y.#st, params);
        return [Array$12.#naryCustom("conv", ([x2, y2]) => AluExp.mul(x2, y2), [x.#reshape(stX), y.#reshape(stY)], { reduceAxis: true })];
      },
      ["compare"]([x, y], { op }) {
        const custom = ([x2, y2]) => aluCompare(x2, y2, op);
        return [Array$12.#naryCustom("compare", custom, [x, y], { strongTypeOutput: true })];
      },
      ["where"]([cond, x, y]) {
        const custom = ([cond2, x2, y2]) => AluExp.where(cond2, x2, y2);
        return [Array$12.#naryCustom("where", custom, [
          cond,
          x,
          y
        ], { dtypeOverride: ["bool"] })];
      },
      ["concatenate"](xs, { axis }) {
        const ndim2 = xs[0].ndim;
        const sizes = xs.map((x) => x.shape[axis]);
        const finalSize = sizes.reduce((a, b) => a + b, 0);
        const makePadAxis = (start, end) => range(ndim2).map((i) => i === axis ? [start, end] : [0, 0]);
        let cum = 0;
        const xsPadded = [];
        for (let i = 0; i < xs.length; i++) {
          const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
          xsPadded.push(xs[i].#reshape(xs[i].#st.pad(padding)));
          cum += sizes[i];
        }
        const custom = (exps) => exps.reduce(AluExp.add);
        return [Array$12.#naryCustom("concatenate", custom, xsPadded)];
      },
      ["split"]([x], { axis, sizes }) {
        const outputs = [];
        for (let i = 0, start = 0; i < sizes.length; i++) {
          const slice = range(x.ndim).map((d) => d === axis ? [start, start + sizes[i]] : [0, x.shape[d]]);
          outputs.push(x.ref.#reshape(x.#st.shrink(slice)));
          start += sizes[i];
        }
        x.dispose();
        return outputs;
      },
      ["random_bits"]([k0, k1], { shape: shape2, mode }) {
        const keyShape = k0.shape;
        const genShape = shape2.slice(keyShape.length);
        const c0 = zeros(genShape, {
          dtype: "uint32",
          device: k0.device
        });
        const c1 = arange(0, prod(genShape), 1, {
          dtype: "uint32",
          device: k0.device
        }).reshape(genShape);
        k0 = k0.#reshape(k0.#st.reshape(keyShape.concat(rep(genShape.length, 1))));
        k1 = k1.#reshape(k1.#st.reshape(keyShape.concat(rep(genShape.length, 1))));
        const custom = ([k02, k12, c02, c12]) => AluExp.threefry2x32(k02, k12, c02, c12, mode);
        return [Array$12.#naryCustom("random_bits", custom, [
          k0,
          k1,
          c0,
          c1
        ])];
      },
      ["gather"]([x, ...indices2], { axis, outDim }) {
        return [x.#gather(indices2, axis, outDim)];
      },
      ["transpose"]([x], { perm }) {
        return [x.#transpose(perm)];
      },
      ["broadcast"]([x], { shape: shape2, axis }) {
        return [x.#reshape(x.#st.broadcast(shape2, axis))];
      },
      ["reshape"]([x], { shape: shape2 }) {
        return [x.#reshape(x.#st.reshape(shape2))];
      },
      ["flip"]([x], { axis }) {
        const arg = rep(x.ndim, false);
        for (const ax of axis) arg[ax] = true;
        return [x.#reshape(x.#st.flip(arg))];
      },
      ["shrink"]([x], { slice }) {
        return [x.#reshape(x.#st.shrink(slice))];
      },
      ["pad"]([x], { width }) {
        return [x.#reshape(x.#st.pad(width))];
      },
      ["sort"]: Array$12.#routine("sort"),
      ["argsort"]: Array$12.#routine("argsort"),
      ["scatter"]: Array$12.#routine("scatter"),
      ["triangular_solve"]: Array$12.#routine("triangular_solve"),
      ["cholesky"]: Array$12.#routine("cholesky"),
      ["lu"]: Array$12.#routine("lu"),
      ["jacobi_eigh"]: Array$12.#routine("jacobi_eigh"),
      ["fft"]: Array$12.#routine("fft"),
      ["jit"](args, { jaxpr }) {
        if (jaxpr.inBinders.length !== args.length) throw new Error(`jit expects ${jaxpr.inBinders.length} args, got ${args.length}`);
        const { backend, committed } = Array$12.#computeBackend("jit", args);
        args = args.map((ar) => ar._putSync(backend));
        const { outputs, pending } = jitCompile(backend, jaxpr).execute(args.map((x) => x._realizeSource()));
        for (const exe of pending) exe.updateRc(+outputs.length - 1);
        const prevPending = [...new Set(args.flatMap((x) => x.#pending))];
        for (const exe of prevPending) exe.updateRc(+outputs.length);
        pending.splice(0, 0, ...prevPending);
        args.forEach((x) => x.dispose());
        return outputs.map((source, i) => {
          return new Array$12({
            source,
            st: ShapeTracker.fromShape(jaxpr.outs[i].aval.shape),
            dtype: jaxpr.outs[i].aval.dtype,
            weakType: jaxpr.outs[i].aval.weakType,
            backend,
            committed,
            pending
          });
        });
      }
    };
  }
  /** @private */
  _realizeSource() {
    this.#realize();
    return this.#source;
  }
  /** @private Put this array on a new backend, asynchronously. */
  async _put(backend) {
    if (this.#backend === backend) return this;
    if (this.#source instanceof AluExp) {
      const ar = this.#newArrayFrom({
        backend,
        committed: true
      });
      this.dispose();
      return ar;
    } else return arrayFromData(await this.data(), this.shape, {
      dtype: this.#dtype,
      device: backend.type
    }, this.#weakType);
  }
  /** @private Put this array on a new backend, synchronously. */
  _putSync(backend) {
    if (this.#backend === backend) return this;
    if (this.#source instanceof AluExp) {
      const ar = this.#newArrayFrom({
        backend,
        committed: true
      });
      this.dispose();
      return ar;
    } else return arrayFromData(this.dataSync(), this.shape, {
      dtype: this.#dtype,
      device: backend.type
    }, this.#weakType);
  }
};
function array(values, { shape: shape2, dtype, device } = {}) {
  if (values instanceof Tracer) {
    if (shape2 && !deepEqual(values.shape, shape2)) values = values.reshape(shape2);
    if (dtype && values.dtype !== dtype) values = values.astype(dtype);
    return values;
  } else if (ArrayBuffer.isView(values)) return arrayFromData(values, shape2 ?? [values.length], {
    dtype,
    device
  });
  else {
    if (!shape2) {
      shape2 = [];
      let cur = values;
      while (JsArray$3.isArray(cur)) {
        shape2.push(cur.length);
        cur = cur[0];
      }
    }
    const size2 = prod(shape2);
    const flat = recursiveFlatten(values);
    if (flat.length !== size2) throw new Error(`Jagged shape: ${JSON.stringify(shape2)} vs ${flat.length}`);
    if (size2 === 0) return zeros(shape2, {
      dtype,
      device
    });
    if (size2 === 1) return full(shape2, flat[0], {
      dtype,
      device
    });
    if (typeof flat[0] === "boolean") {
      dtype = dtype ?? "bool";
      return arrayFromData(new Int32Array(flat.map((x) => x ? 1 : 0)), shape2, {
        dtype,
        device
      });
    } else {
      const weakType = dtype == void 0 && shape2.length === 0;
      dtype = dtype ?? "float32";
      return arrayFromData(dtypedJsArray(dtype, flat), shape2, {
        dtype,
        device
      }, weakType);
    }
  }
}
function arrayFromData(data, shape2, { dtype, device }, weakType = false) {
  if (data instanceof Float32Array) {
    if (dtype && dtype !== "float32") throw new Error("Float32Array must have float32 type");
    dtype ??= "float32";
  } else if (data instanceof Int32Array) {
    if (dtype && dtype !== "int32" && dtype !== "bool") throw new Error("Int32Array must have int32 or bool type");
    dtype ??= "int32";
  } else if (data instanceof Uint32Array) {
    if (dtype && dtype !== "uint32") throw new Error("Uint32Array must have uint32 type");
    dtype ??= "uint32";
  } else if (data instanceof Float16Array) {
    if (dtype && dtype !== "float16") throw new Error("Float16Array must have float16 type");
    dtype ??= "float16";
  } else if (data instanceof Float64Array) {
    if (dtype && dtype !== "float64") throw new Error("Float64Array must have float64 type");
    dtype ??= "float64";
  } else throw new Error("Unsupported data array type: " + data.constructor.name);
  if (data.length < inlineArrayLimit) {
    let allEqual = true;
    for (let i = 1; i < data.length; i++) if (data[i] !== data[0] || !Object.is(data[i], data[0])) {
      allEqual = false;
      break;
    }
    if (allEqual) return fullInternal(new ShapedArray(shape2, dtype, weakType), data[0], device);
  }
  const backend = getBackend(device);
  const buf = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Array$1({
    source: backend.malloc(data.byteLength, buf),
    st: ShapeTracker.fromShape(shape2),
    dtype,
    weakType,
    backend,
    committed: device != void 0
  });
}
function dataToJs(dtype, data, shape2) {
  if (shape2.length === 0) return dtype === "bool" ? Boolean(data[0]) : data[0];
  const [first, ...rest] = shape2;
  const restSize = prod(rest);
  const ret = [];
  for (let i = 0; i < first; i++) {
    const subarray = data.slice(i * restSize, (i + 1) * restSize);
    ret.push(dataToJs(dtype, subarray, rest));
  }
  return ret;
}
function pureArray(x) {
  if (x instanceof Tracer) return x;
  else return array(x);
}
var EvalTrace = class extends Trace {
  pure = (x) => pureArray(x);
  lift = (x) => x;
  processPrimitive(primitive, tracers, params) {
    return implRules[primitive](tracers, params);
  }
};
var baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));
var implRules = Array$1._implRules();
function fullInternal(aval, fillValue, device) {
  return new Array$1({
    source: AluExp.const(aval.dtype, fillValue),
    st: ShapeTracker.fromShape(aval.shape),
    dtype: aval.dtype,
    weakType: aval.weakType,
    backend: getBackend(device),
    committed: device != void 0
  });
}
function zerosLike$1(val, opts) {
  return fullLike$1(val, 0, opts);
}
function onesLike$1(val, opts) {
  return fullLike$1(val, 1, opts);
}
function fullLike$1(val, fillValue, { dtype, shape: shape2, device } = {}) {
  const aval = getAval(val);
  if (val instanceof Tracer) val.dispose();
  if (fillValue instanceof Tracer) throw new Error("numpy.fullLike() with array argument not implemented yet");
  return fullInternal(new ShapedArray(shape2 ?? aval.shape, dtype ?? aval.dtype, aval.weakType && dtype === void 0), fillValue, device);
}
function zeros(shape2, opts) {
  return full(shape2, 0, opts);
}
function ones(shape2, opts) {
  return full(shape2, 1, opts);
}
function full(shape2, fillValue, { dtype, device } = {}) {
  let weakType = dtype == void 0 && shape2.length === 0;
  if (typeof fillValue === "number") dtype = dtype ?? "float32";
  else if (typeof fillValue === "boolean") {
    dtype = dtype ?? "bool";
    weakType = false;
  } else if (fillValue instanceof Tracer) throw new Error("numpy.full() with array argument not implemented yet");
  else throw new TypeError(`Invalid type for full: ${fillValue}`);
  return fullInternal(new ShapedArray(shape2, dtype, weakType), fillValue, device);
}
function eye(numRows, numCols, { dtype, device } = {}) {
  numCols = numCols ?? numRows;
  const weakType = dtype == void 0;
  dtype = dtype ?? "float32";
  if (numCols < numRows) return eye(numCols, numRows, {
    dtype,
    device
  }).transpose();
  if (numRows === 0) return zeros([0, numCols], {
    dtype,
    device
  });
  const exp3 = AluExp.cmplt(AluExp.mod(AluVar.idx, AluExp.i32(numCols + 1)), AluExp.i32(1));
  return new Array$1({
    source: AluExp.cast(dtype, exp3),
    st: ShapeTracker.fromShape([numRows, numCols]),
    dtype,
    weakType,
    backend: getBackend(device),
    committed: device != void 0
  });
}
function identity$1(n, { dtype, device } = {}) {
  return eye(n, n, {
    dtype,
    device
  });
}
function arange(start, stop, step = 1, { dtype, device } = {}) {
  dtype = dtype ?? "int32";
  if (stop === void 0) {
    stop = start;
    start = 0;
  }
  if (step === 0) throw new RangeError(`Invalid step for arange: ${step}. Step must be non-zero.`);
  const size2 = Math.max(0, Math.ceil((stop - start) / step));
  if (size2 === 0) return zeros([0], {
    dtype,
    device
  });
  return new Array$1({
    source: AluExp.add(AluExp.const(dtype, start), AluExp.mul(AluExp.cast(dtype, AluVar.idx), AluExp.const(dtype, step))),
    st: ShapeTracker.fromShape([size2]),
    dtype,
    weakType: false,
    backend: getBackend(device),
    committed: device != void 0
  });
}
function tri(n, m, k = 0, { dtype, device } = {}) {
  m ??= n;
  dtype ??= "float32";
  if (!Number.isInteger(n) || n < 0) throw new Error(`tri: n must be a non-negative integer, got ${n}`);
  if (!Number.isInteger(m) || m < 0) throw new Error(`tri: m must be a non-negative integer, got ${m}`);
  if (!Number.isInteger(k)) throw new Error(`tri: k must be an integer, got ${k}`);
  const rows = arange(k, n + k, 1, {
    dtype: "int32",
    device
  });
  const cols = arange(0, m, 1, {
    dtype: "int32",
    device
  });
  return rows.reshape([n, 1]).greaterEqual(cols).astype(dtype);
}
function tril(a, k = 0) {
  if (ndim$1(a) < 2) throw new Error(`tril: input array must be at least 2D, got ${ndim$1(a)}D`);
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  return where$1(tri(n, m, k, { dtype: "bool" }), a.ref, zerosLike$1(a));
}
function triu(a, k = 0) {
  if (ndim$1(a) < 2) throw new Error(`tril: input array must be at least 2D, got ${ndim$1(a)}D`);
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  return where$1(tri(n, m, k - 1, { dtype: "bool" }), zerosLike$1(a.ref), a);
}
function linspace(start, stop, num = 50, endpoint = true, { dtype, device } = {}) {
  dtype = dtype ?? "float32";
  if (num < 0 || !Number.isInteger(num)) throw new RangeError(`Invalid num for linspace: ${num}. Must be non-negative integer.`);
  else if (num === 0) return zeros([0], {
    dtype,
    device
  });
  else if (num === 1) return full([1], start, {
    dtype,
    device
  });
  else if (start === stop) return full([num], start, {
    dtype,
    device
  });
  const delta = stop - start;
  const denom = endpoint ? num - 1 : num;
  return new Array$1({
    source: AluExp.cast(dtype, AluExp.add(AluExp.f32(start), AluExp.mul(AluExp.f32(delta / denom), AluExp.cast("float32", AluVar.idx)))),
    st: ShapeTracker.fromShape([num]),
    dtype,
    weakType: false,
    backend: getBackend(device),
    committed: device != void 0
  });
}
function logspace(start, stop, num = 50, endpoint = true, base = 10, { dtype, device } = {}) {
  return exp$1(mul(linspace(start, stop, num, endpoint, {
    dtype,
    device
  }), Math.log(base)));
}
function geomspace(start, stop, num = 50, endpoint = true, { dtype, device } = {}) {
  if (start === 0 || stop === 0 || Math.sign(start) !== Math.sign(stop)) throw new RangeError(`geomspace: start (${start}) and stop (${stop}) must be nonzero and have the same sign`);
  dtype ??= "float32";
  const computationDtype = isFloatDtype(dtype) ? dtype : "float32";
  const sign2 = Math.sign(start);
  const y = logspace(Math.log10(Math.abs(start)), Math.log10(Math.abs(stop)), num, endpoint, 10, {
    dtype: computationDtype,
    device
  });
  const result = sign2 < 0 ? y.mul(-1) : y;
  return result.dtype === dtype ? result : result.astype(dtype);
}
function aluCompare(a, b, op) {
  switch (op) {
    case "less":
      return AluExp.cmplt(a, b);
    case "equal":
      return AluExp.cmpne(a, b).not();
    case "not_equal":
      return AluExp.cmpne(a, b);
    case "less_equal":
      return AluExp.add(AluExp.cmplt(a, b), AluExp.cmpne(a, b).not());
  }
}
function mappedAval(batchDim, aval) {
  const shape2 = [...aval.shape];
  shape2.splice(batchDim, 1);
  return new ShapedArray(shape2, aval.dtype, aval.weakType);
}
function moveaxis$1(x, src, dst) {
  const t = pureArray(x);
  src = checkAxis(src, t.ndim);
  dst = checkAxis(dst, t.ndim);
  if (src === dst) return t;
  const perm = range(t.ndim);
  perm.splice(src, 1);
  perm.splice(dst, 0, src);
  return transpose$1(t, perm);
}
function moveBatchAxis(axisSize, src, dst, x) {
  if (src === null) {
    const targetShape = [...x.shape];
    targetShape.splice(dst, 0, axisSize);
    return broadcast(x, targetShape, [dst]);
  } else if (src === dst) return x;
  else return moveaxis$1(x, src, dst);
}
var BatchTracer = class extends Tracer {
  val;
  batchDim;
  constructor(trace2, val, batchDim) {
    super(trace2);
    this.val = val;
    this.batchDim = batchDim;
  }
  get aval() {
    if (this.batchDim === null) return this.val.aval;
    else return mappedAval(this.batchDim, this.val.aval);
  }
  toString() {
    return `BatchTracer(${this.val.toString()}, ${this.batchDim})`;
  }
  get ref() {
    this.val.ref;
    return this;
  }
  dispose() {
    this.val.dispose();
  }
  fullLower() {
    if (this.batchDim === null) return this.val.fullLower();
    else return this;
  }
};
var BatchTrace = class extends Trace {
  pure(val) {
    return this.lift(pureArray(val));
  }
  lift(val) {
    return new BatchTracer(this, val, null);
  }
  processPrimitive(primitive, tracers, params) {
    const [valsIn, bdimsIn] = unzip2(tracers.map((t) => [t.val, t.batchDim]));
    const vmapRule = vmapRules[primitive];
    if (vmapRule === void 0) throw new Error(`No vmap rule for: ${primitive}`);
    if (bdimsIn.every((d) => d === null)) return bind(primitive, valsIn, params).map((x) => new BatchTracer(this, x, null));
    const [valOuts, bdimOuts] = vmapRule(this.axisSize, valsIn, bdimsIn, params);
    if (valOuts.length !== bdimOuts.length) throw new Error(`vmap rule for ${primitive} returned mismatched lengths: ${valOuts.length} vs ${bdimOuts.length}`);
    return zip(valOuts, bdimOuts).map(([x, bd]) => new BatchTracer(this, x, bd));
  }
  get axisSize() {
    return this.main.globalData;
  }
};
function broadcastBatcher(prim) {
  return (axisSize, args, dims, params) => {
    if (args.length === 0) throw new Error("Empty list in broadcastBatcher");
    const nd = Math.max(...args.map((x, i) => ndim$1(x) + (dims[i] === null ? 1 : 0)));
    const firstIdx = dims.findIndex((d) => d !== null);
    const firstBdim = dims[firstIdx] - args[firstIdx].ndim;
    if (zip(args, dims).every(([x, d]) => d === null && ndim$1(x) < -firstBdim || d !== null && d - x.ndim === firstBdim)) return [[bind1(prim, args, params)], [nd + firstBdim]];
    args = args.map((x, i) => {
      if (dims[i] === null) return x;
      x = moveBatchAxis(axisSize, dims[i], 0, x);
      if (x.ndim < nd) x = x.reshape([
        x.shape[0],
        ...rep(nd - x.ndim, 1),
        ...x.shape.slice(1)
      ]);
      return x;
    });
    return [[bind1(prim, args, params)], [0]];
  };
}
function unopBatcher(prim) {
  return (axisSize, [x], [xBdim], params) => {
    return [[bind1(prim, [x], params)], [xBdim]];
  };
}
function lastDimsBatcher(prim, inputDims, numOutputs = 1) {
  return (axisSize, [x], [xBdim], params) => {
    if (xBdim < x.ndim - inputDims) return [bind(prim, [x], params), rep(numOutputs, xBdim)];
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    return [bind(prim, [x], params), rep(numOutputs, 0)];
  };
}
function batchIndexArrays(axisSize, indices2, batchDims) {
  const rank = Math.max(...indices2.map((index, i) => index.ndim + (batchDims[i] === null ? 1 : 0)));
  return [indices2.map((index, i) => {
    const batchDim = batchDims[i];
    if (batchDim === null) return index;
    index = moveBatchAxis(axisSize, batchDim, 0, index);
    if (index.ndim < rank) index = index.reshape([
      index.shape[0],
      ...rep(rank - index.ndim, 1),
      ...index.shape.slice(1)
    ]);
    return index;
  }), rank];
}
var vmapRules = {
  ["add"]: broadcastBatcher("add"),
  ["mul"]: broadcastBatcher("mul"),
  ["idiv"]: broadcastBatcher("idiv"),
  ["mod"]: broadcastBatcher("mod"),
  ["min"]: broadcastBatcher("min"),
  ["max"]: broadcastBatcher("max"),
  ["bit_combine"]: broadcastBatcher("bit_combine"),
  ["bit_shift"]: broadcastBatcher("bit_shift"),
  ["neg"]: unopBatcher("neg"),
  ["reciprocal"]: unopBatcher("reciprocal"),
  ["floor"]: unopBatcher("floor"),
  ["ceil"]: unopBatcher("ceil"),
  ["stop_gradient"]: unopBatcher("stop_gradient"),
  ["cast"]: unopBatcher("cast"),
  ["bitcast"]: unopBatcher("bitcast"),
  ["sin"]: unopBatcher("sin"),
  ["cos"]: unopBatcher("cos"),
  ["asin"]: unopBatcher("asin"),
  ["atan"]: unopBatcher("atan"),
  ["exp"]: unopBatcher("exp"),
  ["log"]: unopBatcher("log"),
  ["erf"]: unopBatcher("erf"),
  ["erfc"]: unopBatcher("erfc"),
  ["sqrt"]: unopBatcher("sqrt"),
  ["reduce"](axisSize, [x], [xBdim], { op, axis }) {
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
    return [[reduce(x, op, newAxis)], [outBdim]];
  },
  ["dot"](axisSize, [x, y], [xBdim, yBdim]) {
    x = moveBatchAxis(axisSize, xBdim, x.ndim - (xBdim === null ? 1 : 2), x);
    y = moveBatchAxis(axisSize, yBdim, y.ndim - (yBdim === null ? 1 : 2), y);
    const z = dot$2(x, y);
    return [[z], [z.ndim - 1]];
  },
  ["conv"](axisSize, [x, y], [xBdim, yBdim], params) {
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    y = moveBatchAxis(axisSize, yBdim, 0, y);
    return [[conv$1(x, y, {
      ...params,
      vmapDims: params.vmapDims + 1
    })], [0]];
  },
  ["compare"]: broadcastBatcher("compare"),
  ["where"]: broadcastBatcher("where"),
  ["concatenate"](axisSize, xs, xBdims, { axis }) {
    const minBdim = Math.min(...xBdims.filter((d) => d !== null));
    xs = xs.map((x, i) => moveBatchAxis(axisSize, xBdims[i], minBdim, x));
    const newAxis = axis + (minBdim <= axis ? 1 : 0);
    return [[concatenate$1(xs, newAxis)], [minBdim]];
  },
  ["split"](axisSize, [x], [xBdim], { axis, sizes }) {
    const outs = split$2(x, axis + (xBdim <= axis ? 1 : 0), sizes);
    return [outs, rep(outs.length, xBdim)];
  },
  ["random_bits"](axisSize, [k0, k1], [bdim0, bdim1], { shape: shape2, mode }) {
    k0 = moveBatchAxis(axisSize, bdim0, 0, k0);
    k1 = moveBatchAxis(axisSize, bdim1, 0, k1);
    return [[randomBits(k0, k1, [axisSize, ...shape2], mode)], [0]];
  },
  ["gather"](axisSize, [x, ...indices2], [xBdim, ...indicesBdim], { axis, outDim, uniqueIndices }) {
    if (indicesBdim.every((d) => d === null)) {
      const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
      let newBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
      let newOutDim = outDim;
      if (newOutDim < newBdim) newBdim += axis.length;
      else newOutDim += 1;
      return [[gather(x, indices2, newAxis, newOutDim, uniqueIndices)], [newBdim]];
    }
    const [batchedIndices, indexRank] = batchIndexArrays(axisSize, indices2, indicesBdim);
    indices2 = batchedIndices;
    if (xBdim === null) return [[gather(x, indices2, axis, outDim, false)], [outDim]];
    else {
      x = moveBatchAxis(axisSize, xBdim, 0, x);
      const newAxis = [0, ...axis.map((ax) => ax + 1)];
      const extraBatchIndex = arange(axisSize).reshape([-1, ...rep(indexRank - 1, 1)]);
      indices2.splice(0, 0, extraBatchIndex);
      return [[gather(x, indices2, newAxis, outDim, uniqueIndices)], [outDim]];
    }
  },
  ["transpose"](axisSize, [x], [xBdim], { perm }) {
    const newPerm = perm.map((p) => p + (xBdim <= p ? 1 : 0));
    newPerm.splice(xBdim, 0, xBdim);
    return [[transpose$1(x, newPerm)], [xBdim]];
  },
  ["broadcast"](axisSize, [x], [xBdim], { shape: shape2, axis }) {
    return [[broadcast(x, shape2.toSpliced(xBdim, 0, axisSize), axis.map((ax) => ax + (xBdim <= ax ? 1 : 0)))], [xBdim]];
  },
  ["reshape"](axisSize, [x], [xBdim], { shape: shape2 }) {
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    return [[reshape$1(x, [axisSize, ...shape2])], [0]];
  },
  ["flip"](axisSize, [x], [xBdim], { axis }) {
    return [[flip$1(x, axis.map((ax) => ax + (xBdim <= ax ? 1 : 0)))], [xBdim]];
  },
  ["shrink"](axisSize, [x], [xBdim], { slice }) {
    return [[shrink(x, slice.toSpliced(xBdim, 0, [0, axisSize]))], [xBdim]];
  },
  ["pad"](axisSize, [x], [xBdim], { width }) {
    return [[pad$1(x, width.toSpliced(xBdim, 0, [0, 0]))], [xBdim]];
  },
  ["sort"]: lastDimsBatcher("sort", 1),
  ["argsort"]: lastDimsBatcher("argsort", 1, 2),
  ["scatter"]: (axisSize, [updates, ...indices2], [updatesBdim, ...indicesBdim], { op, shape: shape2, axis, outDim, uniqueIndices }) => {
    const [batchedIndices, indexRank] = batchIndexArrays(axisSize, indices2, indicesBdim);
    indices2 = batchedIndices;
    updates = moveBatchAxis(axisSize, updatesBdim, outDim, updates);
    const batchIndex = arange(axisSize).reshape([axisSize, ...rep(indexRank - 1, 1)]);
    return [[scatter(updates, [batchIndex, ...indices2], {
      op,
      shape: [axisSize, ...shape2],
      axis: [0, ...axis.map((a) => a + 1)],
      outDim,
      uniqueIndices
    })], [0]];
  },
  ["triangular_solve"](axisSize, [a, b], [aBdim, bBdim], { unitDiagonal }) {
    if (aBdim === null) {
      b = moveBatchAxis(axisSize, bBdim, -3, b);
      const [s, m, n] = b.shape.slice(-3);
      b = b.reshape([
        ...b.shape.slice(0, -3),
        s * m,
        n
      ]);
      let x = bind1("triangular_solve", [a, b], { unitDiagonal });
      x = x.reshape([
        ...b.shape.slice(0, -2),
        s,
        m,
        n
      ]);
      return [[x], [x.ndim - 3]];
    }
    a = moveBatchAxis(axisSize, aBdim, 0, a);
    b = moveBatchAxis(axisSize, bBdim, 0, b);
    return [[bind1("triangular_solve", [a, b], { unitDiagonal })], [0]];
  },
  ["cholesky"]: lastDimsBatcher("cholesky", 2),
  ["lu"]: lastDimsBatcher("lu", 2, 3),
  ["jacobi_eigh"]: lastDimsBatcher("jacobi_eigh", 2, 2),
  ["fft"](axisSize, [real, imag], [realBdim, imagBdim], params) {
    real = moveBatchAxis(axisSize, realBdim, 0, real);
    imag = moveBatchAxis(axisSize, imagBdim, 0, imag);
    return [bind("fft", [real, imag], params), [0, 0]];
  },
  ["jit"](axisSize, args, dims, { name, jaxpr }) {
    const newJaxpr = vmapJaxpr(jaxpr, axisSize, dims);
    const outs = bind("jit", [...newJaxpr.consts.map((c) => c.ref), ...args], {
      name: `${name}_vmap`,
      jaxpr: newJaxpr.jaxpr,
      numConsts: newJaxpr.consts.length
    });
    return [outs, rep(outs.length, 0)];
  }
};
var vmapJaxprCache = /* @__PURE__ */ new Map();
function vmapJaxpr(jaxpr, axisSize, dims) {
  const cacheKey = JSON.stringify([axisSize, dims]);
  const prevResult = vmapJaxprCache.get(jaxpr)?.get(cacheKey);
  if (prevResult) return prevResult;
  const inAvals = jaxpr.inBinders.map((v, i) => {
    if (dims[i] === null) return v.aval;
    const shape2 = [...v.aval.shape];
    shape2.splice(dims[i], 0, axisSize);
    return new ShapedArray(shape2, v.aval.dtype, v.aval.weakType);
  });
  const { jaxpr: newJaxpr } = makeJaxpr$1((args) => vmapFlat(jaxprAsFun(jaxpr), dims, args))(inAvals);
  if (!vmapJaxprCache.has(jaxpr)) vmapJaxprCache.set(jaxpr, /* @__PURE__ */ new Map());
  vmapJaxprCache.get(jaxpr).set(cacheKey, newJaxpr);
  return newJaxpr;
}
function vmapFlat(f, inAxes, args) {
  let axisSize = void 0;
  for (let i = 0; i < args.length; i++) if (inAxes[i] !== null) {
    const arg = args[i];
    if (!(arg instanceof Tracer)) throw new TypeError("vmap requires Tracer argument for mapped axes");
    const size2 = arg.shape[inAxes[i]];
    if (axisSize === void 0) axisSize = size2;
    else if (axisSize !== size2) throw new TypeError("vmap requires all mapped axes to have the same size");
  }
  if (axisSize === void 0) throw new TypeError("vmap requires at least one mapped axis");
  let valsOut, bdimsOut;
  try {
    var _usingCtx$2 = _usingCtx();
    const trace2 = new BatchTrace(_usingCtx$2.u(newMain(BatchTrace, axisSize)));
    const tracersOut = f(...args.map((x, i) => inAxes[i] === null ? pureArray(x) : new BatchTracer(trace2, pureArray(x), inAxes[i]))).map((out) => fullRaise(trace2, out));
    [valsOut, bdimsOut] = unzip2(tracersOut.map((t) => [t.val, t.batchDim]));
  } catch (_) {
    _usingCtx$2.e = _;
  } finally {
    _usingCtx$2.d();
  }
  return zip(valsOut, bdimsOut).map(([valOut, bdim]) => moveBatchAxis(axisSize, bdim, 0, valOut));
}
function vmap$1(f, inAxes = 0) {
  return (...args) => {
    const [argsFlat, inTree] = flatten(args);
    let inAxesFlat = [];
    if (typeof inAxes === "number") inAxesFlat = rep(argsFlat.length, inAxes);
    else for (let i = 0; i < args.length; i++) if (inAxes[i] == null) inAxesFlat.push(...rep(inTree.childTreedefs[i].size, null));
    else if (typeof inAxes[i] === "number") inAxesFlat.push(...rep(inTree.childTreedefs[i].size, inAxes[i]));
    else {
      const [axesFlat, axesTreeDef] = flatten(inAxes[i]);
      if (!inTree.childTreedefs[i].equals(axesTreeDef)) throw new TreeMismatchError("vmap", inTree.childTreedefs[i], axesTreeDef);
      inAxesFlat.push(...axesFlat);
    }
    const [fFlat, outTree] = flattenFun(f, inTree);
    const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
    if (outTree.value === void 0) throw new Error("outTree was not set in vmap");
    return unflatten(outTree.value, outsFlat);
  };
}
var JVPTracer = class extends Tracer {
  primal;
  tangent;
  constructor(trace2, primal, tangent) {
    super(trace2);
    this.primal = primal;
    this.tangent = tangent;
  }
  get aval() {
    return this.primal.aval;
  }
  toString() {
    return `JVPTracer(${this.primal.toString()}, ${this.tangent.toString()})`;
  }
  get ref() {
    this.primal.ref, this.tangent.ref;
    return this;
  }
  dispose() {
    this.primal.dispose();
    this.tangent.dispose();
  }
};
var JVPTrace = class extends Trace {
  pure(val) {
    return this.lift(pureArray(val));
  }
  lift(val) {
    return new JVPTracer(this, val, zerosLike$1(val.ref));
  }
  processPrimitive(primitive, tracers, params) {
    const [primalsIn, tangentsIn] = unzip2(tracers.map((x) => [x.primal, x.tangent]));
    const jvpRule = jvpRules[primitive];
    if (jvpRule === void 0) throw new Error(`No JVP rule for: ${primitive}`);
    const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
    return zip(primalsOut, tangentsOut).map(([x, t]) => new JVPTracer(this, x, t));
  }
};
function linearTangentsJvp(primitive) {
  return (primals, tangents, params) => {
    return [bind(primitive, primals, params), bind(primitive, tangents, params)];
  };
}
function bilinearTangentsJvp(primitive) {
  return ([x, y], [dx, dy], params) => {
    const primal = bind1(primitive, [x.ref, y.ref], params);
    const tangent = bind1(primitive, [x, dy], params).add(bind1(primitive, [dx, y], params));
    return [[primal], [tangent]];
  };
}
function zeroTangentsJvp(primitive) {
  return (primals, tangents, params) => {
    for (const t of tangents) t.dispose();
    const ys = bind(primitive, primals, params);
    return [ys, ys.map((y) => zerosLike$1(y.ref))];
  };
}
function takeAlongLastAxis(x, indices2) {
  const coords = [];
  for (let axis = 0; axis < x.ndim - 1; axis++) {
    const shape2 = rep(x.ndim, 1);
    shape2[axis] = x.shape[axis];
    coords.push(arange(x.shape[axis]).reshape(shape2));
  }
  coords.push(indices2);
  return gather(x, coords, range(x.ndim), 0, true);
}
function batchMatmulT(a, b) {
  return dot$2(a.reshape(a.shape.toSpliced(-1, 0, 1)), b.reshape(b.shape.toSpliced(-2, 0, 1)));
}
function mT(a) {
  return moveaxis$1(a, -2, -1);
}
function sliceAxis(a, axis, p) {
  const slices = Array(a.shape.length).fill([]);
  slices[checkAxis(axis, a.ndim)] = p;
  return a.slice(...slices);
}
function padAxis(a, axis, p) {
  const pads = Array(a.shape.length).fill([0, 0]);
  pads[checkAxis(axis, a.ndim)] = p;
  return pad$1(a, pads);
}
var jvpRules = {
  ["add"]: linearTangentsJvp("add"),
  ["mul"]: bilinearTangentsJvp("mul"),
  ["idiv"]: zeroTangentsJvp("idiv"),
  ["mod"]([x, y], [dx, dy]) {
    if (!isFloatDtype(x.dtype) && !isFloatDtype(y.dtype)) {
      dx.dispose();
      dy.dispose();
      return [[x.ref, y.ref], [zerosLike$1(x), zerosLike$1(y)]];
    }
    const q = idiv(x.ref, y.ref);
    return [[mod(x, y)], [dx.sub(dy.mul(q))]];
  },
  ["min"]([x, y], [dx, dy]) {
    return [[min$1(x.ref, y.ref)], [where$1(less$1(y, x), dy, dx)]];
  },
  ["max"]([x, y], [dx, dy]) {
    return [[max$1(x.ref, y.ref)], [where$1(less$1(x, y), dy, dx)]];
  },
  ["bit_combine"]: zeroTangentsJvp("bit_combine"),
  ["bit_shift"]: zeroTangentsJvp("bit_shift"),
  ["neg"]: linearTangentsJvp("neg"),
  ["reciprocal"]([x], [dx]) {
    const xRecip = reciprocal$1(x.ref);
    return [[xRecip.ref], [neg(xRecip.ref.mul(xRecip)).mul(dx)]];
  },
  ["floor"]: zeroTangentsJvp("floor"),
  ["ceil"]: zeroTangentsJvp("ceil"),
  ["stop_gradient"]: zeroTangentsJvp("stop_gradient"),
  ["cast"]([x], [dx], { dtype }) {
    if (x.dtype === dtype) return [[x], [dx]];
    if (isFloatDtype(dtype) && isFloatDtype(x.dtype)) return [[cast(x, dtype)], [cast(dx, dtype)]];
    else {
      dx.dispose();
      return [[cast(x.ref, dtype)], [zerosLike$1(x)]];
    }
  },
  ["bitcast"]([x], [dx], { dtype }) {
    if (x.dtype === dtype) return [[x], [dx]];
    dx.dispose();
    return [[bitcast(x.ref, dtype)], [zerosLike$1(x)]];
  },
  ["sin"]([x], [dx]) {
    return [[sin$1(x.ref)], [cos$1(x).mul(dx)]];
  },
  ["cos"]([x], [dx]) {
    return [[cos$1(x.ref)], [neg(sin$1(x)).mul(dx)]];
  },
  ["asin"]([x], [dx]) {
    const denom = sqrt$1(reciprocal$1(cast(1, x.dtype).sub(x.ref.mul(x.ref))));
    return [[asin$1(x)], [denom.mul(dx)]];
  },
  ["atan"]([x], [dx]) {
    const denom = cast(1, x.dtype).add(x.ref.mul(x.ref));
    return [[atan$1(x)], [dx.div(denom)]];
  },
  ["exp"]([x], [dx]) {
    const z = exp$1(x);
    return [[z.ref], [z.mul(dx)]];
  },
  ["log"]([x], [dx]) {
    return [[log$1(x.ref)], [reciprocal$1(x).mul(dx)]];
  },
  ["erf"]([x], [dx]) {
    const coeff = 2 / Math.sqrt(Math.PI);
    const expTerm = exp$1(neg(x.ref.mul(x.ref)));
    return [[erf$1(x)], [expTerm.mul(coeff).mul(dx)]];
  },
  ["erfc"]([x], [dx]) {
    const coeff = -2 / Math.sqrt(Math.PI);
    const expTerm = exp$1(neg(x.ref.mul(x.ref)));
    return [[erfc$1(x)], [expTerm.mul(coeff).mul(dx)]];
  },
  ["sqrt"]([x], [dx]) {
    const z = sqrt$1(x);
    return [[z.ref], [reciprocal$1(z.mul(2)).mul(dx)]];
  },
  ["reduce"]([x], [dx], { op, axis }) {
    if (op === "Add") return [[reduce(x, op, axis)], [reduce(dx, op, axis)]];
    else if (op === "Mul") {
      const primal = reduce(x.ref, op, axis);
      const tangent = broadcast(primal.ref, x.shape, axis).mul(reciprocal$1(x)).mul(dx).sum(axis);
      return [[primal], [tangent]];
    } else if (op === "Min" || op === "Max") {
      const primal = reduce(x.ref, op, axis);
      const notMin = notEqual$1(x, broadcast(primal.ref, x.shape, axis));
      const minCount = where$1(notMin.ref, 0, 1).sum(axis);
      const tangent = where$1(notMin, 0, dx).sum(axis).div(minCount);
      return [[primal], [tangent]];
    } else throw new Error(`JVP rule not implemented for reduce op: ${op}`);
  },
  ["pool"]: linearTangentsJvp("pool"),
  ["pool_transpose"]: linearTangentsJvp("pool_transpose"),
  ["dot"]: bilinearTangentsJvp("dot"),
  ["conv"]: bilinearTangentsJvp("conv"),
  ["compare"]: zeroTangentsJvp("compare"),
  ["where"]([cond, x, y], [dcond, dx, dy]) {
    dcond.dispose();
    return [[where$1(cond.ref, x, y)], [where$1(cond, dx, dy)]];
  },
  ["concatenate"]: linearTangentsJvp("concatenate"),
  ["split"]: linearTangentsJvp("split"),
  ["random_bits"]: zeroTangentsJvp("random_bits"),
  ["gather"]([x, ...indices2], [dx, ...dindices], params) {
    dindices.forEach((index) => index.dispose());
    const tangentIndices = indices2.map((index) => index.ref);
    return [bind("gather", [x, ...indices2], params), bind("gather", [dx, ...tangentIndices], params)];
  },
  ["transpose"]: linearTangentsJvp("transpose"),
  ["broadcast"]: linearTangentsJvp("broadcast"),
  ["reshape"]: linearTangentsJvp("reshape"),
  ["flip"]: linearTangentsJvp("flip"),
  ["shrink"]: linearTangentsJvp("shrink"),
  ["pad"]: linearTangentsJvp("pad"),
  ["sort"]([x], [dx]) {
    const [y, idx] = argsort$1(x);
    return [[y], [takeAlongLastAxis(dx, idx)]];
  },
  ["argsort"]([x], [dx]) {
    const [y, idx] = argsort$1(x);
    return [[y, idx.ref], [takeAlongLastAxis(dx, idx.ref), zerosLike$1(idx)]];
  },
  ["scatter"]([updates, ...indices2], [dupdates, ...dindices], params) {
    dindices.forEach((index) => index.dispose());
    const tangentIndices = indices2.map((index) => index.ref);
    return [[scatter(updates, indices2, params)], [scatter(dupdates, tangentIndices, params)]];
  },
  ["triangular_solve"]([a, b], [da, db], { unitDiagonal }) {
    const x = triangularSolve$1(a.ref, b, { unitDiagonal });
    da = unitDiagonal ? triu(da, 1) : triu(da);
    const dax = batchMatmulT(da, x.ref);
    const dx = triangularSolve$1(a, db.sub(mT(dax)), { unitDiagonal });
    return [[x], [dx]];
  },
  ["cholesky"]([a], [da]) {
    const L = cholesky$2(a.ref);
    da = da.ref.add(mT(da)).mul(0.5);
    const W = triangularSolve$1(L.ref, da, { lower: true });
    const ST = triangularSolve$1(L.ref, mT(W), { lower: true });
    const dL = batchMatmulT(L.ref, triu(ST.ref, 1).add(triu(ST)).mul(0.5));
    return [[L], [dL]];
  },
  ["lu"]([a], [da]) {
    const [luMatrix, pivots, permutation] = lu$1(a);
    const [m, n] = a.shape.slice(-2);
    const k = Math.min(m, n);
    const lLower = tril(sliceAxis(luMatrix.ref, -1, [0, k]), -1);
    const L = (m > k ? padAxis(lLower, -1, [0, m - k]) : lLower).add(eye(m));
    const uUpper = triu(sliceAxis(luMatrix.ref, -2, [0, k]));
    const uPadded = n > k ? padAxis(uUpper, -2, [0, n - k]) : uUpper;
    const uEye = n > k ? padAxis(padAxis(eye(n - k), -1, [k, 0]), -2, [k, 0]) : zerosLike$1(uPadded.ref);
    const U = uPadded.add(uEye);
    const pda = batchMatmulT(permutation.ref.reshape([...permutation.shape, 1]).equal(arange(m)).astype(da.dtype), mT(da));
    const la = mT(triangularSolve$1(L.ref, mT(pda), {
      lower: true,
      unitDiagonal: true
    }));
    const lau = triangularSolve$1(mT(U.ref), la, { lower: true });
    const lDot = batchMatmulT(L, mT(tril(lau.ref, -1)));
    const uDot = batchMatmulT(triu(lau), mT(U));
    return [[
      luMatrix,
      pivots,
      permutation
    ], [
      lDot.add(uDot),
      zerosLike$1(pivots.ref),
      zerosLike$1(permutation.ref)
    ]];
  },
  ["jacobi_eigh"]() {
    throw new Error("JVP rule not implemented for jacobi_eigh");
  },
  ["fft"]: linearTangentsJvp("fft"),
  ["jit"](primals, tangents, { name, jaxpr }) {
    const newJaxpr = jvpJaxpr(jaxpr);
    const outs = bind("jit", [
      ...newJaxpr.consts.map((c) => c.ref),
      ...primals,
      ...tangents
    ], {
      name: `${name}_jvp`,
      jaxpr: newJaxpr.jaxpr,
      numConsts: newJaxpr.consts.length
    });
    const n = outs.length / 2;
    if (!Number.isInteger(n)) throw new Error("internal: JVP Jaxpr output length is not even");
    const [primalsOut, tangentsOut] = [outs.slice(0, n), outs.slice(n)];
    return [primalsOut, tangentsOut];
  }
};
var jvpJaxprCache = /* @__PURE__ */ new Map();
function jvpJaxpr(jaxpr) {
  if (jvpJaxprCache.has(jaxpr)) return jvpJaxprCache.get(jaxpr);
  const inAvals = jaxpr.inBinders.map((v) => v.aval);
  const { jaxpr: newJaxpr } = makeJaxpr$1((primals, tangents) => jvpFlat(jaxprAsFun(jaxpr), primals, tangents))(inAvals, inAvals);
  jvpJaxprCache.set(jaxpr, newJaxpr);
  return newJaxpr;
}
function jvpFlat(f, primals, tangents) {
  try {
    var _usingCtx$1 = _usingCtx();
    const trace2 = new JVPTrace(_usingCtx$1.u(newMain(JVPTrace)));
    return unzip2(f(...zip(primals, tangents).map(([x, t]) => new JVPTracer(trace2, pureArray(x), pureArray(t)))).map((out) => fullRaise(trace2, out)).map((t) => [t.primal, t.tangent]));
  } catch (_) {
    _usingCtx$1.e = _;
  } finally {
    _usingCtx$1.d();
  }
}
var bprod = (...xs) => xs.reduce((acc, x) => acc * BigInt(x), 1n);
var uniq = (arr) => Array.from(new Set(arr));
var EINSUM_COMPONENT_RE = /\p{ID_Start}|\.\.\./gu;
var einsumParseCache = /* @__PURE__ */ new Map();
function parseEinsumExpression(expr, shapes) {
  return runWithCache(einsumParseCache, [expr, shapes], () => {
    const idents = [...expr.split("->")[0].matchAll(EINSUM_COMPONENT_RE).map((m) => m[0]).filter((c) => c !== "...")];
    if (!expr.includes("->")) {
      const counts = /* @__PURE__ */ new Map();
      for (const c of idents) counts.set(c, (counts.get(c) ?? 0) + 1);
      const outputIndices = Array.from(counts.entries()).filter(([, count]) => count === 1).map(([char]) => char).sort();
      if (expr.includes("...")) outputIndices.splice(0, 0, "...");
      expr += "->" + outputIndices.join("");
    }
    const identToIndex = new Map(uniq(idents).sort().map((c, i) => [c, i]));
    const componentsToIndices = (components, rank) => components.flatMap((c) => {
      if (c === "...") {
        const start = rank !== void 0 ? components.length - 1 + ellipsisRank - rank : 0;
        return range(identToIndex.size + start, identToIndex.size + ellipsisRank);
      }
      return identToIndex.get(c);
    });
    let ellipsisRank = 0;
    const [lhs, rhs] = expr.split("->");
    const lhsComponents = lhs.split(",").map((part) => [...part.matchAll(EINSUM_COMPONENT_RE).map((m) => m[0])]);
    const rhsComponents = [...rhs.matchAll(EINSUM_COMPONENT_RE)].map((m) => m[0]);
    for (const [i, components] of lhsComponents.entries()) {
      const shape2 = shapes[i];
      const ellipsisIndex = components.indexOf("...");
      if (ellipsisIndex !== -1) {
        if (components.lastIndexOf("...") !== ellipsisIndex) throw new Error("Multiple ellipses in one einsum operand is not allowed");
        const numExplicit = components.length - 1;
        if (shape2.length < numExplicit) throw new Error(`Einsum operand ${i} has shape ${JSON.stringify(shape2)} but indexed with "${components.join("")}"`);
        ellipsisRank = Math.max(ellipsisRank, shape2.length - numExplicit);
      }
    }
    return {
      shapes,
      lhsIndices: lhsComponents.map((components, i) => componentsToIndices(components, shapes[i].length)),
      rhsIndex: componentsToIndices(rhsComponents)
    };
  });
}
var EinsumPath = class {
  /** Parsed and normalized input for the einsum. */
  input;
  /** Mapping of each index number to its size in the shape array. */
  sizeMap;
  /**
  * A list of tensor contractions.
  *
  * This is ordered by operation order. Each entry corresponds to a single
  * elementwise product and/or inner contraction between two tensors, and it
  * contains the indices of the tensors to be contracted.
  *
  * The indices of input tensors are [0..n), and each intermediate from the
  * path at index i produces a new tensor at index n + i at the end
  * (opt_einsum internally calls this "SSA form").
  *
  * Invariants:
  * - Each group in the path consists of two tensors.
  * - For n input tensors, there are n-1 groups in the path.
  * - Every tensor must be in the path exactly once, except the final output.
  *
  * @example
  * Given einsum for `(A, B, C)`, this path corresponds to `(A, B)` and then
  * `(AB, C)`.
  * ```
  * [[0, 1], [3, 2]]
  * ```
  */
  path;
  constructor(input, sizeMap, path) {
    this.input = input;
    this.sizeMap = sizeMap;
    this.path = path;
  }
  /** Shape of the final output tensor. */
  get outputShape() {
    return this.input.rhsIndex.map((i) => this.sizeMap.get(i));
  }
  /** Estimate the number of FLOPs to execute this einsum path. */
  get approximateFlops() {
    return approximatePathFlops(this.input, this.sizeMap, this.path);
  }
};
function approximatePathFlops(input, sizeMap, path) {
  if (path.length == 0) {
    const [indices3] = input.lhsIndices;
    return bprod(...uniq(indices3).map((i) => sizeMap.get(i)));
  }
  const indexUsageCounts = [];
  for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
  const indices2 = [...input.lhsIndices];
  let totalFlops = 0n;
  for (const tensorGroup of path) {
    const indexReduced = [];
    const indexGroup = [];
    for (const tensorIdx of tensorGroup) for (const idx of indices2[tensorIdx]) {
      if (!indexGroup.includes(idx)) indexGroup.push(idx);
      if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
    }
    totalFlops += approximateCountFlops(indexGroup, indexReduced.length > 0, tensorGroup.length, sizeMap);
    const newIndex = indexGroup.filter((x) => !indexReduced.includes(x));
    for (const idx of newIndex) indexUsageCounts[idx]++;
    indices2.push(newIndex);
  }
  return totalFlops;
}
function approximateCountFlops(indexGroup, hasReduction, numTerms, sizeMap) {
  return bprod(...indexGroup.map((i) => sizeMap.get(i))) * (BigInt(numTerms) - 1n + (hasReduction ? 1n : 0n));
}
function computeSizeMap({ shapes, lhsIndices, rhsIndex }) {
  if (shapes.length === 0) throw new Error("Einsum must have at least one input tensor");
  if (lhsIndices.length !== shapes.length) throw new Error(`Mismatched number of lhs operands (${lhsIndices.length}) and shapes (${shapes.length})`);
  for (let i = 0; i < shapes.length; i++) if (lhsIndices[i].length !== shapes[i].length) throw new Error(`Mismatched number of indices (${lhsIndices[i].length}) and shape (${JSON.stringify(shapes[i])}) for operand ${i}`);
  const rhsIndexSet = /* @__PURE__ */ new Set();
  for (const idx of rhsIndex) {
    if (rhsIndexSet.has(idx)) throw new Error(`Repeated index ${idx} in einsum output`);
    rhsIndexSet.add(idx);
  }
  const sizeMap = /* @__PURE__ */ new Map();
  for (let i = 0; i < shapes.length; i++) {
    const shape2 = shapes[i];
    const lhsIndex = lhsIndices[i];
    for (let j = 0; j < lhsIndex.length; j++) {
      const idx = lhsIndex[j];
      const dim = shape2[j];
      const existing = sizeMap.get(idx);
      if (existing === void 0 || existing === 1) sizeMap.set(idx, dim);
      else if (existing !== dim && dim !== 1) throw new Error(`Inconsistent size for index ${idx} in einsum: ${existing} vs ${dim}`);
    }
  }
  for (const [idx, size2] of sizeMap) if (!Number.isInteger(idx) || idx < 0) throw new Error(`Invalid index ${idx} in einsum expression, must be non-negative integer`);
  else if (size2 < 0) throw new Error(`Invalid size ${size2} for index ${idx} in einsum expression, must be non-negative`);
  for (const idx of rhsIndex) if (!sizeMap.has(idx)) throw new Error(`Output index ${idx} not present in einsum inputs`);
  return sizeMap;
}
var einsumPathCache = /* @__PURE__ */ new Map();
function computeEinsumPath(input, method) {
  if (!method) method = input.shapes.length <= 5 ? "optimal" : "naive";
  return runWithCache(einsumPathCache, [input, method], () => {
    const sizeMap = computeSizeMap(input);
    if (input.shapes.length === 1) return new EinsumPath(input, sizeMap, []);
    switch (method) {
      case "naive":
        return computePathNaive(input, sizeMap);
      case "optimal":
        return computePathOptimal(input, sizeMap);
      default:
        throw new Error(`Unknown computePath method: ${method}`);
    }
  });
}
function computePathNaive(input, sizeMap) {
  const n = input.shapes.length;
  const path = [];
  let lastTensorIndex = 0;
  for (let i = 1; i < n; i++) {
    path.push([lastTensorIndex, i]);
    lastTensorIndex = n + i - 1;
  }
  return new EinsumPath(input, sizeMap, path);
}
function computePathOptimal(input, sizeMap) {
  const n = input.shapes.length;
  let bestPath = null;
  let bestFlops = null;
  for (const path of allPaths(range(n), n)) {
    const flops = approximatePathFlops(input, sizeMap, path);
    if (bestFlops === null || flops < bestFlops) {
      bestPath = path;
      bestFlops = flops;
    }
  }
  return new EinsumPath(input, sizeMap, bestPath);
}
function* allPaths(tensors, next) {
  if (tensors.length === 2) {
    yield [[tensors[0], tensors[1]]];
    return;
  }
  for (let i = 0; i < tensors.length; i++) for (let j = i + 1; j < tensors.length; j++) {
    const pair = [tensors[i], tensors[j]];
    const newTensors = tensors.filter((t) => t !== pair[0] && t !== pair[1]);
    newTensors.push(next);
    for (const subpath of allPaths(newTensors, next + 1)) yield [pair, ...subpath];
  }
}
var numpy_fft_exports = /* @__PURE__ */ __exportAll({
  fft: () => fft,
  fft2: () => fft2,
  fftfreq: () => fftfreq,
  fftn: () => fftn,
  fftshift: () => fftshift,
  hfft: () => hfft,
  ifft: () => ifft,
  ifft2: () => ifft2,
  ifftn: () => ifftn,
  ifftshift: () => ifftshift,
  ihfft: () => ihfft,
  irfft: () => irfft,
  irfft2: () => irfft2,
  irfftn: () => irfftn,
  rfft: () => rfft,
  rfft2: () => rfft2,
  rfftfreq: () => rfftfreq,
  rfftn: () => rfftn
});
function checkPairInput(name, a) {
  const fullName = `jax.numpy.fft.${name}`;
  if (!deepEqual(a.real.shape, a.imag.shape)) throw new Error(`${fullName}: real and imaginary parts must have the same shape, got ${JSON.stringify(a.real.shape)} and ${JSON.stringify(a.imag.shape)}`);
  if (a.real.dtype !== a.imag.dtype) throw new Error(`${fullName}: real and imaginary parts must have the same dtype, got ${a.real.dtype} and ${a.imag.dtype}`);
  if (!isFloatDtype(a.real.dtype)) throw new Error(`${fullName}: input must have a float dtype, got ${a.real.dtype}`);
}
function factorFftSize(n) {
  const factors = [];
  for (const radix of [
    4,
    2,
    3,
    5,
    7
  ]) while (n % radix === 0) {
    factors.push(radix);
    n /= radix;
  }
  for (let radix = 11; radix * radix <= n; radix += 2) while (n % radix === 0) {
    factors.push(radix);
    n /= radix;
  }
  if (n > 1) factors.push(n);
  return factors;
}
function checkRealInput(name, a) {
  if (!isFloatDtype(a.dtype)) throw new Error(`jax.numpy.fft.${name}: input must have a float dtype, got ${a.dtype}`);
}
function checkFrequencyArgs(name, n, d) {
  if (!Number.isInteger(n) || n < 1) throw new Error(`jax.numpy.fft.${name}: n must be a positive integer, got ${n}`);
  if (!Number.isFinite(d) || d === 0) throw new Error(`jax.numpy.fft.${name}: d must be a finite non-zero number, got ${d}`);
}
function sliceAlongAxis(a, axis, start, end) {
  const index = globalThis.Array.from({ length: a.ndim }, () => []);
  index[axis] = [start, end];
  return a.slice(...index);
}
function fft(a, axis = -1) {
  return fftChecked("fft", a, axis, false);
}
function fftChecked(name, a, axis, inverse) {
  checkPairInput(name, a);
  axis = checkAxis(axis, a.real.ndim);
  const n = a.real.shape[axis];
  if (!Number.isInteger(n) || n < 1) throw new Error(`jax.numpy.fft.${name}: size must be a positive integer, got ${n}`);
  return fftRoutine(a, axis, inverse);
}
function fftRoutine(a, axis, inverse) {
  let { real, imag } = a;
  const n = real.shape[axis];
  let perm = null;
  if (axis !== real.ndim - 1) {
    perm = range(real.ndim);
    perm.splice(axis, 1);
    perm.push(axis);
    real = real.transpose(perm);
    imag = imag.transpose(perm);
  }
  [real, imag] = fft$1(real, imag, {
    factors: factorFftSize(n),
    inverse
  });
  if (perm !== null) {
    real = real.transpose(invertPermutation(perm));
    imag = imag.transpose(invertPermutation(perm));
  }
  return {
    real,
    imag
  };
}
function transformN(name, a, axes, transform) {
  checkPairInput(name, a);
  const normalizedAxes = normalizeAxis(axes, a.real.ndim, false);
  let result = a;
  for (const axis of normalizedAxes) result = transform(result, axis);
  return result;
}
function fftn(a, axes = null) {
  return transformN("fftn", a, axes, fft);
}
function fft2(a, axes = [-2, -1]) {
  return fftn(a, axes);
}
function ifft(a, axis = -1) {
  return fftChecked("ifft", a, axis, true);
}
function ifftn(a, axes = null) {
  return transformN("ifftn", a, axes, ifft);
}
function ifft2(a, axes = [-2, -1]) {
  return ifftn(a, axes);
}
function rfft(a, axis = -1) {
  checkRealInput("rfft", a);
  axis = checkAxis(axis, a.ndim);
  const n = a.shape[axis];
  const result = fft({
    real: a,
    imag: zerosLike(a.ref)
  }, axis);
  const stop = Math.floor(n / 2) + 1;
  return {
    real: sliceAlongAxis(result.real, axis, 0, stop),
    imag: sliceAlongAxis(result.imag, axis, 0, stop)
  };
}
function irfft(a, axis = -1) {
  checkPairInput("irfft", a);
  const { real, imag } = a;
  axis = checkAxis(axis, real.ndim);
  const m = real.shape[axis];
  if (m < 2) throw new Error(`jax.numpy.fft.irfft: packed input length must be at least 2, got ${m}`);
  const mirroredReal = flip(sliceAlongAxis(real.ref, axis, 1, m - 1), axis);
  const mirroredImag = flip(sliceAlongAxis(imag.ref, axis, 1, m - 1), axis).mul(-1);
  const result = ifft({
    real: concatenate([real, mirroredReal], axis),
    imag: concatenate([imag, mirroredImag], axis)
  }, axis);
  result.imag.dispose();
  return result.real;
}
function rfftn(a, axes = null) {
  checkRealInput("rfftn", a);
  const normalizedAxes = normalizeAxis(axes, a.ndim, false);
  if (normalizedAxes.length === 0) return {
    real: a,
    imag: zerosLike(a.ref)
  };
  const realAxis = normalizedAxes[normalizedAxes.length - 1];
  let result = rfft(a, realAxis);
  for (const axis of normalizedAxes.slice(0, -1)) result = fft(result, axis);
  return result;
}
function rfft2(a, axes = [-2, -1]) {
  return rfftn(a, axes);
}
function irfftn(a, axes = null) {
  checkPairInput("irfftn", a);
  const normalizedAxes = normalizeAxis(axes, a.real.ndim, false);
  if (normalizedAxes.length === 0) {
    a.imag.dispose();
    return a.real;
  }
  const realAxis = normalizedAxes[normalizedAxes.length - 1];
  let result = a;
  for (const axis of normalizedAxes.slice(0, -1)) result = ifft(result, axis);
  return irfft(result, realAxis);
}
function irfft2(a, axes = [-2, -1]) {
  return irfftn(a, axes);
}
function hfft(a, axis = -1) {
  checkPairInput("hfft", a);
  axis = checkAxis(axis, a.real.ndim);
  const n = 2 * (a.real.shape[axis] - 1);
  return irfft({
    real: a.real,
    imag: a.imag.mul(-1)
  }, axis).mul(n);
}
function ihfft(a, axis = -1) {
  checkRealInput("ihfft", a);
  axis = checkAxis(axis, a.ndim);
  const n = a.shape[axis];
  const result = rfft(a, axis);
  return {
    real: result.real.div(n),
    imag: result.imag.mul(-1).div(n)
  };
}
function fftfreq(n, d = 1) {
  checkFrequencyArgs("fftfreq", n, d);
  const scale = 1 / (n * d);
  const positiveEnd = Math.floor((n - 1) / 2) + 1;
  const values = [];
  for (let i = 0; i < positiveEnd; i++) values.push(i * scale);
  for (let i = -Math.floor(n / 2); i < 0; i++) values.push(i * scale);
  return array(values, { dtype: "float32" });
}
function rfftfreq(n, d = 1) {
  checkFrequencyArgs("rfftfreq", n, d);
  const scale = 1 / (n * d);
  const values = [];
  for (let i = 0; i <= Math.floor(n / 2); i++) values.push(i * scale);
  return array(values, { dtype: "float32" });
}
function fftshift(a, axes = null) {
  const normalizedAxes = normalizeAxis(axes, a.ndim, false);
  return roll(a, normalizedAxes.map((axis) => Math.floor(a.shape[axis] / 2)), normalizedAxes);
}
function ifftshift(a, axes = null) {
  const normalizedAxes = normalizeAxis(axes, a.ndim, false);
  return roll(a, normalizedAxes.map((axis) => -Math.floor(a.shape[axis] / 2)), normalizedAxes);
}
var numpy_linalg_exports = /* @__PURE__ */ __exportAll({
  cholesky: () => cholesky$1,
  cross: () => cross$1,
  det: () => det,
  diagonal: () => diagonal,
  eigh: () => eigh$1,
  eigvalsh: () => eigvalsh,
  inv: () => inv,
  lstsq: () => lstsq,
  matmul: () => matmul,
  matrixNorm: () => matrixNorm,
  matrixPower: () => matrixPower,
  matrixTranspose: () => matrixTranspose,
  multiDot: () => multiDot,
  norm: () => norm,
  outer: () => outer,
  slogdet: () => slogdet,
  solve: () => solve,
  svd: () => svd$1,
  svdvals: () => svdvals,
  tensordot: () => tensordot,
  trace: () => trace,
  vecdot: () => vecdot,
  vectorNorm: () => vectorNorm
});
function cholesky$1(a, { upper = false, symmetrizeInput = true } = {}) {
  a = fudgeArray(a);
  checkSquare("cholesky", a.shape);
  if (symmetrizeInput) a = a.ref.add(matrixTranspose(a)).mul(0.5);
  return cholesky(a, { upper });
}
function cross$1(x1, x2, axis = -1) {
  const a1 = checkAxis(axis, ndim(x1));
  const a2 = checkAxis(axis, ndim(x2));
  if (shape(x1)[a1] !== 3) throw new Error(`linalg.cross: x1 must have size 3 along axis ${axis}, got ${shape(x1)[a1]}`);
  if (shape(x2)[a2] !== 3) throw new Error(`linalg.cross: x2 must have size 3 along axis ${axis}, got ${shape(x2)[a2]}`);
  return cross(x1, x2, { axis });
}
function det(a) {
  a = fudgeArray(a);
  const n = checkSquare("det", a.shape);
  const [lu$2, pivots, permutation] = lu(a);
  permutation.dispose();
  const sign2 = pivots.notEqual(arange(n)).astype(int32).sum(-1).mod(2).mul(-2).add(1);
  return prod2(lu$2.diagonal(0, -1, -2), -1).mul(sign2);
}
function eigh$1(a, opts) {
  const [vectors, values] = eigh(a, { symmetrizeInput: opts?.symmetrizeInput });
  return [values, vectors];
}
function eigvalsh(a, opts) {
  const [vectors, values] = eigh(a, { symmetrizeInput: opts?.symmetrizeInput });
  vectors.dispose();
  return values;
}
function svd$1(a, opts) {
  const result = svd(a, {
    fullMatrices: opts?.fullMatrices,
    computeUv: opts?.computeUv
  });
  if (opts?.computeUv === false) return result;
  return result;
}
function svdvals(a) {
  return svd(a, { computeUv: false });
}
function inv(a) {
  a = fudgeArray(a);
  const n = checkSquare("inv", a.shape);
  return solve(a, eye(n, void 0, { dtype: a.dtype }));
}
function lstsq(a, b) {
  a = fudgeArray(a);
  b = fudgeArray(b);
  if (a.ndim !== 2) throw new Error(`lstsq: 'a' must be a 2D array, got ${a.aval}`);
  const [m, n] = a.shape;
  if (b.shape[0] !== m) throw new Error(`lstsq: leading dimension of 'b' must match number of rows of 'a', got ${b.aval}`);
  const at = matrixTranspose(a.ref);
  if (m <= n) {
    const l = cholesky$1(matmul(a, at.ref), { symmetrizeInput: false });
    return matmul(at, triangularSolve(l, triangularSolve(l.ref, b, {
      leftSide: true,
      lower: true
    }), {
      leftSide: true,
      lower: true,
      transposeA: true
    }));
  } else {
    const l = cholesky$1(matmul(at.ref, a), { symmetrizeInput: false });
    const atb = matmul(at, b);
    return triangularSolve(l, triangularSolve(l.ref, atb, {
      leftSide: true,
      lower: true
    }), {
      leftSide: true,
      lower: true,
      transposeA: true
    });
  }
}
function matrixPower(a, n) {
  if (!Number.isInteger(n)) throw new Error(`matrixPower: exponent must be an integer, got ${n}`);
  a = fudgeArray(a);
  const m = checkSquare("matrixPower", a.shape);
  if (n === 0) {
    const dtype = a.dtype;
    a.dispose();
    return broadcastTo(eye(m, void 0, { dtype }), a.shape);
  }
  if (n < 0) {
    a = inv(a);
    n = -n;
  }
  let result = null;
  let a2k = a;
  for (let k = 0; n; k++) {
    if (k > 0) a2k = matmul(a2k.ref, a2k);
    if (n % 2 === 1) result = result === null ? a2k.ref : matmul(result, a2k.ref);
    n = Math.floor(n / 2);
  }
  a2k.dispose();
  return result;
}
function multiDot(arrays) {
  if (arrays.length < 2) throw new Error("Expected at least two arrays");
  if (arrays[0].ndim === 1) {
    const [first, ...rest] = arrays;
    return multiDot([first.reshape([1, -1]), ...rest]).slice(0);
  }
  if (arrays[arrays.length - 1].ndim === 1) {
    const [rest, last] = [arrays.slice(0, -1), arrays[arrays.length - 1]];
    return multiDot([...rest, last.reshape([-1, 1])]).slice([], 0);
  }
  for (let i = 0; i < arrays.length; i++) if (arrays[i].ndim !== 2) throw new Error(`multiDot: expected 2D, got ${arrays[i].aval} at index ${i}`);
  const einsumArgs = [];
  for (let i = 0; i < arrays.length; i++) einsumArgs.push(arrays[i], [i, i + 1]);
  return einsum(...einsumArgs);
}
function slogdet(a) {
  a = fudgeArray(a);
  const n = checkSquare("slogdet", a.shape);
  const [lu$3, pivots, permutation] = lu(a);
  permutation.dispose();
  let parity = pivots.notEqual(arange(n)).astype(int32).sum(-1);
  const diag2 = lu$3.diagonal(0, -1, -2);
  parity = parity.add(diag2.ref.less(0).astype(int32).sum(-1)).mod(2);
  const logabsdet = log(absolute(diag2)).sum(-1);
  return [parity.mul(-2).add(1), logabsdet];
}
function solve(a, b) {
  a = fudgeArray(a);
  b = fudgeArray(b);
  const n = checkSquare("solve", a.shape);
  if (b.ndim === 0) throw new Error(`solve: b cannot be scalar`);
  const bIs1d = b.ndim === 1;
  if (bIs1d) b = b.reshape([...b.shape, 1]);
  if (b.shape[b.ndim - 2] !== n) throw new Error(`solve: leading dimension of b must match size of a, got a=${a.aval}, b=${b.aval}`);
  const m = b.shape[b.ndim - 1];
  const batchDims = generalBroadcast(a.shape.slice(0, -2), b.shape.slice(0, -2));
  a = broadcastTo(a, [
    ...batchDims,
    n,
    n
  ]);
  b = broadcastTo(b, [
    ...batchDims,
    n,
    m
  ]);
  const [lu$4, pivots, permutation] = lu(a);
  pivots.dispose();
  const P = arange(n).equal(permutation.reshape([...permutation.shape, 1])).astype(b.dtype);
  let x = triangularSolve(lu$4, triangularSolve(lu$4.ref, matmul(P, b), {
    leftSide: true,
    lower: true,
    unitDiagonal: true
  }), {
    leftSide: true,
    lower: false
  });
  if (bIs1d) x = squeeze(x, -1);
  return x;
}
function norm(x, { ord = null, axis = null, keepdims = false } = {}) {
  x = fudgeArray(x);
  if (!isFloatDtype(x.dtype)) x = x.astype(float32);
  const ndim2 = x.ndim;
  let axes;
  if (axis === null) {
    if (ord === null) return sqrt(sum(square(x), null, { keepdims }));
    axes = range(ndim2);
  } else if (typeof axis === "number") axes = [checkAxis(axis, ndim2)];
  else axes = axis.map((a) => checkAxis(a, ndim2));
  if (axes.length === 1) {
    if (typeof ord === "string") throw new Error(`norm: invalid order '${ord}' for vector norm`);
    return vectorNorm(x, {
      ord: ord ?? 2,
      axis: axes[0],
      keepdims
    });
  } else if (axes.length === 2) {
    let [rowAxis, colAxis] = axes;
    if (rowAxis === colAxis) throw new Error(`norm: duplicate axes ${JSON.stringify(axis)}`);
    if (ord === null || ord === "f" || ord === "fro") return sqrt(sum(square(x), axes, { keepdims }));
    else if (ord === 1 || ord === -1) {
      const sums = sum(absolute(x), rowAxis, { keepdims });
      if (!keepdims && colAxis > rowAxis) colAxis -= 1;
      return ord === 1 ? max(sums, colAxis, { keepdims }) : min(sums, colAxis, { keepdims });
    } else if (ord === Infinity || ord === -Infinity) {
      const sums = sum(absolute(x), colAxis, { keepdims });
      if (!keepdims && rowAxis > colAxis) rowAxis -= 1;
      return ord === Infinity ? max(sums, rowAxis, { keepdims }) : min(sums, rowAxis, { keepdims });
    } else if (ord === 2 || ord === -2 || ord === "nuc") {
      let y = moveaxis(x, rowAxis, -1);
      y = moveaxis(y, colAxis > rowAxis ? colAxis - 1 : colAxis, -1);
      const s = svdvals(y);
      const result = ord === 2 ? max(s, -1) : ord === -2 ? min(s, -1) : sum(s, -1);
      return keepdims ? expandDims(result, axes) : result;
    } else throw new Error(`norm: invalid order '${ord}' for matrix norm`);
  } else throw new Error(`norm: axis must be null, an integer, or a pair of integers, got ${JSON.stringify(axis)}`);
}
function vectorNorm(x, { ord = 2, axis = null, keepdims = false } = {}) {
  x = fudgeArray(x);
  const ax = axis ?? null;
  if (ord === Infinity) return max(absolute(x), ax, { keepdims });
  else if (ord === -Infinity) return min(absolute(x), ax, { keepdims });
  else if (ord === 0) return x.notEqual(0).astype(x.dtype).sum(ax, { keepdims });
  else return power(power(absolute(x), ord).sum(ax, { keepdims }), 1 / ord);
}
function matrixNorm(x, { ord = "fro", keepdims = false } = {}) {
  x = fudgeArray(x);
  if (x.ndim < 2) throw new Error(`Input must be at least 2-dimensional. Recieved ${x.ndim}-dimensional array`);
  if (ord === "fro") return sqrt(sum(square(x), [-2, -1], { keepdims }));
  else if (ord === "nuc") throw new Error("Order nuc is not supported");
  else if (ord === -1) return min(sum(absolute(x), [-2], { keepdims }), [-1], { keepdims });
  else if (ord === 1) return max(sum(absolute(x), [-2], { keepdims }), [-1], { keepdims });
  else if (ord === -2) throw new Error("Order -2 is not supported");
  else if (ord === 2) throw new Error("Order 2 is not supported");
  else if (ord === -Infinity) return min(sum(absolute(x), [-1], { keepdims }), [keepdims ? -2 : -1], { keepdims });
  else if (ord === Infinity) return max(sum(absolute(x), [-1], { keepdims }), [keepdims ? -2 : -1], { keepdims });
  else throw new Error(`Order ${ord} is not supported`);
}
function finfo(dtype) {
  if (!isFloatDtype(dtype)) throw new Error(`finfo: received ${dtype}, must be a floating-point type`);
  switch (dtype) {
    case "float16":
      return Object.freeze({
        bits: 16,
        dtype: "float16",
        eps: 2 ** -10,
        epsneg: 2 ** -11,
        machep: -10,
        max: 65504,
        maxexp: 16,
        min: -65504,
        minexp: -14,
        negep: -24,
        nexp: 5,
        nmant: 10,
        precision: 3,
        resolution: 1e-3,
        smallestNormal: 2 ** -14,
        smallestSubnormal: 2 ** -24
      });
    case "float32":
      return Object.freeze({
        bits: 32,
        dtype: "float32",
        eps: 2 ** -23,
        epsneg: 2 ** -24,
        machep: -23,
        max: 34028234663852886e22,
        maxexp: 128,
        min: -34028234663852886e22,
        minexp: -126,
        negep: -24,
        nexp: 8,
        nmant: 23,
        precision: 6,
        resolution: 1e-6,
        smallestNormal: 2 ** -126,
        smallestSubnormal: 2 ** -149
      });
    case "float64":
      return Object.freeze({
        bits: 64,
        dtype: "float64",
        eps: 2 ** -52,
        epsneg: 2 ** -53,
        machep: -52,
        max: Number.MAX_VALUE,
        maxexp: 1024,
        min: -Number.MAX_VALUE,
        minexp: -1022,
        negep: -53,
        nexp: 11,
        nmant: 52,
        precision: 15,
        resolution: 1e-15,
        smallestNormal: 2 ** -1022,
        smallestSubnormal: 2 ** -1074
      });
    default:
      throw new Error(`finfo: unsupported dtype ${dtype}`);
  }
}
function iinfo(dtype) {
  switch (dtype) {
    case "int32":
      return Object.freeze({
        bits: 32,
        dtype: "int32",
        max: 2147483647,
        min: -2147483648
      });
    case "uint32":
      return Object.freeze({
        bits: 32,
        dtype: "uint32",
        max: 4294967295,
        min: 0
      });
    default:
      throw new Error(`iinfo: unsupported dtype ${dtype}`);
  }
}
var numpy_exports = /* @__PURE__ */ __exportAll({
  Array: () => Array$1,
  DType: () => DType,
  abs: () => absolute,
  absolute: () => absolute,
  acos: () => acos,
  acosh: () => arccosh,
  add: () => add,
  all: () => all,
  allclose: () => allclose,
  any: () => any,
  append: () => append,
  applyAlongAxis: () => applyAlongAxis,
  applyOverAxes: () => applyOverAxes,
  arange: () => arange,
  arccos: () => acos,
  arccosh: () => arccosh,
  arcsin: () => asin,
  arcsinh: () => arcsinh,
  arctan: () => atan,
  arctan2: () => atan2,
  arctanh: () => arctanh,
  argmax: () => argmax,
  argmin: () => argmin,
  argsort: () => argsort,
  around: () => round,
  array: () => array,
  arrayEqual: () => arrayEqual,
  arrayEquiv: () => arrayEquiv,
  arraySplit: () => arraySplit,
  asin: () => asin,
  asinh: () => arcsinh,
  astype: () => astype,
  atan: () => atan,
  atan2: () => atan2,
  atanh: () => arctanh,
  average: () => average,
  bartlett: () => bartlett,
  bitwiseAnd: () => bitwiseAnd,
  bitwiseInvert: () => invert,
  bitwiseLeftShift: () => leftShift,
  bitwiseNot: () => invert,
  bitwiseOr: () => bitwiseOr,
  bitwiseRightShift: () => rightShift,
  bitwiseXor: () => bitwiseXor,
  blackman: () => blackman,
  bool: () => bool,
  broadcastArrays: () => broadcastArrays,
  broadcastShapes: () => broadcastShapes,
  broadcastTo: () => broadcastTo,
  cbrt: () => cbrt,
  ceil: () => ceil,
  clip: () => clip,
  columnStack: () => columnStack,
  concatenate: () => concatenate,
  convolve: () => convolve,
  copysign: () => copysign,
  corrcoef: () => corrcoef,
  correlate: () => correlate,
  cos: () => cos,
  cosh: () => cosh,
  countNonzero: () => countNonzero,
  cov: () => cov,
  cross: () => cross,
  cumprod: () => cumprod,
  cumsum: () => cumsum,
  cumulativeProd: () => cumulativeProd,
  cumulativeSum: () => cumulativeSum,
  deg2rad: () => deg2rad,
  degrees: () => degrees,
  diag: () => diag,
  diagIndices: () => diagIndices,
  diagIndicesFrom: () => diagIndicesFrom,
  diagflat: () => diagflat,
  diagonal: () => diagonal,
  diff: () => diff,
  divide: () => trueDivide,
  divmod: () => divmod,
  dot: () => dot$1,
  dsplit: () => dsplit,
  dstack: () => dstack,
  e: () => e,
  ediff1d: () => ediff1d,
  einsum: () => einsum,
  equal: () => equal,
  eulerGamma: () => eulerGamma,
  exp: () => exp,
  exp2: () => exp2,
  expandDims: () => expandDims,
  expm1: () => expm1,
  eye: () => eye,
  fft: () => numpy_fft_exports,
  finfo: () => finfo,
  flip: () => flip,
  fliplr: () => fliplr,
  flipud: () => flipud,
  float16: () => float16,
  float32: () => float32,
  float64: () => float64,
  floatPower: () => floatPower,
  floor: () => floor,
  floorDivide: () => floorDivide,
  fmax: () => fmax,
  fmin: () => fmin,
  fmod: () => fmod,
  frexp: () => frexp,
  fromfunction: () => fromfunction,
  full: () => full,
  fullLike: () => fullLike,
  geomspace: () => geomspace,
  greater: () => greater,
  greaterEqual: () => greaterEqual,
  hann: () => hann,
  heaviside: () => heaviside,
  hsplit: () => hsplit,
  hstack: () => hstack,
  hypot: () => hypot,
  identity: () => identity$1,
  iinfo: () => iinfo,
  indices: () => indices,
  inf: () => inf,
  inner: () => inner,
  int32: () => int32,
  invert: () => invert,
  isfinite: () => isfinite,
  isin: () => isin,
  isinf: () => isinf,
  isnan: () => isnan,
  isneginf: () => isneginf,
  isposinf: () => isposinf,
  isscalar: () => isscalar,
  ldexp: () => ldexp,
  leftShift: () => leftShift,
  less: () => less,
  lessEqual: () => lessEqual,
  linalg: () => numpy_linalg_exports,
  linspace: () => linspace,
  log: () => log,
  log10: () => log10,
  log1p: () => log1p,
  log2: () => log2,
  logaddexp: () => logaddexp,
  logaddexp2: () => logaddexp22,
  logicalAnd: () => logicalAnd,
  logicalNot: () => logicalNot,
  logicalOr: () => logicalOr,
  logicalXor: () => logicalXor,
  logspace: () => logspace,
  matmul: () => matmul,
  matrixTranspose: () => matrixTranspose,
  matvec: () => matvec,
  max: () => max,
  maximum: () => maximum,
  mean: () => mean,
  meshgrid: () => meshgrid,
  min: () => min,
  minimum: () => minimum,
  modf: () => modf,
  moveaxis: () => moveaxis,
  multiply: () => multiply,
  nan: () => nan,
  nanToNum: () => nanToNum,
  nanargmax: () => nanargmax,
  nanargmin: () => nanargmin,
  nancumprod: () => nancumprod,
  nancumsum: () => nancumsum,
  nanmax: () => nanmax,
  nanmean: () => nanmean,
  nanmin: () => nanmin,
  nanprod: () => nanprod,
  nanstd: () => nanstd,
  nansum: () => nansum,
  nanvar: () => nanvar,
  ndim: () => ndim,
  negative: () => negative,
  notEqual: () => notEqual,
  ones: () => ones,
  onesLike: () => onesLike,
  outer: () => outer,
  pad: () => pad,
  permuteDims: () => transpose,
  pi: () => pi,
  polyadd: () => polyadd,
  polyder: () => polyder,
  polymul: () => polymul,
  polysub: () => polysub,
  polyval: () => polyval,
  positive: () => positive,
  pow: () => power,
  power: () => power,
  prod: () => prod2,
  promoteTypes: () => promoteTypes,
  ptp: () => ptp,
  rad2deg: () => rad2deg,
  radians: () => radians,
  ravel: () => ravel,
  reciprocal: () => reciprocal,
  remainder: () => remainder,
  repeat: () => repeat,
  reshape: () => reshape,
  resultType: () => resultType,
  rightShift: () => rightShift,
  rint: () => rint,
  roll: () => roll,
  rot90: () => rot90,
  round: () => round,
  select: () => select,
  shape: () => shape,
  sign: () => sign,
  signbit: () => signbit,
  sin: () => sin,
  sinc: () => sinc,
  sinh: () => sinh,
  size: () => size,
  sort: () => sort,
  split: () => split$1,
  sqrt: () => sqrt,
  square: () => square,
  squeeze: () => squeeze,
  stack: () => stack,
  std: () => std,
  subtract: () => subtract,
  sum: () => sum,
  swapaxes: () => swapaxes,
  take: () => take,
  takeAlongAxis: () => takeAlongAxis,
  tan: () => tan,
  tanh: () => tanh,
  tensordot: () => tensordot,
  tile: () => tile,
  trace: () => trace,
  transpose: () => transpose,
  trapezoid: () => trapezoid,
  tri: () => tri,
  tril: () => tril,
  triu: () => triu,
  trueDivide: () => trueDivide,
  trunc: () => trunc,
  uint32: () => uint32,
  unstack: () => unstack,
  unwrap: () => unwrap,
  vander: () => vander,
  var_: () => var_,
  vdot: () => vdot,
  vecdot: () => vecdot,
  vecmat: () => vecmat,
  vsplit: () => vsplit,
  vstack: () => vstack,
  where: () => where,
  zeros: () => zeros,
  zerosLike: () => zerosLike
});
var JsArray$2 = globalThis.Array;
var float32 = "float32";
var int32 = "int32";
var uint32 = "uint32";
var bool = "bool";
var float16 = "float16";
var float64 = "float64";
var e = Math.E;
var eulerGamma = 0.5772156649015329;
var inf = Number.POSITIVE_INFINITY;
var nan = NaN;
var pi = Math.PI;
var add = add$1;
var multiply = mul;
var negative = neg;
var reciprocal = reciprocal$1;
var floor = floor$1;
var ceil = ceil$1;
var sin = sin$1;
var cos = cos$1;
var asin = asin$1;
var atan = atan$1;
var exp = exp$1;
var log = log$1;
var sqrt = sqrt$1;
var minimum = min$1;
var maximum = max$1;
var greater = greater$1;
var less = less$1;
var equal = equal$1;
var notEqual = notEqual$1;
var greaterEqual = greaterEqual$1;
var lessEqual = lessEqual$1;
function logicalAnd(x, y) {
  return astype(x, bool).mul(astype(y, bool));
}
function logicalOr(x, y) {
  return astype(x, bool).add(astype(y, bool));
}
function logicalXor(x, y) {
  return notEqual(astype(x, bool), astype(y, bool));
}
function logicalNot(x) {
  return notEqual(astype(x, bool), true);
}
function bitwiseAnd(x, y) {
  return bitCombine(x, y, "and");
}
function bitwiseOr(x, y) {
  return bitCombine(x, y, "or");
}
function bitwiseXor(x, y) {
  return bitCombine(x, y, "xor");
}
function invert(x) {
  const arr = fudgeArray(x);
  let allOnes;
  switch (arr.dtype) {
    case "bool":
      allOnes = true;
      break;
    case "uint32":
      allOnes = 4294967295;
      break;
    case "int32":
      allOnes = -1;
      break;
    default:
      throw new TypeError(`invert: unsupported dtype ${arr.dtype}`);
  }
  return bitCombine(arr, allOnes, "xor");
}
function leftShift(x, y) {
  return bitShift(x, y, "shl");
}
function rightShift(x, y) {
  return bitShift(x, y, "shr");
}
var where = where$1;
var transpose = transpose$1;
var reshape = reshape$1;
var moveaxis = moveaxis$1;
var pad = pad$1;
var ndim = ndim$1;
var shape = getShape;
var zerosLike = zerosLike$1;
var onesLike = onesLike$1;
var fullLike = fullLike$1;
function size(a, axis) {
  const s = shape(a);
  return axis === void 0 ? prod(s) : s[axis];
}
function isscalar(element) {
  if (element instanceof Tracer) return element.ndim === 0;
  return typeof element === "number" || typeof element === "boolean";
}
function astype(a, dtype) {
  return fudgeArray(a).astype(dtype);
}
function resultType(...args) {
  if (args.length === 0) throw new TypeError("resultType requires at least one argument");
  return args.map((x) => {
    if (typeof x === "string") {
      if (!Object.values(DType).includes(x)) throw new TypeError(`resultType: invalid dtype '${x}'`);
      return new ShapedArray([], x, false);
    }
    const { dtype, weakType } = getAval(x);
    return new ShapedArray([], dtype, weakType);
  }).reduce((a, b) => promoteAvals(a, b)).dtype;
}
function sum(a, axis = null, opts) {
  return reduce(a, "Add", axis, opts);
}
function nansum(a, axis = null, opts) {
  a = fudgeArray(a);
  if (isFloatDtype(a.dtype)) a = where(isnan(a.ref), 0, a);
  return sum(a, axis, opts);
}
function countNonzero(a, axis = null, opts) {
  return sum(astype(notEqual(a, 0), int32), axis, opts);
}
function prod2(a, axis = null, opts) {
  return reduce(a, "Mul", axis, opts);
}
function nanprod(a, axis = null, opts) {
  a = fudgeArray(a);
  if (isFloatDtype(a.dtype)) a = where(isnan(a.ref), 1, a);
  return prod2(a, axis, opts);
}
function min(a, axis = null, opts) {
  return reduce(a, "Min", axis, opts);
}
function max(a, axis = null, opts) {
  return reduce(a, "Max", axis, opts);
}
function nanmax(a, axis = null, opts) {
  a = fudgeArray(a);
  axis = normalizeAxis(axis, a.ndim);
  if (axis.some((i) => a.shape[i] === 0)) throw new Error("zero-size array to reduction operation max which has no identity");
  if (!isFloatDtype(a.dtype)) return max(a, axis, opts);
  const mask = isnan(a.ref);
  const out = max(where(mask.ref, -Infinity, a), axis, opts);
  return where(all(mask, axis, opts), NaN, out);
}
function any(a, axis = null, opts) {
  return fudgeArray(a).any(axis, opts);
}
function all(a, axis = null, opts) {
  return fudgeArray(a).all(axis, opts);
}
function ptp(a, axis = null, opts) {
  a = fudgeArray(a);
  return max(a.ref, axis, opts).sub(min(a, axis, opts));
}
function mean(a, axis = null, opts) {
  return fudgeArray(a).mean(axis, opts);
}
function nanmean(a, axis = null, opts) {
  a = fudgeArray(a);
  if (!isFloatDtype(a.dtype)) return mean(a, axis, opts);
  const normalizer = sum(astype(logicalNot(isnan(a.ref)), a.dtype), axis, opts);
  return nansum(a, axis, opts).div(normalizer);
}
function average(a, axis = null, opts) {
  a = fudgeArray(a);
  if (opts?.weights == null) return mean(a, axis, opts);
  const weights = fudgeArray(opts.weights);
  axis = normalizeAxis(axis, ndim(a));
  const wShape = weights.shape;
  const aShape = a.shape;
  if (deepEqual(wShape, aShape)) {
    const scl = sum(weights.ref, axis, opts);
    return sum(multiply(a, weights), axis, opts).div(scl);
  } else if (axis.length === 1 && wShape.length === 1 && wShape[0] === aShape[axis[0]]) {
    const wReshaped = reshape(weights, aShape.map((_, i) => i === axis[0] ? wShape[0] : 1));
    const scl = sum(wReshaped.ref, axis, opts);
    return sum(multiply(a, wReshaped), axis, opts).div(scl);
  } else {
    weights.dispose();
    a.dispose();
    throw new Error(`average: weights shape ${JSON.stringify(wShape)} is not compatible with array shape ${JSON.stringify(aShape)} and axis ${JSON.stringify(axis)}`);
  }
}
function argmin(a, axis, opts) {
  a = fudgeArray(a);
  let flattenedNdim;
  if (axis === void 0) {
    flattenedNdim = a.ndim;
    a = a.ravel();
    axis = 0;
  } else axis = checkAxis(axis, a.ndim);
  const shape2 = a.shape;
  if (shape2[axis] === 0) throw new Error("attempt to get argmin of an empty sequence");
  const isMax = equal(a, min(a.ref, axis, { keepdims: true }));
  const length = array(shape2[axis], {
    dtype: int32,
    device: a.device
  });
  const idx = isMax.astype("int32").mul(arange(shape2[axis], 0, -1, {
    dtype: int32,
    device: a.device
  }).reshape([shape2[axis], ...rep(shape2.length - axis - 1, 1)]));
  const result = length.sub(max(idx, axis, opts));
  return flattenedNdim !== void 0 && opts?.keepdims ? result.reshape(rep(flattenedNdim, 1)) : result;
}
function nanargmin(a, axis, opts) {
  a = fudgeArray(a);
  if (!isFloatDtype(a.dtype)) return argmin(a, axis, opts);
  const nanMask = isnan(a.ref);
  const result = argmin(where(nanMask.ref, Infinity, a), axis, opts);
  return where(all(nanMask, axis ?? null, opts), -1, result);
}
function argmax(a, axis, opts) {
  a = fudgeArray(a);
  let flattenedNdim;
  if (axis === void 0) {
    flattenedNdim = a.ndim;
    a = a.ravel();
    axis = 0;
  } else axis = checkAxis(axis, a.ndim);
  const shape2 = a.shape;
  if (shape2[axis] === 0) throw new Error("attempt to get argmax of an empty sequence");
  const isMax = equal(a, max(a.ref, axis, { keepdims: true }));
  const length = array(shape2[axis], {
    dtype: int32,
    device: a.device
  });
  const idx = isMax.astype("int32").mul(arange(shape2[axis], 0, -1, {
    dtype: int32,
    device: a.device
  }).reshape([shape2[axis], ...rep(shape2.length - axis - 1, 1)]));
  const result = length.sub(max(idx, axis, opts));
  return flattenedNdim !== void 0 && opts?.keepdims ? result.reshape(rep(flattenedNdim, 1)) : result;
}
function nanargmax(a, axis, opts) {
  a = fudgeArray(a);
  if (!isFloatDtype(a.dtype)) return argmax(a, axis, opts);
  const nanMask = isnan(a.ref);
  const result = argmax(where(nanMask.ref, -Infinity, a), axis, opts);
  return where(all(nanMask, axis ?? null, opts), -1, result);
}
function cumulativeQuadratic(op, a, padValue, k = 0) {
  const n = a.shape[a.ndim - 1];
  a = broadcast(a, a.shape.concat(n), [-2]);
  return reduce(where(tri(n, n, k, { dtype: bool }), a, padValue), op, -1);
}
function cumulativeHelper(op, a, axis) {
  const padValue = op === "Add" ? 0 : 1;
  a = fudgeArray(a);
  if (a.ndim === 0) a = a.reshape([1]);
  axis = checkAxis(axis, a.ndim);
  if (a.size === 0) return a;
  a = moveaxis(a, axis, -1);
  const n = a.shape[a.ndim - 1];
  const split2 = 256;
  if (n <= split2 * 2) return moveaxis(cumulativeQuadratic(op, a, padValue), -1, axis);
  const batch = a.shape.slice(0, -1);
  const cols = batch.map(() => []);
  const m = Math.ceil(n / split2);
  a = pad$1(a, { [a.ndim - 1]: [0, m * split2 - n] });
  const scans = cumulativeQuadratic(op, a.reshape([
    ...batch,
    m,
    split2
  ]), padValue);
  const offsets = cumulativeQuadratic(op, scans.ref.slice(...cols, [], -1), padValue, -1);
  return moveaxis((op === "Add" ? scans.add(offsets.reshape([
    ...batch,
    m,
    1
  ])) : scans.mul(offsets.reshape([
    ...batch,
    m,
    1
  ]))).reshape([...batch, m * split2]).slice(...cols, [0, n]), -1, axis);
}
function cumsum(a, axis) {
  if (axis === void 0) a = ravel(a), axis = 0;
  return cumulativeHelper("Add", a, axis);
}
function cumulativeSum(x, opts) {
  x = fudgeArray(x);
  if (x.ndim === 0) x = x.reshape([1]);
  if (x.ndim > 1 && opts?.axis == void 0) throw new Error("cumulativeSum: axis is required for arrays of ndim > 1");
  const axis = checkAxis(opts?.axis ?? 0, x.ndim);
  if (opts?.includeInitial) x = concatenate([zerosLike(x.ref, { shape: x.shape.toSpliced(axis, 1, 1) }), x], axis);
  return cumulativeHelper("Add", x, axis);
}
function cumprod(a, axis) {
  if (axis === void 0) a = ravel(a), axis = 0;
  return cumulativeHelper("Mul", a, axis);
}
function cumulativeProd(x, opts) {
  x = fudgeArray(x);
  if (x.ndim === 0) x = x.reshape([1]);
  if (x.ndim > 1 && opts?.axis == void 0) throw new Error("cumulativeProd: axis is required for arrays of ndim > 1");
  const axis = checkAxis(opts?.axis ?? 0, x.ndim);
  if (opts?.includeInitial) x = concatenate([onesLike(x.ref, { shape: x.shape.toSpliced(axis, 1, 1) }), x], axis);
  return cumulativeHelper("Mul", x, axis);
}
function nancumprod(a, axis) {
  a = fudgeArray(a);
  if (isFloatDtype(a.dtype)) a = where(isnan(a.ref), 1, a);
  return cumprod(a, axis);
}
function nancumsum(a, axis) {
  a = fudgeArray(a);
  if (isFloatDtype(a.dtype)) a = where(isnan(a.ref), 0, a);
  return cumsum(a, axis);
}
function trapezoid(y, x = null, opts) {
  y = fudgeArray(y);
  const requestedAxis = opts?.axis ?? -1;
  const axis = checkAxis(requestedAxis, y.ndim);
  const sliceAxis2 = (a, ax, s) => a.slice(...rep(ax, []), s);
  let dx;
  if (x === null) {
    if (!isFloatDtype(y.dtype)) y = y.astype(float32);
    dx = fudgeArray(opts?.dx ?? 1);
    if (dx.ndim > 0) {
      if (dx.ndim < y.ndim) dx = dx.reshape(rep(y.ndim - dx.ndim, 1).concat(dx.shape));
      dx = moveaxis(dx, requestedAxis, -1);
    }
  } else {
    x = fudgeArray(x);
    let dtype = promoteTypes(y.dtype, x.dtype);
    if (!isFloatDtype(dtype)) dtype = float32;
    if (y.dtype !== dtype) y = y.astype(dtype);
    if (x.dtype !== dtype) x = x.astype(dtype);
    const xAxis = x.ndim === 1 ? 0 : checkAxis(requestedAxis, x.ndim);
    const diff2 = sliceAxis2(x.ref, xAxis, [1]).sub(sliceAxis2(x, xAxis, [0, -1]));
    dx = x.ndim === 1 ? diff2 : moveaxis(diff2, xAxis, -1);
  }
  y = moveaxis(y, axis, -1);
  return sliceAxis2(y.ref, y.ndim - 1, [1]).add(sliceAxis2(y, y.ndim - 1, [0, -1])).mul(0.5).mul(dx).sum(-1);
}
function diff(a, n = 1, axis = -1, opts) {
  a = fudgeArray(a);
  if (!Number.isInteger(n) || n < 0) {
    a.dispose();
    throw new Error(`diff: order must be a non-negative integer, got ${n}`);
  }
  if (n === 0) return a;
  if (a.ndim === 0) {
    a.dispose();
    throw new Error("diff: input must be at least one-dimensional");
  }
  axis = checkAxis(axis, a.ndim);
  if (opts?.prepend !== void 0 || opts?.append !== void 0) {
    const edgeShape = a.shape.toSpliced(axis, 1, 1);
    const combined = [a];
    if (opts.prepend !== void 0) {
      let prepend = fudgeArray(opts.prepend);
      if (prepend.ndim === 0) prepend = broadcastTo(prepend, edgeShape);
      combined.unshift(prepend);
    }
    if (opts.append !== void 0) {
      let append2 = fudgeArray(opts.append);
      if (append2.ndim === 0) append2 = broadcastTo(append2, edgeShape);
      combined.push(append2);
    }
    a = concatenate(combined, axis);
  }
  const op = a.dtype === bool ? notEqual : subtract;
  const upper = a.shape.map((_, i) => i === axis ? [1] : []);
  const lower = a.shape.map((_, i) => i === axis ? [0, -1] : []);
  for (let i = Math.min(n, a.shape[axis]); i > 0; i--) a = op(a.ref.slice(...upper), a.slice(...lower));
  return a;
}
function ediff1d(ary, opts) {
  const arr = ravel(ary);
  const n = arr.size;
  const dtype = arr.dtype;
  const parts = [arr.ref.slice([Math.min(n, 1)]).sub(arr.slice([0, Math.max(n - 1, 0)]))];
  if (opts?.toBegin !== void 0) parts.unshift(ravel(opts.toBegin).astype(dtype));
  if (opts?.toEnd !== void 0) parts.push(ravel(opts.toEnd).astype(dtype));
  return parts.length === 1 ? parts[0] : concatenate(parts, 0);
}
function flip(x, axis = null) {
  const nd = ndim(x);
  axis = normalizeAxis(axis, nd);
  return flip$1(x, axis);
}
function split$1(a, indicesOrSections, axis = 0) {
  a = fudgeArray(a);
  axis = checkAxis(axis, a.ndim);
  const size2 = a.shape[axis];
  let sizes;
  if (typeof indicesOrSections === "number") {
    if (size2 % indicesOrSections !== 0) throw new Error(`Array of size ${size2} cannot be split into ${indicesOrSections} equal parts`);
    sizes = rep(indicesOrSections, size2 / indicesOrSections);
  } else {
    const bounds = [
      0,
      ...indicesOrSections.map((i) => i < 0 ? i + size2 : i),
      size2
    ];
    sizes = range(bounds.length - 1).map((i) => bounds[i + 1] - bounds[i]);
  }
  return splitBySizes(a, sizes, axis);
}
function arraySplit(a, indicesOrSections, axis = 0) {
  a = fudgeArray(a);
  axis = checkAxis(axis, a.ndim);
  if (typeof indicesOrSections !== "number") return split$1(a, indicesOrSections, axis);
  const size2 = a.shape[axis];
  const partSize = Math.floor(size2 / indicesOrSections);
  const remainder3 = size2 % indicesOrSections;
  const sizes = [...rep(remainder3, partSize + 1), ...rep(indicesOrSections - remainder3, partSize)];
  return splitBySizes(a, sizes, axis);
}
function dsplit(a, indicesOrSections) {
  a = fudgeArray(a);
  if (a.ndim < 3) throw new Error("dsplit only works on arrays of 3 or more dimensions");
  return split$1(a, indicesOrSections, 2);
}
function hsplit(a, indicesOrSections) {
  a = fudgeArray(a);
  if (a.ndim === 0) throw new Error("hsplit only works on arrays of 1 or more dimensions");
  return split$1(a, indicesOrSections, a.ndim > 1 ? 1 : 0);
}
function vsplit(a, indicesOrSections) {
  return split$1(a, indicesOrSections, 0);
}
function splitBySizes(a, sizes, axis) {
  const results = [];
  for (let i = 0; i < sizes.length; i += 7) if (i === sizes.length) {
    results.push(a);
    break;
  } else if (i + 8 >= sizes.length) {
    results.push(...split$2(a, axis, sizes.slice(i)));
    break;
  } else {
    const groupSizes = [...sizes.slice(i, i + 7), sizes.slice(i + 7).reduce((x, y) => x + y, 0)];
    const outs = split$2(a, axis, groupSizes);
    results.push(...outs.slice(0, -1));
    a = outs[outs.length - 1];
  }
  return results;
}
function concatenate(xs, axis = 0) {
  if (xs.length === 0) throw new Error("Need at least one array to concatenate");
  const shapes = xs.map(shape);
  axis = checkAxis(axis, shapes[0].length);
  for (let i = 1; i < shapes.length; i++) if (shapes[i].length !== shapes[0].length || !shapes[i].every((d, j) => j === axis || d === shapes[0][j])) throw new Error(`Cannot concatenate arrays ${xs[0].aval} and ${xs[i].aval} along axis ${axis}`);
  let result = xs[0];
  for (let i = 1; i < xs.length; i += 7) {
    const group = xs.slice(i, i + 7);
    result = concatenate$1([result, ...group], axis);
  }
  return result;
}
function stack(xs, axis = 0) {
  if (xs.length === 0) throw new Error("Need at least one array to stack");
  const shapes = xs.map((x) => shape(x));
  if (!shapes.every((s) => deepEqual(s, shapes[0]))) throw new Error(`Cannot stack arrays with different shapes: ${JSON.stringify(shapes)}`);
  axis = checkAxis(axis, shapes[0].length + 1);
  const newShape = shapes[0].toSpliced(axis, 0, 1);
  return concatenate(xs.map((x) => fudgeArray(x).reshape(newShape)), axis);
}
function unstack(x, axis = 0) {
  x = fudgeArray(x);
  if (x.ndim === 0) throw new Error("unstack requires arrays with rank > 0");
  axis = checkAxis(axis, x.ndim);
  const size2 = x.shape[axis];
  if (size2 === 0) {
    x.dispose();
    return [];
  }
  return split$1(x, size2, axis).map((part) => squeeze(part, axis));
}
function hstack(xs) {
  if (xs.length === 0) throw new Error("Need at least one array to hstack");
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  if (nds[0] === 0) return stack(xs);
  else if (nds[0] === 1) return concatenate(xs);
  else return concatenate(xs, 1);
}
function vstack(xs) {
  if (xs.length === 0) throw new Error("Need at least one array to vstack");
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  if (nds[0] === 0) return stack(xs).reshape([-1, 1]);
  else if (nds[0] === 1) return stack(xs);
  else return concatenate(xs);
}
function dstack(xs) {
  if (xs.length === 0) throw new Error("Need at least one array to dstack");
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  if (nds[0] === 0) return stack(xs).reshape([
    1,
    1,
    -1
  ]);
  else if (nds[0] === 1) {
    const ret = stack(xs, -1);
    return ret.reshape([1, ...ret.shape]);
  } else if (nds[0] === 2) return stack(xs, -1);
  else return concatenate(xs, 2);
}
function columnStack(xs) {
  if (xs.length === 0) throw new Error("Need at least one array to columnStack");
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  if (nds[0] === 0) return stack(xs).reshape([1, -1]);
  else if (nds[0] === 1) return stack(xs, -1);
  else return concatenate(xs, 1);
}
function flipud(x) {
  return flip(x, 0);
}
function fliplr(x) {
  return flip(x, 1);
}
function rot90(m, k = 1, axes = [0, 1]) {
  if (!Number.isInteger(k)) throw new Error(`rot90: k must be an integer`);
  if (!JsArray$2.isArray(axes) || axes.length !== 2) throw new Error(`rot90: axes must contain exactly two axes`);
  const a = fudgeArray(m);
  const [axis1, axis2] = normalizeAxis(axes, a.ndim, false);
  switch ((k % 4 + 4) % 4) {
    case 0:
      return a;
    case 1:
      return swapaxes(flip(a, axis2), axis1, axis2);
    case 2:
      return flip(a, [axis1, axis2]);
    case 3:
      return flip(swapaxes(a, axis1, axis2), axis2);
    default:
      throw new Error(`rot90: unreachable ${k}`);
  }
}
function roll(a, shift, axis = null) {
  a = fudgeArray(a);
  if (axis === null) return roll(a.ravel(), shift, 0).reshape(a.shape);
  axis = normalizeAxis(axis, a.ndim, false);
  if (typeof shift === "number") shift = rep(axis.length, shift);
  else if (shift.length !== axis.length) throw new Error(`roll: shift and axis must have the same length, got shift=${JSON.stringify(shift)} and axis=${JSON.stringify(axis)}`);
  for (const [s, ax] of zip(shift, axis)) {
    if (!Number.isInteger(s)) throw new Error(`roll: shift must be an integer, got ${s}`);
    const n = a.shape[ax];
    const s1 = (s % n + n) % n;
    if (s1 !== 0) {
      const parts = split$1(a, [n - s1], ax);
      a = concatenate([parts[1], parts[0]], ax);
    }
  }
  return a;
}
function swapaxes(a, axis1, axis2) {
  a = fudgeArray(a);
  axis1 = checkAxis(axis1, a.ndim);
  axis2 = checkAxis(axis2, a.ndim);
  if (axis1 === axis2) return a;
  const perm = range(a.ndim);
  perm[axis1] = axis2;
  perm[axis2] = axis1;
  return transpose(a, perm);
}
function matrixTranspose(a) {
  if (ndim(a) < 2) throw new Error(`matrixTranspose: input array must be at least 2D`);
  return moveaxis(a, -1, -2);
}
function ravel(a) {
  return fudgeArray(a).ravel();
}
function squeeze(a, axis = null) {
  const as = shape(a);
  if (axis === null) axis = range(as.length).filter((i) => as[i] === 1);
  else if (typeof axis === "number") axis = [axis];
  axis = axis.map((a2) => checkAxis(a2, as.length));
  for (const a2 of axis) if (as[a2] !== 1) throw new Error("Cannot squeeze axis with size != 1");
  return reshape(a, as.filter((_, i) => !axis.includes(i)));
}
function expandDims(a, axis) {
  const as = shape(a);
  axis = typeof axis === "number" ? [axis] : axis;
  axis = normalizeAxis(axis, as.length + axis.length);
  const newShape = [];
  let srcIdx = 0;
  for (let i = 0; i < as.length + axis.length; i++) if (axis.includes(i)) newShape.push(1);
  else newShape.push(as[srcIdx++]);
  return reshape(a, newShape);
}
function repeat(a, repeats, axis) {
  if (!Number.isInteger(repeats) || repeats < 0) throw new Error(`repeat: repeats must be a non-negative integer, got ${repeats}`);
  a = fudgeArray(a);
  if (axis === void 0) {
    a = ravel(a);
    axis = 0;
  }
  axis = checkAxis(axis, a.ndim);
  if (repeats === 1) return a;
  const broadcastedShape = a.shape.toSpliced(axis + 1, 0, repeats);
  const finalShape = a.shape.toSpliced(axis, 1, a.shape[axis] * repeats);
  return broadcast(a, broadcastedShape, [axis + 1]).reshape(finalShape);
}
function tile(a, reps) {
  a = fudgeArray(a);
  if (typeof reps === "number") reps = [reps];
  if (!reps.every((r) => Number.isInteger(r) && r >= 0)) throw new Error(`tile: reps must be non-negative integers, got ${JSON.stringify(reps)}`);
  const ndiff = reps.length - a.ndim;
  if (ndiff > 0) a = a.reshape([...rep(ndiff, 1), ...a.shape]);
  if (ndiff < 0) reps = [...rep(-ndiff, 1), ...reps];
  const broadcastedShape = [];
  const broadcastAxes = [];
  for (let i = 0; i < a.ndim; i++) {
    if (reps[i] > 1) {
      broadcastedShape.push(reps[i]);
      broadcastAxes.push(broadcastedShape.length - 1);
    }
    broadcastedShape.push(a.shape[i]);
  }
  const finalShape = a.shape.map((d, i) => reps[i] * d);
  return broadcast(a, broadcastedShape, broadcastAxes).reshape(finalShape);
}
function broadcastTo(a, shape2) {
  const nd = ndim(a);
  if (shape2.length < nd) throw new Error(`broadcastTo: target shape ${JSON.stringify(shape2)} has fewer dimensions than input array: ${nd}`);
  return broadcast(a, shape2, range(shape2.length - nd));
}
function broadcastShapes(...shapes) {
  if (shapes.length === 0) return [];
  return shapes.reduce(generalBroadcast);
}
function broadcastArrays(...arrays) {
  const outShape = broadcastShapes(...arrays.map((a) => shape(a)));
  return arrays.map((a) => broadcastTo(a, outShape));
}
function diagonal(a, offset, axis1, axis2) {
  return fudgeArray(a).diagonal(offset, axis1, axis2);
}
function diag(v, k = 0) {
  const a = fudgeArray(v);
  if (!Number.isInteger(k)) throw new Error(`k must be an integer, got ${k}`);
  if (a.ndim === 1) {
    const n = a.shape[0];
    const ret = where(eye(n).equal(1), a.ref, zerosLike(a));
    if (k > 0) return pad(ret, [[0, k], [k, 0]]);
    else if (k < 0) return pad(ret, [[-k, 0], [0, -k]]);
    else return ret;
  } else if (a.ndim === 2) return diagonal(a, k);
  else throw new Error("numpy.diag only supports 1D and 2D arrays");
}
function diagflat(v, k = 0) {
  return diag(ravel(v), k);
}
function diagIndices(n, ndim2 = 2) {
  if (!Number.isInteger(n) || n < 0) throw new Error(`n must be a nonnegative integer, got ${n}`);
  if (!Number.isInteger(ndim2) || ndim2 < 0) throw new Error(`ndim must be a nonnegative integer, got ${ndim2}`);
  if (ndim2 === 0) return [];
  const index = arange(n);
  return [index, ...range(ndim2 - 1).map(() => index.ref)];
}
function diagIndicesFrom(arr) {
  const a = fudgeArray(arr);
  const nd = a.ndim;
  const aShape = a.shape;
  a.dispose();
  if (nd < 2) throw new Error(`diagIndicesFrom: input array must be at least 2D, got ${nd}D`);
  if (!aShape.every((s) => s === aShape[0])) throw new Error(`diagIndicesFrom: all dimensions of input must be equal, got shape ${JSON.stringify(aShape)}`);
  return diagIndices(aShape[0], nd);
}
function trace(a, offset = 0, axis1 = 0, axis2 = 1) {
  return diagonal(a, offset, axis1, axis2).sum(-1);
}
function sort(a, axis = -1) {
  return fudgeArray(a).sort(axis);
}
function argsort(a, axis = -1) {
  return fudgeArray(a).argsort(axis);
}
function append(arr, values, axis = null) {
  if (axis === null) return concatenate([ravel(arr), ravel(values)], 0);
  const a = fudgeArray(arr);
  axis = checkAxis(axis, a.ndim);
  return concatenate([a, fudgeArray(values)], axis);
}
function take(a, indices2, axis = null) {
  if (axis === null) {
    a = ravel(a);
    axis = 0;
  }
  axis = checkAxis(axis, ndim(a));
  return gather(a, [indices2], [axis], axis);
}
function takeAlongAxis(a, indices2, axis = -1) {
  a = fudgeArray(a);
  indices2 = fudgeArray(indices2);
  if (a.ndim !== indices2.ndim) {
    const aShape = a.shape;
    const indexShape = indices2.shape;
    a.dispose();
    indices2.dispose();
    throw new Error(`takeAlongAxis: input and indices must have the same rank, got ${JSON.stringify(aShape)} and ${JSON.stringify(indexShape)}`);
  }
  axis = checkAxis(axis, a.ndim);
  const outShape = indices2.shape.slice();
  for (let i = 0; i < a.ndim; i++) {
    if (i === axis) continue;
    const aDim = a.shape[i];
    const indexDim = indices2.shape[i];
    if (aDim !== indexDim && aDim !== 1 && indexDim !== 1) {
      const aShape = a.shape;
      const indexShape = indices2.shape;
      a.dispose();
      indices2.dispose();
      throw new Error(`takeAlongAxis: non-axis dimensions must broadcast, got ${JSON.stringify(aShape)} and ${JSON.stringify(indexShape)}`);
    }
    outShape[i] = Math.max(aDim, indexDim);
  }
  const coords = [];
  for (let i = 0; i < a.ndim; i++) if (i === axis) coords.push(broadcastTo(indices2, outShape));
  else {
    const shape2 = rep(a.ndim, 1);
    shape2[i] = a.shape[i];
    coords.push(broadcastTo(arange(0, a.shape[i], 1, {
      dtype: int32,
      device: a.device
    }).reshape(shape2), outShape));
  }
  return gather(a, coords, range(a.ndim), 0);
}
function select(condlist, choicelist, defaultValue = 0) {
  if (condlist.length !== choicelist.length) throw new Error(`select: condlist must have length equal to choicelist (${condlist.length} vs ${choicelist.length})`);
  if (condlist.length === 0) throw new Error("select: condlist must be non-empty");
  let output = fudgeArray(defaultValue);
  for (let i = condlist.length - 1; i >= 0; i--) output = where(astype(condlist[i], bool), choicelist[i], output);
  return output;
}
function applyAlongAxis(func1d, axis, x) {
  const nd = ndim(x);
  axis = checkAxis(axis, nd);
  let func = func1d;
  for (let i = 1; i < nd - axis; i++) func = vmap$1(func, i);
  for (let i = 0; i < axis; i++) func = vmap$1(func, 0);
  const y = func(fudgeArray(x));
  const post = nd - axis - 1;
  return y.transpose([
    ...range(axis),
    ...range(axis + post, y.ndim),
    ...range(axis, axis + post).reverse()
  ]);
}
function applyOverAxes(func, a, axes) {
  a = fudgeArray(a);
  for (const axis of axes) {
    const b = func(a, axis);
    if (b.ndim === a.ndim) a = b;
    else if (b.ndim === a.ndim - 1) a = expandDims(b, axis);
    else throw new Error("function must return array of the correct shape");
  }
  return a;
}
function allclose(actual, expected, options) {
  const { rtol = 1e-5, atol = 1e-7, equalNaN = false } = options ?? {};
  const x = array(actual);
  const y = array(expected);
  if (!deepEqual(x.shape, y.shape)) return false;
  const xData = x.dataSync();
  const yData = y.dataSync();
  for (let i = 0; i < xData.length; i++) {
    if (equalNaN ? isNaN(xData[i]) !== isNaN(yData[i]) : isNaN(xData[i]) || isNaN(yData[i])) return false;
    if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i])) return false;
  }
  return true;
}
function arrayEqual(a1, a2, opts) {
  a1 = fudgeArray(a1);
  a2 = fudgeArray(a2);
  if (!deepEqual(a1.shape, a2.shape)) {
    a1.dispose();
    a2.dispose();
    return array(false);
  }
  if (opts?.equalNaN) return where(isnan(a1.ref).mul(isnan(a2.ref)), true, equal(a1, a2)).all();
  return equal(a1, a2).all();
}
function arrayEquiv(a1, a2) {
  a1 = fudgeArray(a1);
  a2 = fudgeArray(a2);
  try {
    const [b1, b2] = broadcastArrays(a1, a2);
    return equal(b1, b2).all();
  } catch {
    a1.dispose();
    a2.dispose();
    return array(false);
  }
}
function matmul(x, y) {
  if (ndim(x) === 0 || ndim(y) === 0) throw new Error("matmul: x and y must be at least 1D");
  x = x, y = y;
  if (y.ndim === 1) return dot$2(x, y);
  const numBatchDims = Math.min(Math.max(x.ndim, 2), y.ndim) - 2;
  return dot(x, y, {
    lhsContractingDims: [-1],
    rhsContractingDims: [-2],
    lhsBatchDims: range(-2 - numBatchDims, -2),
    rhsBatchDims: range(-2 - numBatchDims, -2)
  });
}
function matvec(x1, x2) {
  if (ndim(x1) < 2 || ndim(x2) < 1) throw new Error("matvec: x1 must be at least 2D and x2 at least 1D");
  return einsum("...mn,...n->...m", x1, x2);
}
function vecmat(x1, x2) {
  if (ndim(x1) < 1 || ndim(x2) < 2) throw new Error("vecmat: x1 must be at least 1D and x2 at least 2D");
  return einsum("...n,...nm->...m", x1, x2);
}
function dot$1(x, y) {
  if (ndim(x) === 0 || ndim(y) === 0) return multiply(x, y);
  x = x, y = y;
  if (y.ndim === 1) return dot$2(x, y);
  return dot(x, y, {
    lhsContractingDims: [-1],
    rhsContractingDims: [-2]
  });
}
function tensordot(x, y, axes = 2) {
  x = fudgeArray(x);
  y = fudgeArray(y);
  if (typeof axes === "number") axes = [range(-axes, 0), range(axes)];
  return dot(x, y, {
    lhsContractingDims: axes[0],
    rhsContractingDims: axes[1]
  });
}
function einsum(...args) {
  if (args.length === 0) throw new Error("einsum: must provide at least one argument");
  let input;
  let operands = [];
  if (typeof args[0] === "string") {
    operands = args.slice(1).map(fudgeArray);
    input = parseEinsumExpression(args[0], operands.map((x) => x.shape));
  } else {
    const n = args.length >> 1;
    const shapes = [];
    const lhsIndices = [];
    for (let i = 0; i < n; i++) {
      operands.push(fudgeArray(args[2 * i]));
      shapes.push(operands[i].shape);
      lhsIndices.push(args[2 * i + 1]);
    }
    let rhsIndex;
    if (args.length % 2 === 1) rhsIndex = args[2 * n];
    else {
      const indexCount = [];
      for (const i of lhsIndices.flat()) indexCount[i] = (indexCount[i] ?? 0) + 1;
      rhsIndex = [...indexCount.entries()].filter(([_, count]) => count === 1).map(([i, _]) => i);
    }
    input = {
      lhsIndices,
      rhsIndex,
      shapes
    };
  }
  const path = computeEinsumPath(input);
  if (DEBUG >= 3) console.info(`einsum: computed path: ${path.approximateFlops} flops`);
  const indexUsageCounts = [];
  for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
  const indices2 = [...input.lhsIndices];
  const processSingleTensor = (ar2, index2, doNotReduce = []) => {
    index2 = index2.slice();
    diag: while (true) {
      for (let i = 0; i < index2.length; i++) {
        const idx = index2[i];
        const j = index2.indexOf(idx, i + 1);
        if (j !== -1) {
          ar2 = diagonal(ar2, 0, i, j);
          index2.splice(j, 1);
          index2.splice(i, 1);
          index2.push(idx);
          continue diag;
        }
      }
      break;
    }
    for (let i = index2.length - 1; i >= 0; i--) {
      const idx = index2[i];
      if (indexUsageCounts[idx] === 0 && !doNotReduce.includes(idx)) {
        ar2 = sum(ar2, i);
        index2.splice(i, 1);
      }
    }
    return [ar2, index2];
  };
  for (const [i, j] of path.path) {
    let indexReduced = [];
    const indexGroup = [];
    for (const idx of [...indices2[i], ...indices2[j]]) {
      if (!indexGroup.includes(idx)) indexGroup.push(idx);
      if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
    }
    const [a, aidx] = processSingleTensor(operands[i], indices2[i], indices2[j]);
    const [b, bidx] = processSingleTensor(operands[j], indices2[j], indices2[i]);
    indexReduced = indexReduced.filter((idx) => aidx.includes(idx));
    const indexBatch = aidx.filter((idx) => bidx.includes(idx) && !indexReduced.includes(idx));
    const result = dot(a, b, {
      lhsContractingDims: indexReduced.map((idx) => aidx.indexOf(idx)),
      rhsContractingDims: indexReduced.map((idx) => bidx.indexOf(idx)),
      lhsBatchDims: indexBatch.map((idx) => aidx.indexOf(idx)),
      rhsBatchDims: indexBatch.map((idx) => bidx.indexOf(idx))
    });
    operands.push(result);
    indices2.push([
      ...indexBatch,
      ...aidx.filter((idx) => !bidx.includes(idx)),
      ...bidx.filter((idx) => !aidx.includes(idx))
    ]);
    for (const idx of indices2[indices2.length - 1]) ++indexUsageCounts[idx];
  }
  for (const idx of indices2[operands.length - 1]) --indexUsageCounts[idx];
  const [ar, index] = processSingleTensor(operands[operands.length - 1], indices2[operands.length - 1]);
  const finalPerm = input.rhsIndex.map((idx) => index.indexOf(idx));
  return ar.transpose(finalPerm);
}
function inner(x, y) {
  return dot(fudgeArray(x), fudgeArray(y), {
    lhsContractingDims: [-1],
    rhsContractingDims: [-1]
  });
}
function outer(x, y) {
  x = ravel(x);
  y = ravel(y);
  return multiply(x.reshape([x.shape[0], 1]), y);
}
function vander(x, { n, increasing = false } = {}) {
  x = fudgeArray(x);
  if (x.ndim !== 1) {
    const ndim2 = x.ndim;
    x.dispose();
    throw new Error(`vander: input must be 1D, got ${ndim2}D`);
  }
  n ??= x.shape[0];
  if (!Number.isInteger(n) || n < 0) {
    x.dispose();
    throw new Error(`vander: n must be a non-negative integer, got ${n}`);
  }
  const rows = x.shape[0];
  if (n <= 1) return onesLike(x, { shape: [rows, n] });
  const result = cumulativeProd(broadcastTo(x.reshape([rows, 1]), [rows, n - 1]), {
    axis: 1,
    includeInitial: true
  });
  return increasing ? result : flip(result, 1);
}
function polyval(p, x) {
  p = fudgeArray(p);
  x = fudgeArray(x);
  if (p.ndim === 0) {
    const ndim2 = p.ndim;
    p.dispose();
    x.dispose();
    throw new Error(`polyval: coefficients must have at least one dimension, got ${ndim2}D`);
  }
  const n = p.shape[0];
  const powers = vander(ravel(x), { n }).reshape([...x.shape, n]);
  return vecdot(moveaxis(p, 0, -1), powers);
}
function polyadd(a1, a2) {
  a1 = fudgeArray(a1);
  a2 = fudgeArray(a2);
  if (a1.ndim === 0 || a2.ndim === 0) {
    const [ndim1, ndim2] = [a1.ndim, a2.ndim];
    a1.dispose();
    a2.dispose();
    throw new Error(`polyadd: both inputs must be at least 1D arrays, got ${ndim1}D and ${ndim2}D`);
  }
  let base = a1;
  let update = a2;
  if (a2.shape[0] > a1.shape[0]) [base, update] = [a2, a1];
  const updateShape = [update.shape[0], ...base.shape.slice(1)];
  try {
    update = broadcastTo(update, updateShape);
  } catch (e2) {
    base.dispose();
    update.dispose();
    throw e2;
  }
  const leading = base.shape[0] - update.shape[0];
  if (leading > 0) update = pad(update, { 0: [leading, 0] });
  return add(base, update);
}
function polysub(a1, a2) {
  a1 = fudgeArray(a1);
  a2 = fudgeArray(a2);
  if (a1.ndim === 0 || a2.ndim === 0) {
    const [ndim1, ndim2] = [a1.ndim, a2.ndim];
    a1.dispose();
    a2.dispose();
    throw new Error(`polysub: both inputs must be at least 1D arrays, got ${ndim1}D and ${ndim2}D`);
  }
  const a2IsLonger = a2.shape[0] > a1.shape[0];
  let base = a1;
  let update = a2;
  if (a2IsLonger) [base, update] = [a2, a1];
  const updateShape = [update.shape[0], ...base.shape.slice(1)];
  try {
    update = broadcastTo(update, updateShape);
  } catch (e2) {
    base.dispose();
    update.dispose();
    throw e2;
  }
  const leading = base.shape[0] - update.shape[0];
  if (leading > 0) update = pad(update, { 0: [leading, 0] });
  return a2IsLonger ? subtract(update, base) : subtract(base, update);
}
function polyder(p, m = 1) {
  p = fudgeArray(p);
  if (!Number.isInteger(m) || m < 0) {
    p.dispose();
    throw new Error(`polyder: order of derivative must be a non-negative integer, got ${m}`);
  }
  if (!isFloatDtype(p.dtype)) p = astype(p, float32);
  if (m === 0) return p;
  if (p.ndim === 0) {
    const ndim2 = p.ndim;
    p.dispose();
    throw new Error(`polyder: coefficients must have at least one dimension, got ${ndim2}D`);
  }
  const n = p.shape[0];
  const length = Math.max(n - m, 0);
  const coeff = [];
  for (let j = 0; j < length; j++) {
    let c = 1;
    for (let i = 1; i <= m; i++) c *= n - j - i;
    coeff.push(c);
  }
  const batchBroadcastDims = rep(p.ndim - 1, 1);
  const scale = array(coeff, {
    dtype: p.dtype,
    device: p.device
  }).reshape([length, ...batchBroadcastDims]);
  return multiply(p.slice([0, length]), scale);
}
function polymul(a1, a2) {
  a1 = fudgeArray(a1);
  a2 = fudgeArray(a2);
  if (a1.ndim !== 1 || a2.ndim !== 1) {
    const [ndim1, ndim2] = [a1.ndim, a2.ndim];
    a1.dispose();
    a2.dispose();
    throw new Error(`polymul: both inputs must be 1D arrays, got ${ndim1}D and ${ndim2}D`);
  }
  const promotedDtype = resultType(a1, a2);
  const dtype = isFloatDtype(promotedDtype) ? promotedDtype : "float32";
  a1 = a1.astype(dtype);
  a2 = a2.astype(dtype);
  if (a1.shape[0] === 0) {
    a1.dispose();
    a1 = zeros([1], { dtype });
  }
  if (a2.shape[0] === 0) {
    a2.dispose();
    a2 = zeros([1], { dtype });
  }
  return convolve(a1, a2, "full");
}
var cross = jit$1(function cross2(a, b, { axisa = -1, axisb = -1, axisc = -1, axis } = {}) {
  if (axis !== void 0) {
    axisa = axis;
    axisb = axis;
    axisc = axis;
  }
  axisa = checkAxis(axisa, ndim(a));
  axisb = checkAxis(axisb, ndim(b));
  a = moveaxis(a, axisa, -1);
  b = moveaxis(b, axisb, -1);
  const da = a.shape.at(-1);
  const db = b.shape.at(-1);
  if (da !== 2 && da !== 3 || db !== 2 && db !== 3) throw new Error(`cross: incompatible dimensions for cross product (got ${da} and ${db})`);
  if (da === 2 && db === 2) {
    const [a02, a12] = split$1(a, 2, -1);
    const [b02, b12] = split$1(b, 2, -1);
    return squeeze(a02.mul(b12).sub(a12.mul(b02)), -1);
  }
  if (da === 2) {
    const zeroShape = [...a.shape.slice(0, -1), 1];
    a = concatenate([a, zeros(zeroShape)], -1);
  }
  if (db === 2) {
    const zeroShape = [...b.shape.slice(0, -1), 1];
    b = concatenate([b, zeros(zeroShape)], -1);
  }
  const [a0, a1, a2] = split$1(a, 3, -1);
  const [b0, b1, b2] = split$1(b, 3, -1);
  return moveaxis(concatenate([
    a1.ref.mul(b2.ref).sub(a2.ref.mul(b1.ref)),
    a2.mul(b0.ref).sub(a0.ref.mul(b2)),
    a0.mul(b1).sub(a1.mul(b0))
  ], -1), -1, axisc);
}, { staticArgnums: [2] });
function vecdot(x, y, { axis } = {}) {
  const xaxis = checkAxis(axis ?? -1, ndim(x));
  const yaxis = checkAxis(axis ?? -1, ndim(y));
  if (shape(x)[xaxis] !== shape(y)[yaxis]) throw new Error(`vecdot: shapes ${JSON.stringify(shape(x))} and ${JSON.stringify(shape(y))} not aligned along axis ${axis}: ${shape(x)[xaxis]} != ${shape(y)[yaxis]}`);
  x = moveaxis(x, xaxis, -1);
  y = moveaxis(y, yaxis, -1);
  return dot$2(x, y);
}
function vdot(x, y) {
  return dot$2(ravel(x), ravel(y));
}
function _convImpl(name, x, y, mode) {
  if (x.ndim !== 1 || y.ndim !== 1) throw new Error(`${name}: both inputs must be 1D arrays, got ${x.ndim}D and ${y.ndim}D`);
  let flipOutput = false;
  if (x.shape[0] < y.shape[0]) {
    [x, y] = [y, x];
    if (name === "correlate") flipOutput = true;
  }
  if (name === "convolve") y = flip(y);
  let padding;
  if (mode === "valid") padding = "VALID";
  else if (mode === "same") padding = "SAME_LOWER";
  else if (mode === "full") padding = [[y.shape[0] - 1, y.shape[0] - 1]];
  else throw new Error(`${name}: invalid mode ${mode}, expected "full", "same", or "valid"`);
  const z = conv(x.slice(null, null), y.slice(null, null), [1], padding).slice(0, 0);
  return flipOutput ? flip(z) : z;
}
function convolve(x, y, mode = "full") {
  return _convImpl("convolve", x, y, mode);
}
function correlate(x, y, mode = "valid") {
  return _convImpl("correlate", x, y, mode);
}
function meshgrid(xs, { indexing } = {}) {
  indexing ??= "xy";
  for (const x of xs) if (x.ndim !== 1) throw new Error(`meshgrid: all inputs must be 1D arrays, got ${x.ndim}D array`);
  if (xs.length <= 1) return xs;
  if (indexing === "xy") {
    const [a, b, ...rest] = xs;
    const [rb, ra, ...rrest] = meshgrid([
      b,
      a,
      ...rest
    ], { indexing: "ij" });
    return [
      ra,
      rb,
      ...rrest
    ];
  }
  const shape2 = xs.map((x) => x.shape[0]);
  return xs.map((x, i) => broadcast(x, shape2, [...range(i), ...range(i + 1, xs.length)]));
}
function indices(dimensions, { dtype, device, sparse } = {}) {
  dtype = dtype ?? int32;
  if (dimensions.some((d) => !Number.isInteger(d) || d < 0)) throw new Error(`indices: dimensions must be non-negative integers, got ${JSON.stringify(dimensions)}`);
  const output = dimensions.map((dim, i) => {
    const shape2 = rep(dimensions.length, 1);
    shape2[i] = dim;
    const x = arange(0, dim, 1, {
      dtype,
      device
    }).reshape(shape2);
    return sparse ? x : broadcastTo(x, dimensions);
  });
  if (sparse) return output;
  return output.length > 0 ? stack(output) : zeros([0], {
    dtype,
    device
  });
}
function fromfunction(func, shape2, { dtype, device } = {}) {
  dtype = dtype ?? float32;
  if (shape2.some((d) => !Number.isInteger(d) || d < 0)) throw new Error(`fromfunction: shape must be non-negative integers, got ${JSON.stringify(shape2)}`);
  let f = func;
  for (let loopIndex = 0; loopIndex < shape2.length; loopIndex++) {
    const mappedIndex = shape2.length - 1 - loopIndex;
    const inputAxes = shape2.map((_, dimensionIndex) => dimensionIndex === mappedIndex ? 0 : null);
    f = vmap$1(f, inputAxes);
  }
  return map(fudgeArray, f(...shape2.map((s) => arange(0, s, 1, {
    dtype,
    device
  }))));
}
function clip(a, min2, max2) {
  a = fudgeArray(a);
  if (max2 !== void 0) a = minimum(a, max2);
  if (min2 !== void 0) a = maximum(a, min2);
  return a;
}
function absolute(x) {
  x = fudgeArray(x);
  return where(less(x.ref, 0), x.ref.mul(-1), x);
}
function sign(x) {
  x = fudgeArray(x);
  return where(notEqual(x.ref, 0), where(less(x, 0), -1, 1), 0);
}
function signbit(x) {
  const arr = fudgeArray(x);
  switch (arr.dtype) {
    case "bool":
    case "uint32":
      return zerosLike(arr, { dtype: "bool" });
    case "int32":
      return less(arr, 0);
    default:
      return less(arr.astype("float32").view("int32"), 0);
  }
}
var copysign = jit$1(function copysign2(x, y) {
  return absolute(x).mul(sign(y));
});
var fmax = jit$1(function fmax2(x1, x2) {
  return where(greater(x1.ref, x2.ref).add(isnan(x2.ref)), x1, x2);
});
var fmin = jit$1(function fmin2(x1, x2) {
  return where(logicalOr(less(x1.ref, x2.ref), isnan(x2.ref)), x1, x2);
});
var positive = fudgeArray;
function bartlett(M) {
  if (M < 0 || !Number.isInteger(M)) throw new RangeError(`Invalid window size for bartlett: ${M}. Must be a non-negative integer.`);
  if (M <= 1) return ones([M]);
  return subtract(1, absolute(linspace(-1, 1, M)));
}
function hann(M) {
  return cos(linspace(0, 2 * Math.PI, M)).mul(-0.5).add(0.5);
}
function blackman(M) {
  if (M < 0 || !Number.isInteger(M)) throw new RangeError(`Invalid window size for blackman: ${M}. Must be a non-negative integer.`);
  if (M <= 1) return ones([M]);
  return cos(linspace(0, 2 * Math.PI, M)).mul(-0.5).add(cos(linspace(0, 4 * Math.PI, M)).mul(0.08)).add(0.42);
}
var heaviside = jit$1(function heaviside2(x1, x2) {
  return where(less(x1.ref, 0), 0, where(equal(x1, 0), x2, 1));
});
function square(x) {
  x = fudgeArray(x);
  return x.ref.mul(x);
}
function tan(x) {
  x = fudgeArray(x);
  return sin(x.ref).div(cos(x));
}
var sinc = jit$1(function sinc2(x) {
  const pix = x.ref.mul(Math.PI);
  return where(equal(x, 0), 1, sin(pix.ref).div(pix));
});
function acos(x) {
  return subtract(pi / 2, asin(x));
}
var hypot = jit$1(function hypot2(x1, x2) {
  return sqrt(square(x1).add(square(x2)));
});
var atan2 = jit$1(function atan22(y, x) {
  const r = sqrt(square(x.ref).add(square(y.ref)));
  const xNeg = less(x.ref, 0);
  const numer = where(xNeg.ref, r.ref.sub(x.ref), y.ref);
  const denom = where(xNeg, y, r.add(x));
  return atan(numer.div(denom)).mul(2);
});
function subtract(x, y) {
  x = fudgeArray(x);
  y = fudgeArray(y);
  return x.sub(y);
}
function trueDivide(x, y) {
  x = fudgeArray(x);
  y = fudgeArray(y);
  if (!isFloatDtype(x.dtype) && !isFloatDtype(y.dtype)) {
    x = x.astype("float32");
    y = y.astype("float32");
  }
  return x.div(y);
}
function floorDivide(x, y) {
  x = fudgeArray(x);
  y = fudgeArray(y);
  if (isFloatDtype(x.dtype) || isFloatDtype(y.dtype)) return floor(trueDivide(x, y));
  return subtract(x, remainder(x.ref, y.ref)).div(y);
}
var fmod = jit$1(function fmod2(x, y) {
  return x.ref.sub(y.ref.mul(idiv(x, y)));
});
var remainder = jit$1(function remainder2(x, y) {
  return mod(mod(x, y.ref).add(y.ref), y);
});
function divmod(x, y) {
  const xArr = fudgeArray(x);
  const yArr = fudgeArray(y);
  return [floorDivide(xArr.ref, yArr.ref), remainder(xArr, yArr)];
}
function trunc(x) {
  return idiv(x, 1);
}
function modf(x) {
  x = fudgeArray(x);
  if (!isFloatDtype(x.dtype)) x = x.astype("float32");
  const whole = trunc(x.ref);
  return [x.sub(whole.ref), whole];
}
var round = jit$1(function round2(a, decimals = 0) {
  if (decimals === 0) return rint(a);
  const factor = 10 ** decimals;
  return rint(a.mul(factor)).mul(1 / factor);
}, { staticArgnums: [1] });
var rint = jit$1(function rint2(x) {
  const rounded = floor(x.ref.add(0.5));
  const half = x.ref.sub(floor(x)).equal(0.5);
  const odd = remainder(rounded.ref, 2).notEqual(0);
  return where(half.mul(odd), rounded.ref.sub(1), rounded);
});
function ldexp(x1, x2) {
  return multiply(x1, exp2(x2));
}
function frexp(x) {
  x = fudgeArray(x);
  const absx = absolute(x.ref);
  const exponent = where(equal(x.ref, 0), 0, floor(log2(absx)).add(1).astype("int32"));
  return [x.div(exp2(exponent.ref.astype(x.dtype))), exponent];
}
function exp2(p) {
  return exp(multiply(p, Math.LN2));
}
function log2(x) {
  return log(x).mul(Math.LOG2E);
}
function log10(x) {
  return log(x).mul(Math.LOG10E);
}
function expm1(x) {
  x = fudgeArray(x);
  if (!isFloatDtype(x.dtype)) x = x.astype("float32");
  const cutoff = x.dtype === "float64" ? 0.01 : 0.1;
  const useSeries = absolute(x.ref).lessEqual(cutoff);
  const seriesX = clip(x.ref, -cutoff, cutoff);
  return where(useSeries, seriesX.ref.mul(1 / 5040).add(1 / 720).mul(seriesX.ref).add(1 / 120).mul(seriesX.ref).add(1 / 24).mul(seriesX.ref).add(1 / 6).mul(seriesX.ref).add(1 / 2).mul(seriesX.ref).add(1).mul(seriesX), exp(x).sub(1));
}
function log1p(x) {
  x = fudgeArray(x);
  if (!isFloatDtype(x.dtype)) x = x.astype("float32");
  const cutoff = x.dtype === "float64" ? 0.01 : 0.2;
  const useSeries = absolute(x.ref).lessEqual(cutoff);
  const seriesX = clip(x.ref, -cutoff, cutoff);
  return where(useSeries, seriesX.ref.mul(-1 / 10).add(1 / 9).mul(seriesX.ref).sub(1 / 8).mul(seriesX.ref).add(1 / 7).mul(seriesX.ref).sub(1 / 6).mul(seriesX.ref).add(1 / 5).mul(seriesX.ref).sub(1 / 4).mul(seriesX.ref).add(1 / 3).mul(seriesX.ref).sub(1 / 2).mul(seriesX.ref).add(1).mul(seriesX), log(x.add(1)));
}
var logaddexp = jit$1(function logaddexp2(x1, x2) {
  const xmax = maximum(x1.ref, x2.ref);
  return log(exp(x1.sub(xmax.ref)).add(exp(x2.sub(xmax.ref)))).add(xmax);
});
var logaddexp22 = jit$1(function logaddexp23(x1, x2) {
  const xmax = maximum(x1.ref, x2.ref);
  return log2(exp2(x1.sub(xmax.ref)).add(exp2(x2.sub(xmax.ref)))).add(xmax);
});
function deg2rad(x) {
  return multiply(x, pi / 180);
}
var radians = deg2rad;
function rad2deg(x) {
  return multiply(x, 180 / pi);
}
var degrees = rad2deg;
function unwrap(p, discont = null, axis = -1, period = 2 * pi) {
  let x = fudgeArray(p);
  axis = checkAxis(axis, x.ndim);
  if (!isFloatDtype(x.dtype)) x = astype(x, float32);
  if (x.shape[axis] <= 1) return x;
  const toDtype = x.dtype === "float64" ? (v) => v : x.dtype === "float16" ? Math.f16round : Math.fround;
  const T = toDtype(period);
  const interval = toDtype(T / 2);
  const disc = toDtype(discont ?? T / 2);
  const skip = rep(axis, []);
  const [head, tail] = split$1(x.ref, [1], axis);
  const dd = tail.ref.sub(x.slice(...skip, [0, -1]));
  const r = mod(dd.ref.add(interval), T);
  let ddmod = (T >= 0 ? where(less(r.ref, 0), r.ref.add(T), r) : where(greater(r.ref, 0), r.ref.add(T), r)).sub(interval);
  ddmod = where(logicalAnd(equal(ddmod.ref, -interval), greater(dd.ref, 0)), interval, ddmod);
  const phaseCorrection = where(less(absolute(dd.ref), disc), 0, ddmod.sub(dd));
  return concatenate([head, tail.add(cumsum(phaseCorrection, axis))], axis);
}
var power = jit$1(function power2(x1, x2) {
  const x2i = trunc(x2.ref);
  const shouldBeNaN = multiply(x2.ref.notEqual(x2i.ref), x1.ref.less(0));
  const resultSign = where(mod(x2i, 2).notEqual(0), where(x1.ref.less(0), -1, 1), 1);
  return where(shouldBeNaN, nan, exp(log(absolute(x1)).mul(x2)).mul(resultSign));
});
function floatPower(x1, x2) {
  x1 = fudgeArray(x1);
  x2 = fudgeArray(x2);
  const promotedDtype = promoteAvals(x1.aval, x2.aval).dtype;
  const dtype = isFloatDtype(promotedDtype) ? promotedDtype : "float32";
  x1 = x1.astype(dtype);
  x2 = x2.astype(dtype);
  return power(x1, x2);
}
var cbrt = jit$1(function cbrt2(x) {
  const sgn = where(less(x.ref, 0), -1, 1);
  return sgn.ref.mul(exp(log(x.mul(sgn)).mul(1 / 3)));
});
var sinh = jit$1(function sinh2(x) {
  const ex = exp(x);
  const emx = reciprocal(ex.ref);
  return ex.sub(emx).mul(0.5);
});
var cosh = jit$1(function cosh2(x) {
  const ex = exp(x);
  const emx = reciprocal(ex.ref);
  return ex.add(emx).mul(0.5);
});
var tanh = jit$1(function tanh2(x) {
  const negsgn = where(less(x.ref, 0), 1, -1);
  const en2x = exp(x.mul(negsgn.ref).mul(2));
  return en2x.ref.sub(1).div(en2x.add(1)).mul(negsgn);
});
var arcsinh = jit$1(function arcsinh2(x) {
  return log(x.ref.add(sqrt(square(x).add(1))));
});
var arccosh = jit$1(function arccosh2(x) {
  return log(x.ref.add(sqrt(square(x).sub(1))));
});
var arctanh = jit$1(function arctanh2(x) {
  return log(add(1, x.ref).div(subtract(1, x))).mul(0.5);
});
function var_(x, axis = null, opts) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
  if (n === 0) throw new Error("var: cannot compute variance over zero-length axis");
  const mu = opts?.mean !== void 0 ? opts.mean : mean(x.ref, axis, { keepdims: true });
  return square(x.sub(mu)).sum(axis, { keepdims: opts?.keepdims }).mul(1 / (n - (opts?.correction ?? 0)));
}
function nanvar(x, axis = null, opts) {
  x = fudgeArray(x);
  if (!isFloatDtype(x.dtype)) x = x.astype(float32);
  const nanMask = isnan(x.ref);
  const count = sum(astype(logicalNot(nanMask.ref), int32), axis, { keepdims: opts?.keepdims });
  const mu = nanmean(x.ref, axis, { keepdims: true });
  const sq = sum(square(where(nanMask, 0, x.sub(mu))), axis, { keepdims: opts?.keepdims });
  const divisor = count.sub(opts?.correction ?? 0);
  const invalid = divisor.ref.lessEqual(0);
  const numerator = where(invalid.ref, NaN, sq);
  return numerator.div(astype(where(invalid, 1, divisor), numerator.dtype));
}
function std(x, axis = null, opts) {
  return sqrt(var_(x, axis, opts));
}
function nanstd(x, axis = null, opts) {
  return sqrt(nanvar(x, axis, opts));
}
function cov(x, y = null, { rowvar = true } = {}) {
  x = fudgeArray(x);
  if (x.ndim === 1) x = x.reshape([1, x.shape[0]]);
  if (y !== null) {
    y = fudgeArray(y);
    if (y.ndim === 1) y = y.reshape([1, y.shape[0]]);
    x = vstack([x, y]);
  }
  if (!rowvar) x = x.transpose();
  const [_M, N] = x.shape;
  x = x.ref.sub(x.mean(1, { keepdims: true }));
  return dot$1(x.ref, x.transpose()).div(N - 1);
}
function corrcoef(x, y) {
  const c = cov(x, y);
  const variances = diag(c.ref);
  const norm2 = sqrt(outer(variances.ref, variances));
  return c.div(norm2);
}
function isin(element, testElements, opts) {
  element = fudgeArray(element);
  const outShape = element.shape;
  if (prod(outShape) === 0) {
    fudgeArray(testElements).dispose();
    return astype(element, bool);
  }
  const result = any(equal(expandDims(ravel(element), 1), ravel(testElements)), 1);
  return reshape(opts?.invert ? logicalNot(result) : result, outShape);
}
function isinf(x) {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.ref.equal(Infinity).add(x.equal(-Infinity)) : fullLike(x, false);
}
function isnan(x) {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.ref.notEqual(x) : fullLike(x, false);
}
function isneginf(x) {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.equal(-Infinity) : fullLike(x, false);
}
function isposinf(x) {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.equal(Infinity) : fullLike(x, false);
}
function nanToNum(x, { nan: nan2 = 0, posinf = null, neginf = null } = {}) {
  x = fudgeArray(x);
  x = where(isnan(x.ref), nan2, x);
  posinf ??= isFloatDtype(x.dtype) ? finfo(x.dtype).max : iinfo(x.dtype).max;
  neginf ??= isFloatDtype(x.dtype) ? finfo(x.dtype).min : iinfo(x.dtype).min;
  x = where(isposinf(x.ref), posinf, x);
  x = where(isneginf(x.ref), neginf, x);
  return x;
}
function nanmin(a, axis = null, opts) {
  a = fudgeArray(a);
  if (!isFloatDtype(a.dtype)) return min(a, axis, opts);
  const nanMask = isnan(a.ref);
  const result = min(where(nanMask.ref, Infinity, a), axis, opts);
  return where(all(nanMask, axis, opts), NaN, result);
}
var isfinite = jit$1(function isfinite2(x) {
  if (!isFloatDtype(x.dtype)) return fullLike(x, true);
  return isnan(x.ref).add(isinf(x)).notEqual(true);
});
function cholesky(a, { upper = false } = {}) {
  const L = cholesky$2(a);
  return upper ? moveaxis(L, -2, -1) : L;
}
function eigh(x, { lower = true, sortEigenvalues = true, symmetrizeInput = true } = {}) {
  x = fudgeArray(x);
  const n = checkSquare("eigh", x.shape);
  if (!isFloatDtype(x.dtype) || x.dtype === "float16") x = x.astype(float32);
  if (symmetrizeInput) x = x.ref.add(matrixTranspose(x)).mul(0.5);
  else if (!lower) x = matrixTranspose(x);
  const batchShape = x.shape.slice(0, -2);
  let v;
  [x, v] = jacobiEigh(x, {
    maxSweeps: Math.max(8, 2 * n),
    tolerance: x.dtype === "float64" ? 1e-12 : 1e-6
  });
  const valuesUnsorted = diagonal(x, 0, -2, -1);
  if (!sortEigenvalues) return [v, valuesUnsorted];
  const order = argsort(valuesUnsorted.ref);
  const values = takeAlongAxis(valuesUnsorted, order.ref, -1);
  return [takeAlongAxis(v, order.reshape([
    ...batchShape,
    1,
    n
  ]), -1), values];
}
function svd(a, { computeUv = true, fullMatrices = false } = {}) {
  a = fudgeArray(a);
  if (a.ndim < 2) throw new Error(`svd: input must be at least 2D, got ${a}`);
  if (!isFloatDtype(a.dtype) || a.dtype === "float16") a = a.astype(float32);
  const [m, n] = a.shape.slice(-2);
  if (fullMatrices && m !== n) throw new Error("svd: fullMatrices=true is only supported for square input");
  const batchShape = a.shape.slice(0, -2);
  const k = Math.min(m, n);
  const singularValues = (values) => sqrt(maximum(flip(values, -1), 0));
  const invSingularValues = (s) => where(greater(s.ref, 0), reciprocal(s.ref), 0);
  if (m >= n) {
    const [v, values] = eigh(matmul(matrixTranspose(a.ref), a.ref), { symmetrizeInput: false });
    const s = singularValues(values);
    if (!computeUv) {
      v.dispose();
      return s;
    }
    const vDesc = flip(v, -1);
    const invS = invSingularValues(s.ref);
    return [
      matmul(a, vDesc.ref).mul(invS.reshape([
        ...batchShape,
        1,
        k
      ])),
      s,
      matrixTranspose(vDesc)
    ];
  } else {
    const [u, values] = eigh(matmul(a.ref, matrixTranspose(a.ref)), { symmetrizeInput: false });
    const s = singularValues(values);
    if (!computeUv) {
      u.dispose();
      return s;
    }
    const uDesc = flip(u, -1);
    const invS = invSingularValues(s.ref);
    return [
      uDesc,
      s,
      matmul(matrixTranspose(uDesc.ref), a).mul(invS.reshape([
        ...batchShape,
        k,
        1
      ]))
    ];
  }
}
function lu(x) {
  return lu$1(x);
}
function triangularSolve(a, b, { leftSide = false, lower = false, transposeA = false, unitDiagonal = false } = {}) {
  a = fudgeArray(a);
  b = fudgeArray(b);
  if (!leftSide) transposeA = !transposeA;
  else b = moveaxis(b, -2, -1);
  if (transposeA) {
    a = moveaxis(a, -2, -1);
    lower = !lower;
  }
  let x = triangularSolve$1(a, b, {
    lower,
    unitDiagonal
  });
  if (leftSide) x = moveaxis(x, -2, -1);
  return x;
}
var JsArray$1 = globalThis.Array;
function dot(lhs, rhs, { lhsContractingDims: lc = [], rhsContractingDims: rc = [], lhsBatchDims: lb = [], rhsBatchDims: rb = [] } = {}) {
  if (lc.length !== rc.length) throw new Error(`dot: contracting dims lengths mismatch, got ${JSON.stringify(lc)} and ${JSON.stringify(rc)}`);
  else if (lb.length !== rb.length) throw new Error(`dot: batch dims lengths mismatch, got ${JSON.stringify(lb)} and ${JSON.stringify(rb)}`);
  lc = lc.map((a) => checkAxis(a, lhs.ndim));
  rc = rc.map((a) => checkAxis(a, rhs.ndim));
  lb = lb.map((a) => checkAxis(a, lhs.ndim));
  rb = rb.map((a) => checkAxis(a, rhs.ndim));
  if (lc.some((a) => lb.includes(a))) throw new Error(`dot: lhs contracting dims ${JSON.stringify(lc)} overlap with batch dims ${JSON.stringify(lb)}`);
  else if (rc.some((a) => rb.includes(a))) throw new Error(`dot: rhs contracting dims ${JSON.stringify(rc)} overlap with batch dims ${JSON.stringify(rb)}`);
  const lf = range(lhs.ndim).filter((a) => !lc.includes(a) && !lb.includes(a));
  const rf = range(rhs.ndim).filter((a) => !rc.includes(a) && !rb.includes(a));
  const lhs2 = lhs.transpose([
    ...lb,
    ...lf,
    ...lc
  ]);
  const rhs2 = rhs.transpose([
    ...rb,
    ...rf,
    ...rc
  ]);
  if (lc.length === 0) return mul(lhs2.reshape([
    ...lb.map((a) => lhs.shape[a]),
    ...lf.map((a) => lhs.shape[a]),
    ...rep(rf.length, 1)
  ]), rhs2.reshape([
    ...rb.map((a) => rhs.shape[a]),
    ...rep(lf.length, 1),
    ...rf.map((a) => rhs.shape[a])
  ]));
  const dotShapeX = lc.map((a) => lhs.shape[a]);
  const dotShapeY = rc.map((a) => rhs.shape[a]);
  if (!deepEqual(dotShapeX, dotShapeY)) throw new Error(`dot: shapes not aligned along contracting dims: ${JSON.stringify(dotShapeX)} != ${JSON.stringify(dotShapeY)}`);
  return dot$2(lhs2.reshape([
    ...lb.map((a) => lhs.shape[a]),
    ...lf.map((a) => lhs.shape[a]),
    ...rep(rf.length, 1),
    prod(dotShapeX)
  ]), rhs2.reshape([
    ...rb.map((a) => rhs.shape[a]),
    ...rep(lf.length, 1),
    ...rf.map((a) => rhs.shape[a]),
    prod(dotShapeY)
  ]));
}
function padtypeToPads(inShape, filterShape, strides, dilation, padding) {
  const padType = padding.toUpperCase();
  switch (padType) {
    case "VALID":
      return rep(inShape.length, [0, 0]);
    case "SAME":
    case "SAME_LOWER": {
      const padSizes = zipn(inShape.map((size2, i) => Math.ceil(size2 / strides[i])), strides, filterShape, dilation, inShape).map(([o, s, k, d, i]) => Math.max(0, (o - 1) * s + 1 + (k - 1) * d - i));
      if (padType === "SAME") return padSizes.map((size2) => [size2 >> 1, size2 - (size2 >> 1)]);
      else return padSizes.map((size2) => [size2 - (size2 >> 1), size2 >> 1]);
    }
    default:
      throw new Error(`Unknown padding type: ${padType}`);
  }
}
function lowerConvTranspose1d(lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation) {
  if (lhs.ndim !== 3 || rhs.ndim !== 3 || windowStrides.length !== 1 || windowStrides[0] !== 1 || lhsDilation.length !== 1 || lhsDilation[0] <= 1 || rhsDilation.length !== 1 || rhsDilation[0] !== 1 || padding.length !== 1 || padding[0][0] < 0 || padding[0][1] < 0 || lhs.shape.some((size2) => size2 <= 0) || rhs.shape.some((size2) => size2 <= 0)) return null;
  const outputShape = checkConvShape(lhs.shape, rhs.shape, {
    vmapDims: 0,
    strides: windowStrides,
    padding,
    lhsDilation,
    rhsDilation
  });
  if (outputShape.some((size2) => size2 <= 0)) return null;
  const [batchSize, inputChannels, inputLength] = lhs.shape;
  const [outputChannels, , kernelSize] = rhs.shape;
  const stride = lhsDilation[0];
  const contributionCount = Math.ceil(kernelSize / stride);
  if (contributionCount > 8) return null;
  lhs = lhs.transpose([
    0,
    2,
    1
  ]).reshape([
    batchSize,
    1,
    1,
    inputLength,
    inputChannels
  ]);
  rhs = rhs.transpose([
    0,
    2,
    1
  ]).reshape([
    1,
    outputChannels,
    kernelSize,
    1,
    inputChannels
  ]);
  let columns = dot$2(lhs, rhs);
  columns = flip$1(columns.transpose([
    0,
    1,
    3,
    2
  ]), [3]);
  columns = pad$1(columns, { 3: [0, contributionCount * stride - kernelSize] });
  columns = columns.reshape([
    batchSize,
    outputChannels,
    inputLength,
    contributionCount,
    stride
  ]);
  let output = split$2(columns, 3, rep(contributionCount, 1)).map((term, index) => {
    term = term.reshape([
      batchSize,
      outputChannels,
      inputLength,
      stride
    ]);
    return pad$1(term, { 2: [index, contributionCount - 1 - index] });
  }).reduce((lhs2, rhs2) => lhs2.add(rhs2)).reshape([
    batchSize,
    outputChannels,
    -1
  ]);
  const fullLength = (inputLength - 1) * stride + kernelSize;
  output = output.slice([], [], [0, fullLength]);
  const outputLength = outputShape[2];
  const offset = kernelSize - 1 - padding[0][0];
  const padLeft = Math.max(0, -offset);
  const padRight = Math.max(0, offset + outputLength - fullLength);
  output = pad$1(output, { 2: [padLeft, padRight] });
  return output.slice([], [], [offset + padLeft, offset + padLeft + outputLength]);
}
function convGeneralDilated(lhs, rhs, windowStrides, padding, { lhsDilation, rhsDilation, featureGroupCount = 1 } = {}) {
  if (lhs.ndim < 2) throw new Error("lhs must have at least 2 dimensions");
  if (rhs.ndim < 2) throw new Error("rhs must have at least 2 dimensions");
  if (typeof padding === "string") {
    if (lhsDilation?.some((d) => d !== 1)) throw new Error("String padding is not supported for transposed convolutions");
    padding = padtypeToPads(lhs.shape.slice(2), rhs.shape.slice(2), windowStrides, rhsDilation ?? rep(rhs.ndim - 2, 1), padding);
  }
  if (featureGroupCount === 1) {
    const spatialDims = lhs.ndim - 2;
    const lowered = lowerConvTranspose1d(lhs, rhs, windowStrides, padding, lhsDilation ?? rep(spatialDims, 1), rhsDilation ?? rep(spatialDims, 1));
    if (lowered !== null) return lowered;
  }
  if (featureGroupCount !== 1) {
    const G = featureGroupCount;
    const [N, C_in, ...xs] = lhs.shape;
    const [C_out, C_in_per_group, ...ks] = rhs.shape;
    if (C_in % G !== 0) throw new Error(`featureGroupCount=${G} must divide input channels=${C_in}`);
    if (C_out % G !== 0) throw new Error(`featureGroupCount=${G} must divide output channels=${C_out}`);
    if (C_in / G !== C_in_per_group) throw new Error(`rhs input channels=${C_in_per_group} must equal lhs input channels / groups=${C_in / G}`);
    const result = conv$1(moveaxis$1(lhs.reshape([
      N,
      G,
      C_in / G,
      ...xs
    ]), 1, 0), rhs.reshape([
      G,
      C_out / G,
      C_in_per_group,
      ...ks
    ]), {
      vmapDims: 1,
      strides: windowStrides,
      padding,
      lhsDilation,
      rhsDilation
    });
    const ys = result.shape.slice(3);
    return moveaxis$1(result, 0, 1).reshape([
      N,
      C_out,
      ...ys
    ]);
  }
  return conv$1(lhs, rhs, {
    strides: windowStrides,
    padding,
    lhsDilation,
    rhsDilation
  });
}
function conv(lhs, rhs, windowStrides, padding) {
  return convGeneralDilated(lhs, rhs, windowStrides, padding);
}
function topK(x, k, axis = -1) {
  x = fudgeArray(x);
  axis = checkAxis(axis, x.ndim);
  const size2 = x.shape[axis];
  if (k < 0 || k > size2) throw new Error(`topK: k must be in the range [0, ${size2}], got ${k}`);
  if (k === 0) {
    const outShape = x.shape.slice();
    outShape[axis] = 0;
    return [zerosLike$1(x.ref, { shape: outShape }), zerosLike$1(x, {
      dtype: "int32",
      shape: outShape
    })];
  }
  x = flip$1(x, [axis]);
  x = moveaxis$1(x, axis, -1);
  const [y, yi] = argsort$1(x);
  const extract = (a) => {
    a = a.slice(...rep(a.ndim - 1, []), [-k]);
    return flip$1(moveaxis$1(a, -1, axis), [axis]);
  };
  return [extract(y), extract(yi.neg().add(size2 - 1))];
}
function sigmoid(x) {
  return reciprocal(exp(negative(x)).add(1));
}
var sparsePlus = jit$1((x) => {
  return where(x.ref.lessEqual(-1), 0, where(x.ref.less(1), square(x.ref.add(1)).mul(0.25), x));
});
var sparseSigmoid = jit$1((x) => {
  return clip(x.add(1).mul(0.5), 0, 1);
});
var silu = jit$1(function silu2(x) {
  return x.ref.mul(sigmoid(x));
});
var selu = jit$1(function selu2(x) {
  return where(x.ref.less(0), expm1(x.ref).mul(1.6732632423543772), x).mul(1.0507009873554805);
});
var gelu = jit$1(function gelu2(x, opts) {
  if (opts?.approximate ?? true) {
    const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
    return x.ref.mul(0.5).mul(tanh(x.ref.mul(x.ref.mul(x).mul(0.044715).add(1)).mul(SQRT_2_OVER_PI)).add(1));
  } else return x.ref.mul(0.5).mul(erfc$1(negative(x.mul(Math.SQRT1_2))));
}, { staticArgnums: [1] });
var log1mexp = jit$1(function log1mexp2(x) {
  return where(x.ref.lessEqual(Math.LN2), log(negative(expm1(negative(x.ref)))), log1p(negative(exp(negative(x)))));
});
var JsArray = globalThis.Array;
function validateKeyShape(key, scalar2 = false) {
  if (key.ndim === 0) throw new Error("Key must have at least one dimension.");
  if (key.shape[key.shape.length - 1] !== 2) throw new Error(`Invalid key shape: ${key.shape}. Expected last dimension to be 2.`);
  if (scalar2 && key.shape.length > 1) throw new Error(`Expected a single PRNG key, but got a batch of keys with shape ${JSON.stringify(key.shape)} - use jax.vmap for batching.`);
  return key.shape.slice(0, -1);
}
function getK01(key) {
  const keyShape = validateKeyShape(key, true);
  let [k0, k1] = split$2(key, -1, [1, 1]);
  k0 = k0.reshape(keyShape);
  k1 = k1.reshape(keyShape);
  return [k0, k1];
}
function split(key, num = 2) {
  const shape2 = typeof num === "number" ? [num] : num;
  for (const len of shape2) if (len <= 0 || !Number.isInteger(len)) throw new Error(`Invalid split length: ${len}. Must be a positive integer.`);
  const [k0, k1] = getK01(key);
  return stack([randomBits(k0.ref, k1.ref, shape2, 0), randomBits(k0, k1, shape2, 1)], -1);
}
function bits(key, shape2 = []) {
  const [k0, k1] = getK01(key);
  return randomBits(k0, k1, shape2);
}
var uniform = jit$1(function uniform2(key, shape2 = [], { minval = 0, maxval = 1 } = {}) {
  if (minval >= maxval) throw new Error(`Invalid range: [${minval}, ${maxval}).`);
  const rand = bitcast(bits(key, shape2).div(array(512, {
    dtype: "uint32",
    device: key.device
  })).add(array(1065353216, {
    dtype: "uint32",
    device: key.device
  })), "float32").sub(1);
  if (minval === 0 && maxval === 1) return rand;
  else return rand.mul(maxval - minval).add(minval);
}, { staticArgnums: [1, 2] });
var ball = jit$1(function ball2(key, d, { p = 2, shape: shape2 = [] } = {}) {
  if (!Number.isInteger(d) || d <= 0) throw new Error(`ball: dimension must be a positive integer, got ${d}`);
  if (p !== 2) throw new Error("ball: only the Euclidean p=2 case is supported");
  const [k1, k2] = split(key, 2);
  const z = normal(k1, [...shape2, d]);
  const norm2 = sqrt(z.ref.mul(z.ref).sum(-1, { keepdims: true }));
  const radius = exp(log(uniform(k2, [...shape2, 1])).mul(1 / d));
  return z.div(norm2).mul(radius);
}, { staticArgnums: [1, 2] });
function bernoulli(key, p = 0.5, shape2 = []) {
  p = fudgeArray(p);
  return uniform(key, shape2).less(p);
}
var categorical = jit$1(function categorical2(key, logits, { axis = -1, shape: shape2, replace = true } = {}) {
  logits = fudgeArray(logits);
  axis = checkAxis(axis, logits.ndim);
  const numCategories = logits.shape[axis];
  const batchShape = logits.shape.toSpliced(axis, 1);
  if (shape2 === void 0) shape2 = batchShape;
  else if (!deepEqual(generalBroadcast(shape2, batchShape), shape2)) throw new Error(`Shape ${shape2} is not broadcast-compatible with batch shape ${batchShape}.`);
  const shapePrefix = shape2.slice(0, shape2.length - batchShape.length);
  if (replace) return argmax(gumbel(key, [...shapePrefix, ...logits.shape]).add(logits), axis + shapePrefix.length);
  else {
    const k = shapePrefix.reduce((a, b) => a * b, 1);
    if (k > numCategories) throw new Error(`Number of samples without replacement (${k}) cannot exceed number of categories (${numCategories}).`);
    const [values, indices2] = topK(gumbel(key, logits.shape).add(logits), k, axis);
    values.dispose();
    return indices2.reshape(shape2);
  }
}, { staticArgnums: [2] });
var cauchy = jit$1(function cauchy2(key, shape2 = []) {
  return tan(uniform(key, shape2).sub(0.5).mul(Math.PI));
}, { staticArgnums: [1] });
var doubleSidedMaxwell = jit$1(function doubleSidedMaxwell2(key, loc, scale, shape2 = []) {
  loc = fudgeArray(loc);
  scale = fudgeArray(scale);
  const [k1, k2] = split(key, 2);
  return rademacher(k1, {
    shape: shape2,
    dtype: "float32"
  }).mul(maxwell(k2, shape2)).mul(scale).add(loc);
}, { staticArgnums: [3] });
var exponential = jit$1(function exponential2(key, shape2 = []) {
  return negative(log1p(negative(uniform(key, shape2))));
}, { staticArgnums: [1] });
var geometric = jit$1(function geometric2(key, p, { shape: shape2 = [], dtype = "int32" } = {}) {
  p = fudgeArray(p);
  return floor(log1p(negative(uniform(key, shape2))).div(log1p(negative(p)))).add(1).astype(dtype);
}, { staticArgnums: [2] });
var gumbel = jit$1(function gumbel2(key, shape2 = []) {
  return negative(log(negative(log1p(negative(uniform(key, shape2))))));
}, { staticArgnums: [1] });
var laplace = jit$1(function laplace2(key, shape2 = []) {
  const centered = uniform(key, shape2).sub(0.5);
  const s = sign(centered.ref);
  const absVal = absolute(centered);
  return s.mul(log1p(absVal.mul(-2)).mul(-1));
}, { staticArgnums: [1] });
var logistic = jit$1(function logistic2(key, shape2 = []) {
  const u = uniform(key, shape2);
  return log(u.ref).sub(log1p(negative(u)));
}, { staticArgnums: [1] });
var lognormal = jit$1(function lognormal2(key, sigma = 1, shape2 = []) {
  sigma = fudgeArray(sigma);
  return exp(normal(key, shape2).mul(sigma));
}, { staticArgnums: [2] });
var maxwell = jit$1(function maxwell2(key, shape2 = []) {
  const z = normal(key, [...shape2, 3]);
  return sqrt(z.ref.mul(z).sum(-1));
}, { staticArgnums: [1] });
var multivariateNormal = jit$1(function multivariateNormal2(key, mean2, cov2, shape2 = []) {
  mean2 = fudgeArray(mean2);
  cov2 = fudgeArray(cov2);
  const n = mean2.shape[mean2.ndim - 1];
  if (cov2.shape[cov2.ndim - 1] !== n || cov2.shape[cov2.ndim - 2] !== n) throw new Error(`Invalid covariance shape: ${cov2.shape}. Expected last two dimensions to be [${n}, ${n}].`);
  const outputShape = broadcastShapes(shape2, mean2.shape.slice(0, -1), cov2.shape.slice(0, -2)).concat(n);
  return einsum("...ij,...j->...i", cholesky$1(cov2), normal(key, outputShape)).add(mean2);
}, { staticArgnums: [3] });
var normal = jit$1(function normal2(key, shape2 = []) {
  const [k1, k2] = split(key, 2);
  const u1 = uniform(k1, shape2);
  const u2 = uniform(k2, shape2);
  const radius = sqrt(log1p(negative(u1)).mul(-2));
  const theta = u2.mul(2 * Math.PI);
  return radius.mul(cos(theta));
}, { staticArgnums: [1] });
var pareto = jit$1(function pareto2(key, b, shape2 = []) {
  b = fudgeArray(b);
  return exp(exponential(key, shape2).div(b));
}, { staticArgnums: [2] });
var rademacher = jit$1(function rademacher2(key, { shape: shape2 = [], dtype = "int32" } = {}) {
  if (dtype === "uint32" || dtype === "bool") throw new Error(`rademacher: unsupported dtype ${dtype}`);
  const one = array(1, {
    dtype,
    device: key.device
  });
  const minusOne = array(-1, {
    dtype,
    device: key.device
  });
  return where(bernoulli(key, 0.5, shape2), one, minusOne);
}, { staticArgnums: [1] });
var randint = jit$1(function randint2(key, { minval, maxval, shape: shape2 = [], dtype = "int32" }) {
  if (!Number.isInteger(minval) || !Number.isInteger(maxval)) throw new Error("randint: minval and maxval must be integers");
  if (minval >= maxval) throw new Error(`Invalid range: [${minval}, ${maxval}).`);
  if (dtype !== "int32" && dtype !== "uint32") throw new Error(`randint: dtype must be int32 or uint32, got ${dtype}`);
  if (dtype === "uint32" && minval < 0) throw new Error("randint: uint32 dtype requires minval >= 0");
  const range2 = maxval - minval;
  return bits(key, shape2).mod(range2).astype(dtype).add(minval);
}, { staticArgnums: [1] });
var rayleigh = jit$1(function rayleigh2(key, scale = 1, shape2 = []) {
  scale = fudgeArray(scale);
  return sqrt(exponential(key, shape2).mul(2)).mul(scale);
}, { staticArgnums: [2] });
var triangular = jit$1(function triangular2(key, left, mode, right, shape2 = []) {
  left = fudgeArray(left);
  mode = fudgeArray(mode);
  right = fudgeArray(right);
  const u = uniform(key, shape2);
  const width = right.ref.sub(left.ref);
  const leftSpan = mode.ref.sub(left.ref);
  const rightSpan = right.ref.sub(mode);
  const cutoff = leftSpan.ref.div(width.ref);
  return where(u.ref.less(cutoff), left.add(sqrt(u.ref.mul(width.ref).mul(leftSpan))), right.sub(sqrt(negative(u).add(1).mul(width).mul(rightSpan))));
}, { staticArgnums: [4] });
var weibullMin = jit$1(function weibullMin2(key, scale, concentration, shape2 = []) {
  scale = fudgeArray(scale);
  concentration = fudgeArray(concentration);
  return scale.mul(exp(log(exponential(key, shape2)).div(concentration)));
}, { staticArgnums: [3] });
var logit = jit$1(function logit2(x) {
  return log(x.ref.div(subtract(1, x)));
});
Symbol.dispose ??= Symbol.for("Symbol.dispose");
Symbol.asyncDispose ??= Symbol.for("Symbol.asyncDispose");

// forward.js
var isNode = typeof process !== "undefined" && Boolean(process.versions?.node);
var fs = isNode ? (await import("node:fs")).default : null;
var MAGIC = "NEEDLEJS1";
var HEADER_BYTES = 4;
var CACT_TAG = 98642563;
var CACT_HDR_BYTES = 120;
var CACT_REC_BYTES = 44;
var D = numpy_exports.float32;
var HELP = `Usage: node forward.js [weights.bin] [options]

Run the Needle forward/verification path and print the final-token logits.

Options:
  --tokens=<ids>          Input token ids, as JSON or whitespace/comma-separated ids
  --prefill-file=<path>   Read input token ids from a JSON/text file
  --cact=<path>           Load quantized weights from a .cact export instead of weights.bin
  --cact <path>           Same as --cact=<path>
  --weights=<path>        Load weights from a weights.bin-compatible file
  --weights <path>        Same as --weights=<path>
  --w4=<path>             Load packed W4 weights
  --w4 <path>             Same as --w4=<path>
  --dump-weights=<path>   Export packed W4 weights
  --dump-weights <path>   Same as --dump-weights=<path>
  --compare-quant         Compare FP32, group-wise W4 and W8 outputs per layer
  --quant-group=<n>       Group size for --compare-quant (default: 128)
  --help                  Show this help message

If neither --tokens nor --prefill-file is provided, input_tokens from the
weights header are used, falling back to [1, 2, 3, 4].

The optional positional argument selects the weights file and defaults to
weights.bin. --cact switches the source to the Cactus quantized export.
`;
function normalizeConfig(c) {
  const numeric = [
    "vocab_size",
    "d_model",
    "attn_dim",
    "num_heads",
    "num_kv_heads",
    "num_layers",
    "max_seq_len",
    "pad_token_id",
    "contrastive_dim",
    "rope_theta",
    "engram_heads",
    "engram_slots",
    "kv_window",
    "kv_bits",
    "act_bits",
    "scan_unroll",
    "mhc_lanes"
  ];
  const out = { ...c };
  for (const k of numeric) if (k in out) out[k] = Number(out[k]);
  out.engram_orders = (out.engram_orders ?? [2, 3]).map(Number);
  out.engram_layers = (out.engram_layers ?? [2, 15]).map(Number);
  out.flash = out.flash === true || out.flash === "True" || out.flash === "true";
  out.remat = out.remat === true || out.remat === "True" || out.remat === "true";
  return out;
}
function parseTokenList(value, source = "tokens") {
  let parsed = value;
  if (!Array.isArray(value)) {
    const text = String(value).trim();
    if (text.startsWith("[")) {
      try {
        parsed = JSON.parse(text);
      } catch (err) {
        throw new Error(`invalid ${source} JSON: ${err.message}`);
      }
    } else {
      parsed = text.split(/[\s,]+/).filter(Boolean);
    }
  }
  if (!Array.isArray(parsed))
    throw new Error(`${source} must be an array of token ids`);
  const tokens = parsed.map((item, i) => {
    const id = item && typeof item === "object" ? item.token_id : item;
    const n = Number(id);
    if (!Number.isInteger(n) || n < 0)
      throw new Error(`invalid token id at ${source}[${i}]: ${id}`);
    return n;
  });
  if (!tokens.length)
    throw new Error(`${source} must contain at least one token`);
  return tokens;
}
function readPrefill(path) {
  let value;
  try {
    value = JSON.parse(fs.readFileSync(path, "utf8"));
  } catch (err) {
    throw new Error(`unable to read prefill file ${path}: ${err.message}`);
  }
  return parseTokenList(value, path);
}
function loadTokenMetadata(header) {
  if (Array.isArray(header.token_metadata)) return header.token_metadata;
  const vocabPath = process.env.NEEDLE_TOKENIZER_VOCAB || "needle/needle/model/tokenizer.vocab";
  try {
    const lines = fs.readFileSync(vocabPath, "utf8").split(/\r?\n/);
    return lines.filter((line) => line.length > 0).map((line, tokenId) => {
      const piece = line.split("	", 1)[0];
      const byteMatch = piece.match(/^<0x([0-9A-Fa-f]{2})>$/);
      const tokenText2 = byteMatch ? String.fromCodePoint(parseInt(byteMatch[1], 16)) : piece.replaceAll(String.fromCodePoint(9601), " ");
      return {
        token_id: tokenId,
        token_text: tokenText2,
        token_bytes_hex: Buffer.from(tokenText2, "utf8").toString("hex").match(/../g)?.join(" ") || ""
      };
    });
  } catch (_) {
    return [];
  }
}
function isPackedW4Tensor(name, shape2) {
  return shape2.length >= 2 && (name === "embedding/embedding" || name.includes("/kernel") || name.includes("/mhc_phi_") || name.startsWith("engrams_") && name.endsWith("/embedding"));
}
function packW4(name, value, group = 128) {
  const shape2 = [...value.shape];
  const src = value.ref.dataSync();
  const reduceSecondLast = name.includes("/kernel") || name.includes("/mhc_phi_");
  const quantDim = reduceSecondLast ? shape2.at(-2) : shape2.at(-1);
  const columns = reduceSecondLast ? shape2.at(-1) : 1;
  const rows = src.length / (quantDim * columns);
  const groupsPerRow = Math.ceil(quantDim / group);
  const packed = Buffer.alloc(Math.ceil(src.length / 2));
  const scales = new Float32Array(rows * columns * groupsPerRow);
  let scaleIndex = 0;
  for (let row = 0; row < rows; ++row) for (let col = 0; col < columns; ++col) {
    const base = row * quantDim * columns + col;
    for (let start = 0; start < quantDim; start += group) {
      const end = Math.min(quantDim, start + group);
      let scale = 0;
      for (let i = start; i < end; ++i) scale = Math.max(scale, Math.abs(src[base + i * columns]));
      scale = scale > 0 ? scale / 7 : 1;
      scales[scaleIndex++] = scale;
      for (let i = start; i < end; ++i) {
        const q = Math.max(-8, Math.min(7, Math.round(src[base + i * columns] / scale))) + 8;
        const index = base + i * columns;
        if (index & 1) packed[index >> 1] |= q << 4;
        else packed[index >> 1] |= q;
      }
    }
  }
  return { packed, scales: Buffer.from(new Uint8Array(scales.buffer)), shape: shape2, group };
}
function unpackW4(bytes, entry, dataStart) {
  const packedBytes = entry.packed_bytes;
  const packed = bytes.subarray(dataStart + entry.offset, dataStart + entry.offset + packedBytes);
  const scalesBytes = bytes.subarray(
    dataStart + entry.scales_offset,
    dataStart + entry.scales_offset + entry.scales_nbytes
  );
  const scales = new Float32Array(Uint8Array.from(scalesBytes).buffer);
  const shape2 = entry.shape;
  const reduceSecondLast = entry.name.includes("/kernel") || entry.name.includes("/mhc_phi_");
  const quantDim = reduceSecondLast ? shape2.at(-2) : shape2.at(-1);
  const columns = reduceSecondLast ? shape2.at(-1) : 1;
  const elementCount = shape2.reduce((a, b) => a * b, 1);
  const rows = elementCount / (quantDim * columns);
  const expectedScales = rows * columns * Math.ceil(quantDim / entry.group_size);
  if (scales.length !== expectedScales)
    throw new Error(`invalid W4 scales for ${entry.name}: expected ${expectedScales}, got ${scales.length}`);
  const out = new Float32Array(elementCount);
  let scaleIndex = 0;
  for (let row = 0; row < rows; ++row) for (let col = 0; col < columns; ++col) {
    const base = row * quantDim * columns + col;
    for (let start = 0; start < quantDim; start += entry.group_size) {
      const scale = scales[scaleIndex++];
      const end = Math.min(quantDim, start + entry.group_size);
      for (let i = start; i < end; ++i) {
        const index = base + i * columns;
        const q = (packed[index >> 1] >> (index & 1) * 4 & 15) - 8;
        out[index] = q * scale;
      }
    }
  }
  return numpy_exports.array(out, { dtype: D }).reshape(shape2);
}
function readWeights(path = "weights.bin") {
  if (!fs) throw new Error("readWeights is only available in Node.js");
  const buf = fs.readFileSync(path);
  const magic = Buffer.from(buf.subarray(0, MAGIC.length)).toString("ascii");
  if (magic !== MAGIC) throw new Error(`bad weights.bin magic: ${magic}`);
  const headerLen = buf.readUInt32LE(MAGIC.length);
  const headerStart = MAGIC.length + HEADER_BYTES;
  const header = JSON.parse(
    Buffer.from(buf.subarray(headerStart, headerStart + headerLen)).toString(
      "utf8"
    )
  );
  const dataStart = headerStart + headerLen;
  const weights = {};
  for (const e2 of header.tensors) {
    if (e2.encoding === "w4") {
      weights[e2.name] = unpackW4(buf, e2, dataStart);
      continue;
    }
    const bytes = buf.subarray(
      dataStart + e2.offset,
      dataStart + e2.offset + e2.nbytes
    );
    const raw = new Float32Array(Uint8Array.from(bytes).buffer);
    weights[e2.name] = numpy_exports.array(raw, { dtype: D }).reshape(e2.shape);
  }
  let reference = null;
  if (header.reference) {
    const r = buf.subarray(
      dataStart + header.reference.offset,
      dataStart + header.reference.offset + header.reference.nbytes
    );
    reference = new Float32Array(Uint8Array.from(r).buffer);
  }
  return { header: { ...header, reference }, weights };
}
function writeWeights(path, header, weights, group = 128) {
  if (!fs) throw new Error("writeWeights is only available in Node.js");
  const tensors = [];
  const chunks = [];
  let dataOffset = 0;
  const append2 = (chunk) => {
    chunks.push(chunk);
    const offset = dataOffset;
    dataOffset += chunk.length;
    return offset;
  };
  for (const [name, value] of Object.entries(weights)) {
    const shape2 = [...value.shape];
    if (isPackedW4Tensor(name, shape2)) {
      const packed = packW4(name, value, group);
      const offset = append2(packed.packed);
      const scalesOffset = append2(packed.scales);
      tensors.push({
        name,
        shape: shape2,
        encoding: "w4",
        offset,
        packed_bytes: packed.packed.length,
        scales_offset: scalesOffset,
        scales_nbytes: packed.scales.length,
        group_size: group
      });
    } else {
      const host = value.ref.dataSync();
      const raw = Buffer.from(host.buffer, host.byteOffset, host.byteLength);
      const offset = append2(raw);
      tensors.push({ name, shape: shape2, encoding: "f32", offset, nbytes: raw.length });
    }
  }
  let reference = null;
  if (header.reference) {
    const ref = Buffer.from(
      header.reference.buffer,
      header.reference.byteOffset,
      header.reference.byteLength
    );
    const offset = append2(ref);
    reference = { offset, nbytes: ref.length, shape: [ref.length / 4] };
  }
  const outHeader = {
    ...header,
    format: "needle-timemachine.jaxjs-weights/v2",
    quantization: { scheme: "symmetric-per-group", bits: 4, group_size: group },
    tensors,
    reference
  };
  delete outHeader.config?.reference;
  const headerBytes = Buffer.from(JSON.stringify(outHeader), "utf8");
  const length = Buffer.alloc(4);
  length.writeUInt32LE(headerBytes.length);
  const output = Buffer.concat([
    Buffer.from(MAGIC, "ascii"),
    length,
    headerBytes,
    ...chunks
  ]);
  fs.writeFileSync(path, output);
  console.log(`weights:    ${path}`);
  console.log(`tensors:    ${tensors.length}`);
  console.log(`bytes:      ${output.length}`);
}
function halfToFloat(h) {
  const s = h >>> 15 & 1, e2 = h >>> 10 & 31, f = h & 1023;
  if (e2 === 0) return (s ? -1 : 1) * (f ? Math.pow(2, -14) * (f / 1024) : 0);
  if (e2 === 31) return f ? NaN : s ? -Infinity : Infinity;
  return (s ? -1 : 1) * Math.pow(2, e2 - 15) * (1 + f / 1024);
}
function cactCodebook(codebook, bits2, group) {
  if (bits2 === 5) {
    const c = 1.2240064 / Math.sqrt(group);
    return new Float32Array([-c, 0, c]);
  }
  const start = bits2 === 2 ? 0 : bits2 === 3 ? 4 : bits2 === 4 ? 12 : -1;
  const len = 1 << bits2;
  if (start < 0 || start + len > codebook.length)
    throw new Error(`unsupported .cact CQ bits=${bits2}`);
  return codebook.subarray(start, start + len);
}
function unpackCactIndex(bytes, byteBase, bits2, k) {
  if (bits2 === 5) {
    const crumb = bytes[byteBase + (k >> 2)] >>> (k & 3) * 2 & 3;
    return crumb === 3 ? 0 : crumb + 1;
  }
  const chunk = Math.floor(k / 8), inChunk = k & 7;
  let word = 0;
  const p = byteBase + chunk * bits2;
  for (let b = 0; b < bits2; ++b) word |= bytes[p + b] << 8 * b;
  return word >>> inChunk * bits2 & (1 << bits2) - 1;
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
function decodeCactCQ(bytes, shape2, offset, nbytes, group, bits2, codebook) {
  const [out, inDim] = shape2;
  const inPad = Math.ceil(inDim / group) * group;
  const packedPerRow = bits2 === 5 ? inPad / 4 : inPad * bits2 / 8;
  const packedBytes = out * packedPerRow;
  if (!Number.isInteger(packedBytes) || packedBytes > nbytes)
    throw new Error(`invalid .cact CQ payload for ${out}x${inDim}`);
  const normBase = offset + packedBytes;
  const result = new Float32Array(out * inDim);
  const cb = cactCodebook(codebook, bits2, group);
  const tmp = new Float32Array(group);
  for (let r = 0; r < out; ++r) {
    const rowPacked = offset + r * packedPerRow;
    const rowNorm = normBase + r * (inPad / group) * 2;
    for (let g = 0; g < inPad / group; ++g) {
      const norm2 = halfToFloat(
        bytes[rowNorm + g * 2] | bytes[rowNorm + g * 2 + 1] << 8
      );
      const baseK = g * group;
      for (let j = 0; j < group; ++j)
        tmp[j] = cb[unpackCactIndex(bytes, rowPacked, bits2, baseK + j)] * norm2;
      fwhtInPlace(tmp);
      const keep = Math.min(group, inDim - baseK);
      if (keep > 0) result.set(tmp.subarray(0, keep), r * inDim + baseK);
    }
  }
  return result;
}
function readCact(path) {
  if (!fs) throw new Error("readCact is only available in Node.js");
  const buf = fs.readFileSync(path);
  if (buf.length < CACT_HDR_BYTES)
    throw new Error(`.cact file is too small: ${path}`);
  const tag = buf.readUInt32LE(0);
  if (tag !== CACT_TAG) throw new Error(`bad .cact tag: 0x${tag.toString(16)}`);
  const numTensors = buf.readUInt32LE(4);
  const codebookLen = buf.readUInt32LE(8);
  const header = {
    kv_window: buf.readUInt32LE(12),
    kv_bits: buf.readUInt32LE(16),
    vocab_size: buf.readUInt32LE(20),
    d_model: buf.readUInt32LE(24),
    num_heads: buf.readUInt32LE(28),
    num_kv_heads: buf.readUInt32LE(32),
    num_layers: buf.readUInt32LE(36),
    head_dim: buf.readUInt32LE(40),
    max_seq_len: buf.readUInt32LE(44),
    hada_n: buf.readUInt32LE(48),
    mhc_lanes: buf.readUInt32LE(52),
    engram_slots: buf.readUInt32LE(56),
    engram_sub_dim: buf.readUInt32LE(60),
    num_engram_tables: buf.readUInt32LE(64),
    engram_conv_taps: buf.readUInt32LE(68),
    engram_conv_dilation: buf.readUInt32LE(72)
  };
  header.attn_dim = header.num_heads * header.head_dim;
  const numOrders = buf.readUInt32LE(76);
  header.engram_orders = Array.from(
    { length: numOrders },
    (_, i) => buf.readUInt32LE(80 + i * 4)
  );
  const numSites = buf.readUInt32LE(96);
  header.engram_layers = Array.from(
    { length: numSites },
    (_, i) => buf.readUInt32LE(100 + i * 4)
  );
  header.rope_theta = buf.readFloatLE(116);
  if (codebookLen !== 28)
    throw new Error(`unsupported .cact codebook length: ${codebookLen}`);
  const codebook = new Float32Array(
    buf.buffer,
    buf.byteOffset + CACT_HDR_BYTES,
    codebookLen
  );
  const recordsStart = CACT_HDR_BYTES + codebookLen * 4;
  if (recordsStart + numTensors * CACT_REC_BYTES > buf.length)
    throw new Error("truncated .cact tensor directory");
  const tensors = [];
  for (let i = 0; i < numTensors; ++i) {
    const p = recordsStart + i * CACT_REC_BYTES;
    const dtype = buf.readUInt8(p), ndim2 = buf.readUInt8(p + 1);
    if (ndim2 > 4)
      throw new Error(`invalid .cact tensor ndim=${ndim2} at index ${i}`);
    const shape2 = [];
    for (let d = 0; d < ndim2; ++d) shape2.push(buf.readUInt32LE(p + 4 + d * 4));
    const offset = Number(buf.readBigUInt64LE(p + 20));
    const nbytes = Number(buf.readBigUInt64LE(p + 28));
    const group = buf.readUInt32LE(p + 36), bits2 = buf.readUInt32LE(p + 40);
    if (offset + nbytes > buf.length)
      throw new Error(`truncated .cact tensor ${i}`);
    let value;
    if (dtype === 3)
      value = decodeCactCQ(buf, shape2, offset, nbytes, group, bits2, codebook);
    else if (dtype === 1) {
      const n = shape2.reduce((a, b) => a * b, 1), out = new Float32Array(n);
      for (let j = 0; j < n; ++j)
        out[j] = halfToFloat(buf.readUInt16LE(offset + j * 2));
      value = out;
    } else if (dtype === 2) {
      value = new Float32Array(
        buf.buffer,
        buf.byteOffset + offset,
        nbytes / 4
      ).slice();
    } else if (dtype === 4) value = buf.subarray(offset, offset + nbytes);
    else throw new Error(`unsupported .cact dtype=${dtype} at index ${i}`);
    tensors.push({ dtype, shape: shape2, value });
  }
  if (tensors.length !== 405)
    throw new Error(`unexpected .cact tensor count ${tensors.length}`);
  const names = ["embedding"];
  for (let l = 0; l < header.num_layers; ++l)
    names.push(
      `layer${l}.norm_in`,
      `layer${l}.q_proj`,
      `layer${l}.k_proj`,
      `layer${l}.v_proj`,
      `layer${l}.q_norm`,
      `layer${l}.k_norm`,
      `layer${l}.gate_proj`,
      `layer${l}.out_proj`,
      `layer${l}.post_norm`,
      `layer${l}.attn_gate`,
      `layer${l}.pre_hada`,
      `layer${l}.d1`,
      `layer${l}.d2`,
      `layer${l}.d3`
    );
  names.push(
    "mhc_a_pre",
    "mhc_a_post",
    "mhc_a_res",
    "mhc_b_pre",
    "mhc_b_post",
    "mhc_b_res",
    "mhc_phi_pre",
    "mhc_phi_post",
    "mhc_phi_res"
  );
  for (let s = 0; s < header.engram_layers.length; ++s)
    names.push(
      `engram${s}.tables`,
      `engram${s}.key_proj`,
      `engram${s}.value_proj`,
      `engram${s}.taps`
    );
  names.push(
    "final_norm",
    "heads.manifest",
    "contrastive_head.probes",
    "contrastive_head.proj",
    "contrastive_head.bias",
    "confidence_head.probes",
    "confidence_head.proj",
    "confidence_head.bias",
    "tokenizer"
  );
  if (names.length !== tensors.length)
    throw new Error(
      `unsupported .cact layout: expected ${names.length} records, got ${tensors.length}`
    );
  const weights = {};
  const add2 = (name, t, shape2 = t.shape) => {
    if (t.dtype === 4) return;
    weights[name] = numpy_exports.array(t.value, { dtype: D }).reshape(shape2);
  };
  add2("embedding/embedding", tensors[0], tensors[0].shape);
  let k = 1;
  const layerWeights = {};
  const addLayer = (name, t, shape2) => {
    const value = numpy_exports.array(t.value, { dtype: D }).reshape(shape2);
    (layerWeights[name] ??= []).push(value);
  };
  for (let l = 0; l < header.num_layers; ++l) {
    const p = "stack/layers/block/";
    addLayer(`${p}ZCRMSNorm_0/scale`, tensors[k++], [header.d_model]);
    for (const q of ["q_proj", "k_proj", "v_proj"]) {
      const t = tensors[k++];
      addLayer(`${p}self_attn/${q}/kernel`, t, [t.shape[1], t.shape[0]]);
    }
    addLayer(`${p}self_attn/q_norm/scale`, tensors[k++], [header.head_dim]);
    addLayer(`${p}self_attn/k_norm/scale`, tensors[k++], [header.head_dim]);
    for (const q of ["gate_proj", "out_proj"]) {
      const t = tensors[k++];
      addLayer(`${p}self_attn/${q}/kernel`, t, [t.shape[1], t.shape[0]]);
    }
    addLayer(`${p}post_attn_norm/scale`, tensors[k++], [header.d_model]);
    addLayer(`${p}attn_gate`, tensors[k++], [1]);
    addLayer(`${p}pre_hada_norm/scale`, tensors[k++], [header.d_model]);
    for (const q of ["d1", "d2", "d3"])
      addLayer(`${p}hadamard_mlp/${q}`, tensors[k++], [header.d_model]);
  }
  for (const [name, values] of Object.entries(layerWeights))
    weights[name] = numpy_exports.stack(values, 0);
  for (const q of [
    "mhc_a_pre",
    "mhc_a_post",
    "mhc_a_res",
    "mhc_b_pre",
    "mhc_b_post",
    "mhc_b_res"
  ]) {
    const t = tensors[k++];
    add2(`stack/${q}`, t, t.shape);
  }
  for (const q of ["mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"]) {
    const t = tensors[k++];
    const [rows, nC] = t.shape;
    const lanes = q === "mhc_phi_res" ? header.mhc_lanes * header.mhc_lanes : header.mhc_lanes;
    add2(`stack/${q}`, t, [header.num_layers, nC, lanes]);
  }
  for (let s = 0; s < header.engram_layers.length; ++s) {
    const t = tensors[k++];
    add2(`engrams_${s}/embedding`, t, [
      header.num_engram_tables,
      header.engram_slots,
      header.engram_sub_dim
    ]);
    for (const q of ["key_proj", "value_proj"]) {
      const u = tensors[k++];
      add2(`engrams_${s}/${q}/kernel`, u, [u.shape[1], u.shape[0]]);
    }
    add2(`engrams_${s}/taps`, tensors[k++], tensors[k - 1].shape);
  }
  add2("stack/final_norm/scale", tensors[k++], tensors[k - 1].shape);
  return { header, weights };
}
function sl(x, starts, ends) {
  return x.ref.slice(...starts.map((start, i) => [start, ends[i]]));
}
function scalar(x) {
  return numpy_exports.array(new Float32Array([x]), { dtype: D }).reshape([]);
}
function sigmoid2(x) {
  return numpy_exports.divide(1, numpy_exports.add(1, numpy_exports.exp(numpy_exports.negative(x))));
}
function silu3(x) {
  return x.mul(sigmoid2(x.ref));
}
function rmsUnit(x, eps = 1e-6) {
  const xf = x.astype(D), sq = xf.ref.mul(xf.ref), mean2 = numpy_exports.mean(sq, -1).reshape([...sq.shape.slice(0, -1), 1]);
  return xf.mul(numpy_exports.reciprocal(numpy_exports.sqrt(numpy_exports.add(mean2, scalar(eps)))));
}
function zcrmsNorm(x, scale, eps = 1e-6) {
  return numpy_exports.multiply(numpy_exports.add(1, scale), rmsUnit(x, eps));
}
function softmax(x, axis = -1) {
  const a = axis < 0 ? x.shape.length + axis : axis, outShape = [...x.shape];
  outShape[a] = 1;
  const m = numpy_exports.max(x.ref, axis).reshape(outShape), e2 = numpy_exports.exp(x.sub(m)), z = numpy_exports.sum(e2.ref, axis).reshape(outShape);
  return e2.div(z);
}
function linear(x, kernel) {
  return numpy_exports.matmul(x, kernel);
}
function shiftRight(x, offset, axis = 1) {
  if (offset === 0) return x;
  const shape2 = [...x.shape];
  if (axis !== 1)
    throw new Error("shiftRight currently expects sequence axis=1");
  if (offset >= shape2[axis]) return numpy_exports.zeros(shape2, { dtype: x.dtype });
  const zshape = [...shape2];
  zshape[axis] = offset;
  return numpy_exports.concatenate(
    [
      numpy_exports.zeros(zshape, { dtype: x.dtype }),
      sl(x, [0, 0], [shape2[0], shape2[1] - offset])
    ],
    1
  );
}
function rope(x, cos2, sin2) {
  const h = x.shape.at(-1) / 2;
  const x1 = sl(x.ref, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], h]), x2 = sl(
    x.ref,
    [0, 0, 0, h],
    [x.shape[0], x.shape[1], x.shape[2], x.shape[3]]
  );
  const c = sl(cos2.ref, [0, 0], [x.shape[2], h]).reshape([1, 1, x.shape[2], h]), s = sl(sin2.ref, [0, 0], [x.shape[2], h]).reshape([1, 1, x.shape[2], h]);
  return numpy_exports.concatenate(
    [x1.ref.mul(c.ref).sub(x2.ref.mul(s.ref)), x2.mul(c).add(x1.mul(s))],
    -1
  );
}
function ropeFreqs(headDim, seqLen, theta) {
  const half = Math.floor(headDim / 2), c = new Float32Array(seqLen * half), s = new Float32Array(seqLen * half);
  for (let t = 0; t < seqLen; ++t)
    for (let j = 0; j < half; ++j) {
      const a = t / Math.pow(theta, 2 * j / headDim);
      c[t * half + j] = Math.cos(a);
      s[t * half + j] = Math.sin(a);
    }
  return [
    numpy_exports.array(c, { dtype: D }).reshape([seqLen, half]),
    numpy_exports.array(s, { dtype: D }).reshape([seqLen, half])
  ];
}
function walsh(n) {
  let h = [[1]];
  while (h.length < n) {
    const m = h.length, next = Array.from({ length: m * 2 }, () => new Array(m * 2));
    for (let r = 0; r < m; ++r)
      for (let c = 0; c < m; ++c) {
        const v = h[r][c];
        next[r][c] = v;
        next[r][c + m] = v;
        next[r + m][c] = v;
        next[r + m][c + m] = -v;
      }
    h = next;
  }
  const a = new Float32Array(n * n), s = Math.sqrt(n);
  for (let i = 0; i < n; ++i)
    for (let j = 0; j < n; ++j) a[i * n + j] = h[i][j] / s;
  return numpy_exports.array(a, { dtype: D }).reshape([n, n]);
}
function engramGeometry(cfg) {
  const orders = cfg.engram_orders ?? [2, 3], heads = cfg.engram_heads || Math.max(1, Math.floor(cfg.d_model / (orders.length * 128))), subDim = Math.floor(cfg.d_model / (orders.length * heads));
  return { orders, heads, subDim };
}
function engramIndices(tokens, orders, heads, slots) {
  const B = tokens.shape[0], T = tokens.shape[1], host = tokens.dataSync(), out = [], SEED = 2654435769 >>> 0, PRIME = 16777619 >>> 0;
  for (let oi = 0; oi < orders.length; ++oi)
    for (let h = 0; h < heads; ++h) {
      const seed = Math.imul(SEED, oi * heads + h + 1) >>> 0, a = new Int32Array(B * T);
      for (let b = 0; b < B; ++b)
        for (let t = 0; t < T; ++t) {
          let acc = seed;
          for (let j = 0; j < orders[oi]; ++j) {
            const u = j <= t ? host[b * T + t - j] >>> 0 : 0;
            acc = Math.imul((acc ^ u) >>> 0, PRIME) >>> 0;
          }
          acc = (acc ^ acc >>> 15) >>> 0;
          a[b * T + t] = acc % slots;
        }
      out.push(numpy_exports.array(a, { dtype: numpy_exports.int32 }).reshape([B, T]));
    }
  return out;
}
function makeEngramKV(tokens, maskKeep, cfg, w) {
  if (!cfg.engram_layers?.length) return null;
  const { orders, heads, subDim } = engramGeometry(cfg), idx = engramIndices(tokens, orders, heads, cfg.engram_slots), numTables = orders.length * heads, embeddings = [];
  for (let s = 0; s < cfg.engram_layers.length; ++s) {
    const table = w[`engrams_${s}/embedding`], fetched = [];
    for (let j = 0; j < numTables; ++j) {
      const one = sl(
        table.ref,
        [j, 0, 0],
        [j + 1, table.shape[1], table.shape[2]]
      ).reshape([table.shape[1], table.shape[2]]), gathered = numpy_exports.take(one, idx[j].ref, 0), order = orders[Math.floor(j / heads)], ok = numpy_exports.array(
        new Float32Array(
          Array.from(
            { length: tokens.shape[1] },
            (_, t) => t >= order - 1 ? 1 : 0
          )
        ),
        { dtype: D }
      ).reshape([1, tokens.shape[1], 1]);
      fetched.push(gathered.mul(ok));
    }
    let e2 = numpy_exports.stack(fetched, 2).reshape([tokens.shape[0], tokens.shape[1], numTables * subDim]);
    e2 = e2.mul(maskKeep.ref.reshape([1, tokens.shape[1], 1]));
    embeddings.push(e2);
  }
  const ks = [], vs = [], maxOrder = Math.max(...orders);
  for (let s = 0; s < cfg.engram_layers.length; ++s) {
    const e2 = embeddings[s];
    let k = linear(e2.ref, w[`engrams_${s}/key_proj/kernel`]), v = linear(e2, w[`engrams_${s}/value_proj/kernel`]);
    const taps = w[`engrams_${s}/taps`];
    let vv = numpy_exports.zeros(v.shape, { dtype: D });
    for (let j = 0; j < 4; ++j) {
      const shifted = shiftRight(v.ref, j * maxOrder), tap = sl(taps.ref, [j, 0], [j + 1, cfg.d_model]).reshape([
        1,
        1,
        cfg.d_model
      ]), ok = numpy_exports.array(
        new Float32Array(
          Array.from(
            { length: tokens.shape[1] },
            (_, t) => t >= j * maxOrder ? 1 : 0
          )
        ),
        { dtype: D }
      ).reshape([1, tokens.shape[1], 1]);
      vv = vv.add(shifted.mul(tap).mul(ok));
    }
    ks.push(k);
    vs.push(vv);
  }
  return { k: numpy_exports.stack(ks, 0), v: numpy_exports.stack(vs, 0) };
}
function attention(x, layer, cfg, w, cos2, sin2, causal) {
  const prefix = "stack/layers/block/", qProj = sl(
    w[`${prefix}self_attn/q_proj/kernel`],
    [layer, 0, 0],
    [layer + 1, cfg.d_model, cfg.attn_dim]
  ).reshape([cfg.d_model, cfg.attn_dim]), kv = cfg.num_kv_heads * (cfg.attn_dim / cfg.num_heads), kProj = sl(
    w[`${prefix}self_attn/k_proj/kernel`],
    [layer, 0, 0],
    [layer + 1, cfg.d_model, kv]
  ).reshape([cfg.d_model, kv]), vProj = sl(
    w[`${prefix}self_attn/v_proj/kernel`],
    [layer, 0, 0],
    [layer + 1, cfg.d_model, kv]
  ).reshape([cfg.d_model, kv]), hd = cfg.attn_dim / cfg.num_heads, qNorm = sl(
    w[`${prefix}self_attn/q_norm/scale`],
    [layer, 0],
    [layer + 1, hd]
  ).reshape([hd]), kNorm = sl(
    w[`${prefix}self_attn/k_norm/scale`],
    [layer, 0],
    [layer + 1, hd]
  ).reshape([hd]), q0 = linear(x.ref, qProj).reshape([x.shape[0], x.shape[1], cfg.num_heads, hd]).transpose([0, 2, 1, 3]), k0 = linear(x.ref, kProj).reshape([x.shape[0], x.shape[1], cfg.num_kv_heads, hd]).transpose([0, 2, 1, 3]), v0 = linear(x.ref, vProj).reshape([x.shape[0], x.shape[1], cfg.num_kv_heads, hd]).transpose([0, 2, 1, 3]), q = rope(zcrmsNorm(q0, qNorm), cos2, sin2);
  let kk = rope(zcrmsNorm(k0, kNorm), cos2, sin2), v = v0;
  const repeat2 = cfg.num_heads / cfg.num_kv_heads;
  if (repeat2 > 1) {
    kk = numpy_exports.repeat(kk, repeat2, 1);
    v = numpy_exports.repeat(v, repeat2, 1);
  }
  let scores = numpy_exports.matmul(q, kk.transpose([0, 1, 3, 2])).div(Math.sqrt(hd));
  scores = numpy_exports.where(causal.ref, scores, -1e30);
  const p = softmax(scores, -1);
  let out = numpy_exports.matmul(p, v).transpose([0, 2, 1, 3]).reshape([x.shape[0], x.shape[1], cfg.attn_dim]);
  const gateKernel = sl(
    w[`${prefix}self_attn/gate_proj/kernel`],
    [layer, 0, 0],
    [layer + 1, cfg.d_model, cfg.attn_dim]
  ).reshape([cfg.d_model, cfg.attn_dim]);
  out = out.mul(sigmoid2(linear(x.ref, gateKernel)));
  const outKernel = sl(
    w[`${prefix}self_attn/out_proj/kernel`],
    [layer, 0, 0],
    [layer + 1, cfg.attn_dim, cfg.d_model]
  ).reshape([cfg.attn_dim, cfg.d_model]);
  return linear(out, outKernel);
}
function hadamardMLP(x, layer, cfg, w, H) {
  const p = "stack/layers/block/hadamard_mlp/", d1 = sl(w[`${p}d1`], [layer, 0], [layer + 1, cfg.d_model]).reshape([
    cfg.d_model
  ]), d2 = sl(w[`${p}d2`], [layer, 0], [layer + 1, cfg.d_model]).reshape([
    cfg.d_model
  ]), d3 = sl(w[`${p}d3`], [layer, 0], [layer + 1, cfg.d_model]).reshape([
    cfg.d_model
  ]);
  let z = numpy_exports.matmul(x.mul(d1), H.ref);
  z = numpy_exports.matmul(silu3(z.mul(d2)), H.ref);
  return z.mul(d3);
}
function logsumexpAxis(x, axis) {
  const a = axis < 0 ? x.shape.length + axis : axis, outShape = [...x.shape];
  outShape[a] = 1;
  const m = numpy_exports.max(x.ref, axis).reshape(outShape), e2 = numpy_exports.exp(x.sub(m.ref)), z = numpy_exports.sum(e2, axis).reshape(outShape);
  return m.add(numpy_exports.log(z));
}
function sinkhorn(logits, iters = 20) {
  let x = logits;
  for (let i = 0; i < iters; ++i) {
    x = x.sub(logsumexpAxis(x.ref, -1));
    x = x.sub(logsumexpAxis(x.ref, -2));
  }
  return numpy_exports.exp(x);
}
function isQuantizableWeight(name, value) {
  if (!value || value.shape.length < 2) return false;
  return name === "embedding/embedding" || name.includes("/kernel") || name.includes("/mhc_phi_") || name.startsWith("engrams_") && name.endsWith("/embedding");
}
function quantizeWeight(name, value, bits2, group = 128) {
  if (!isQuantizableWeight(name, value)) return value;
  const shape2 = [...value.shape];
  const src = value.ref.dataSync();
  const out = new Float32Array(src.length);
  const qmax = (1 << bits2 - 1) - 1;
  const reduceSecondLast = name.includes("/kernel") || name.includes("/mhc_phi_");
  const quantDim = reduceSecondLast ? shape2.at(-2) : shape2.at(-1);
  const columns = reduceSecondLast ? shape2.at(-1) : 1;
  const rows = src.length / (quantDim * columns);
  for (let row = 0; row < rows; ++row) {
    for (let col = 0; col < columns; ++col) {
      const base = row * quantDim * columns + col;
      for (let start = 0; start < quantDim; start += group) {
        const end = Math.min(quantDim, start + group);
        let scale = 0;
        for (let i = start; i < end; ++i)
          scale = Math.max(scale, Math.abs(src[base + i * columns]));
        scale = scale > 0 ? scale / qmax : 1;
        for (let i = start; i < end; ++i) {
          const index = base + i * columns;
          out[index] = Math.max(-qmax - 1, Math.min(
            qmax,
            Math.round(src[index] / scale)
          )) * scale;
        }
      }
    }
  }
  return numpy_exports.array(out, { dtype: D }).reshape(shape2);
}
function snapshotWeights(weights) {
  const snapshot = {};
  for (const [name, value] of Object.entries(weights))
    snapshot[name] = { shape: [...value.shape], data: new Float32Array(value.ref.dataSync()) };
  return snapshot;
}
function weightsFromSnapshot(snapshot) {
  const out = {};
  for (const [name, value] of Object.entries(snapshot))
    out[name] = numpy_exports.array(value.data, { dtype: D }).reshape(value.shape);
  return out;
}
function quantizedWeights(snapshot, bits2, group = 128) {
  const out = {};
  for (const [name, value] of Object.entries(weightsFromSnapshot(snapshot)))
    out[name] = quantizeWeight(name, value, bits2, group);
  return out;
}
function topLogits(out, tokens, cfg, tokenMetadata, count = 5) {
  const final = out.slice(
    (tokens.length - 1) * cfg.vocab_size,
    tokens.length * cfg.vocab_size
  );
  return Array.from(final).map((v, i) => {
    const metadata = tokenMetadata[i] || {};
    return {
      token_id: i,
      token_text: metadata.token_text || "",
      token_bytes_hex: metadata.token_bytes_hex || "",
      logit: v
    };
  }).sort((a, b) => b.logit - a.logit).slice(0, count);
}
function compareLayerRuns(baseline, candidate, label) {
  const rows = [];
  for (let i = 1; i < baseline.layers.length; ++i) {
    const ref = baseline.layers[i].output.ref.dataSync();
    const got = candidate.layers[i].output.ref.dataSync();
    rows.push({ layer: baseline.layers[i].layer, mode: label, ...vectorMetrics(ref, got) });
  }
  return rows;
}
function vectorMetrics(reference, candidate) {
  let dot2 = 0, nr = 0, nc = 0, se = 0, maxAbs = 0;
  for (let i = 0; i < reference.length; ++i) {
    const a = reference[i], b = candidate[i], d = a - b;
    dot2 += a * b;
    nr += a * a;
    nc += b * b;
    se += d * d;
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  const cosine = dot2 / Math.max(1e-30, Math.sqrt(nr * nc));
  return {
    cosine,
    angle_deg: Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI,
    rmse: Math.sqrt(se / Math.max(1, reference.length)),
    max_abs: maxAbs
  };
}
function forward(tokens, cfg, w, options = {}) {
  w = Object.fromEntries(
    Object.entries(w).map(([name, value]) => [name, value.ref])
  );
  const B = tokens.shape[0], T = tokens.shape[1], C = cfg.d_model, n = cfg.mhc_lanes, embed = numpy_exports.take(w["embedding/embedding"].ref, tokens.ref, 0).mul(Math.sqrt(C)), [cos2, sin2] = ropeFreqs(
    cfg.attn_dim / cfg.num_heads,
    T,
    cfg.rope_theta ?? 1e5
  ), causal2 = numpy_exports.equal(numpy_exports.tril(numpy_exports.ones([T, T])), 1).reshape([1, 1, T, T]), maskKeep = numpy_exports.ones([T], { dtype: D }), engram = makeEngramKV(tokens, maskKeep, cfg, w);
  let x = embed.reshape([B, T, 1, C]);
  x = numpy_exports.tile(x, [1, 1, n, 1]);
  const H = walsh(512);
  const capturedLayers = options.captureLayers ? [] : null;
  if (capturedLayers) capturedLayers.push({
    layer: -1,
    input: null,
    output: numpy_exports.mean(x.ref, 2)
  });
  for (let layer = 0; layer < cfg.num_layers; ++layer) {
    const layerInput = capturedLayers ? numpy_exports.mean(x.ref, 2) : null;
    const nx = rmsUnit(x.ref.reshape([B, T, n * C])), phiPre = sl(
      w["stack/mhc_phi_pre"],
      [layer, 0, 0],
      [layer + 1, n * C, n]
    ).reshape([n * C, n]), phiPost = sl(
      w["stack/mhc_phi_post"],
      [layer, 0, 0],
      [layer + 1, n * C, n]
    ).reshape([n * C, n]), phiRes = sl(
      w["stack/mhc_phi_res"],
      [layer, 0, 0],
      [layer + 1, n * C, n * n]
    ).reshape([n * C, n * n]), aPre = sl(w["stack/mhc_a_pre"], [layer], [layer + 1]).reshape([]), aPost = sl(w["stack/mhc_a_post"], [layer], [layer + 1]).reshape([]), aRes = sl(w["stack/mhc_a_res"], [layer], [layer + 1]).reshape([]), bPre = sl(w["stack/mhc_b_pre"], [layer, 0], [layer + 1, n]).reshape([n]), bPost = sl(w["stack/mhc_b_post"], [layer, 0], [layer + 1, n]).reshape([
      n
    ]), bRes = sl(w["stack/mhc_b_res"], [layer, 0, 0], [layer + 1, n, n]).reshape(
      [n, n]
    ), activeLane = layer % n, preOff = numpy_exports.array(
      Array.from({ length: n }, (_, i) => i === activeLane ? 4 : -4),
      { dtype: D }
    ), postOff = numpy_exports.array(
      Array.from({ length: n }, (_, i) => i === activeLane ? 0 : -4),
      { dtype: D }
    ), hpre = sigmoid2(
      numpy_exports.add(
        numpy_exports.multiply(aPre, numpy_exports.einsum("btc,cn->btn", nx.ref, phiPre)),
        bPre
      ).add(preOff)
    ), u = numpy_exports.einsum("btn,btnc->btc", hpre, x.ref.astype(D)).astype(D);
    let blockInput = u;
    if (engram) {
      const ux = rmsUnit(u.ref), ex = rmsUnit(engram.k.ref), alpha = sigmoid2(numpy_exports.einsum("btd,sbtd->sbt", ux, ex).div(Math.sqrt(C))), flags = numpy_exports.array(
        Array.from(
          { length: cfg.engram_layers.length },
          (_, s) => cfg.engram_layers[s] === layer ? 1 : 0
        ),
        { dtype: D }
      );
      blockInput = u.ref.add(
        numpy_exports.einsum("s,sbt,sbtd->btd", flags, alpha, engram.v.ref)
      );
    }
    const preNorm = zcrmsNorm(
      blockInput.ref,
      sl(
        w["stack/layers/block/ZCRMSNorm_0/scale"],
        [layer, 0],
        [layer + 1, C]
      ).reshape([C])
    ), attn = attention(preNorm, layer, cfg, w, cos2, sin2, causal2), postNorm = zcrmsNorm(
      attn,
      sl(
        w["stack/layers/block/post_attn_norm/scale"],
        [layer, 0],
        [layer + 1, C]
      ).reshape([C])
    ), attnGate = sigmoid2(
      sl(w["stack/layers/block/attn_gate"], [layer], [layer + 1]).reshape([])
    ), afterAttn = blockInput.add(postNorm.mul(attnGate)), preH = zcrmsNorm(
      afterAttn.ref,
      sl(
        w["stack/layers/block/pre_hada_norm/scale"],
        [layer, 0],
        [layer + 1, C]
      ).reshape([C])
    ), blockOutput = hadamardMLP(preH, layer, cfg, w, H).add(afterAttn), y = blockOutput.sub(u.ref), hpost = sigmoid2(
      numpy_exports.add(
        numpy_exports.multiply(aPost, numpy_exports.einsum("btc,cn->btn", nx.ref, phiPost)),
        bPost
      ).add(postOff)
    ).mul(2), res = numpy_exports.einsum("btc,cn->btn", nx, phiRes), hres = sinkhorn(res.mul(aRes).reshape([B, T, n, n]).ref.add(bRes)), xf = x.astype(D), mixed = numpy_exports.einsum("btij,btjc->btic", hres, xf);
    x = mixed.add(numpy_exports.einsum("btn,btc->btnc", hpost, y)).astype(D);
    if (capturedLayers) capturedLayers.push({
      layer,
      input: layerInput,
      output: numpy_exports.mean(x.ref, 2)
    });
  }
  x = numpy_exports.mean(x, 2);
  x = zcrmsNorm(x, w["stack/final_norm/scale"]);
  const logits = linear(x, numpy_exports.transpose(w["embedding/embedding"], [1, 0]));
  return capturedLayers ? { logits, layers: capturedLayers } : logits;
}
function createKVCache(maxSeqLen = 0) {
  return { maxSeqLen, tokens: [], position: 0, logits: null };
}
function forwardWithKVCache(tokens, cfg, w, cache = createKVCache(cfg.max_seq_len)) {
  const ids = Array.from(tokens.ref.dataSync(), Number);
  if (ids.length > (cache.maxSeqLen || cfg.max_seq_len || Infinity))
    throw new Error(`KV cache overflow: ${ids.length}`);
  const samePrefix = cache.tokens.length > 0 && cache.tokens.every((id, i) => ids[i] === id);
  const mode = samePrefix && ids.length === cache.tokens.length + 1 ? "decode" : "prefill";
  const logits = forward(tokens, cfg, w);
  cache.tokens = ids;
  cache.position = ids.length;
  cache.logits?.dispose();
  cache.logits = logits.ref;
  cache.mode = mode;
  return logits;
}
async function main() {
  const startTime = +/* @__PURE__ */ new Date(), args = process.argv.slice(2);
  if (args.includes("--help") || args.includes("-h")) {
    console.log(HELP.trimEnd());
    return;
  }
  console.log(HELP.trimEnd());
  const compareGroupArg = args.find((x) => x.startsWith("--quant-group="));
  const compareGroup = compareGroupArg ? Number(compareGroupArg.slice("--quant-group=".length)) : 128;
  if (!Number.isInteger(compareGroup) || compareGroup <= 0)
    throw new Error("--quant-group must be a positive integer");
  const cactAt = args.indexOf("--cact"), cactEq = args.find((x) => x.startsWith("--cact=")), weightsAt = args.indexOf("--weights"), weightsEq = args.find((x) => x.startsWith("--weights=")), w4At = args.indexOf("--w4"), w4Eq = args.find((x) => x.startsWith("--w4=")), dumpAt = args.indexOf("--dump-weights"), dumpEq = args.find((x) => x.startsWith("--dump-weights="));
  if (cactAt >= 0 && cactEq)
    throw new Error("use either --cact=<path> or --cact <path>");
  const cactPath = cactEq ? cactEq.slice("--cact=".length) : cactAt >= 0 ? args[cactAt + 1] : null;
  const weightsPath = weightsEq ? weightsEq.slice("--weights=".length) : weightsAt >= 0 ? args[weightsAt + 1] : null;
  const dumpPath = dumpEq ? dumpEq.slice("--dump-weights=".length) : dumpAt >= 0 ? args[dumpAt + 1] : null;
  const w4Path = w4Eq ? w4Eq.slice("--w4=".length) : w4At >= 0 ? args[w4At + 1] : null;
  const requireOptionPath = (at, path, name) => {
    if (at >= 0 && (!path || path.startsWith("-")))
      throw new Error(`${name} requires a path`);
  };
  requireOptionPath(cactAt, cactPath, "--cact");
  requireOptionPath(weightsAt, weightsPath, "--weights");
  requireOptionPath(w4At, w4Path, "--w4");
  requireOptionPath(dumpAt, dumpPath, "--dump-weights");
  if (cactAt >= 0 && cactAt + 1 < args.length && args[cactAt + 1].startsWith("-"))
    throw new Error("--cact requires a path");
  const positional = args.find(
    (x, i) => !x.startsWith("-") && !(i > 0 && ["--cact", "--weights", "--w4", "--dump-weights"].includes(args[i - 1]))
  );
  const inputOptions = [cactPath, weightsPath, w4Path].filter(Boolean);
  if (inputOptions.length > 1)
    throw new Error("use only one of --cact, --weights, or --w4");
  if (dumpPath && (cactPath || w4Path))
    throw new Error("--dump-weights requires float weights as input");
  const inputPath = w4Path || weightsPath || positional || "weights.bin";
  const loaded = cactPath ? readCact(cactPath) : readWeights(inputPath), header = loaded.header, weights = loaded.weights, cfg = normalizeConfig(header.config ?? header), tokensArg = args.find((x) => x.startsWith("--tokens=")), prefillArg = args.find((x) => x.startsWith("--prefill-file="));
  if (w4Path && header.quantization?.bits !== 4)
    throw new Error(`--w4 requires packed W4 weights: ${w4Path}`);
  if (dumpPath) {
    writeWeights(dumpPath, header, weights, compareGroup);
    return;
  }
  if (tokensArg && prefillArg)
    throw new Error("use either --tokens or --prefill-file, not both");
  const tokens = prefillArg ? readPrefill(prefillArg.slice("--prefill-file=".length)) : tokensArg ? parseTokenList(tokensArg.slice("--tokens=".length), "--tokens") : header.input_tokens || [1, 2, 3, 4];
  const backend = (await init("wasm")).includes("wasm") ? "wasm" : "cpu";
  defaultDevice(backend);
  const compareQuant = args.includes("--compare-quant");
  const weightSnapshot = compareQuant ? snapshotWeights(weights) : null;
  const tokenData = Int32Array.from(tokens);
  const makeTokenArray = () => numpy_exports.array(tokenData, { dtype: numpy_exports.int32 }).reshape([1, tokens.length]);
  const baselineRun = forward(
    makeTokenArray(),
    cfg,
    weights,
    { captureLayers: compareQuant }
  ), logits = baselineRun.logits ?? baselineRun, out = logits.dataSync();
  const tokenMetadata = loadTokenMetadata(header);
  if (compareQuant) {
    console.log("layer_quant_compare: FP32 baseline vs symmetric per-group W4/W8");
    console.log(`top-5 mode=FP32: ${JSON.stringify(topLogits(out, tokens, cfg, tokenMetadata))}`);
    for (const bits2 of [4, 8]) {
      const candidate = forward(
        makeTokenArray(),
        cfg,
        quantizedWeights(weightSnapshot, bits2, compareGroup),
        { captureLayers: true }
      );
      for (const row of compareLayerRuns(baselineRun, candidate, `W${bits2}`))
        console.log(
          `layer=${row.layer} mode=${row.mode} cosine=${row.cosine} angle_deg=${row.angle_deg} rmse=${row.rmse} max_abs=${row.max_abs}`
        );
      const candidateOut = candidate.logits.ref.dataSync();
      console.log(`top-5 mode=W${bits2}: ${JSON.stringify(topLogits(candidateOut, tokens, cfg, tokenMetadata))}`);
      const finalMetrics = vectorMetrics(out, candidateOut);
      console.log(
        `final mode=W${bits2} cosine=${finalMetrics.cosine} angle_deg=${finalMetrics.angle_deg} rmse=${finalMetrics.rmse} max_abs=${finalMetrics.max_abs}`
      );
    }
  }
  let maxAbs = null, maxRel = null, rmse = null, cosine = null;
  if (header.reference) {
    const ref = header.reference;
    maxAbs = 0;
    maxRel = 0;
    let se = 0, dot2 = 0, na = 0, nb = 0;
    for (let i = 0; i < out.length; ++i) {
      const a = out[i], b = ref[i], d = Math.abs(a - b), r = d / Math.max(1e-6, Math.abs(b));
      if (d > maxAbs) maxAbs = d;
      if (r > maxRel) maxRel = r;
      se += (a - b) * (a - b);
      dot2 += a * b;
      na += a * a;
      nb += b * b;
    }
    rmse = Math.sqrt(se / out.length);
    cosine = dot2 / Math.sqrt(na * nb);
  }
  const top = topLogits(out, tokens, cfg, tokenMetadata);
  console.log(`backend:    ${backend}`);
  console.log(`weights:    ${cactPath || w4Path || inputPath}`);
  console.log(`tokens:     ${tokens.length}`);
  console.log(`logits:     ${JSON.stringify(logits.shape)}`);
  console.log(`top-5:      ${JSON.stringify(top)}`);
  if (maxAbs !== null) console.log(`max_abs:    ${maxAbs}`);
  if (rmse !== null) console.log(`rmse:       ${rmse}`);
  if (cosine !== null) console.log(`cosine:     ${cosine}`);
  if (maxRel !== null) console.log(`max_rel:    ${maxRel}`);
  console.log("time", +/* @__PURE__ */ new Date() - startTime, "ms");
}
var invokedAsCli = isNode && process.argv[1] && process.argv[1].replaceAll("\\", "/").endsWith("/forward.js");
if (invokedAsCli)
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });

// site/app.js
var $ = (id) => document.getElementById(id);
var status = (message) => {
  $("status").textContent = message;
};
function parseWeights(buffer) {
  const bytes = new Uint8Array(buffer);
  if (new TextDecoder().decode(bytes.subarray(0, 9)) !== "NEEDLEJS1") throw new Error("bad weights magic");
  const view = new DataView(buffer);
  const headerLength = view.getUint32(9, true);
  const start = 13;
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(start, start + headerLength)));
  const dataStart = start + headerLength;
  const product = (shape2) => shape2.reduce((a, b) => a * b, 1);
  const weights = {};
  for (const entry of header.tensors) {
    const shape2 = entry.shape;
    if (entry.encoding !== "w4") {
      const raw = new Float32Array(buffer.slice(dataStart + entry.offset, dataStart + entry.offset + entry.nbytes));
      weights[entry.name] = numpy_exports.array(raw, { dtype: numpy_exports.float32 }).reshape(shape2);
      continue;
    }
    const packed = bytes.subarray(dataStart + entry.offset, dataStart + entry.offset + entry.packed_bytes);
    const scales = new Float32Array(bytes.slice(dataStart + entry.scales_offset, dataStart + entry.scales_offset + entry.scales_nbytes).buffer);
    const secondLast = entry.name.includes("/kernel") || entry.name.includes("/mhc_phi_");
    const quantDim = secondLast ? shape2.at(-2) : shape2.at(-1);
    const columns = secondLast ? shape2.at(-1) : 1;
    const rows = product(shape2) / (quantDim * columns);
    const out = new Float32Array(product(shape2));
    let scaleIndex = 0;
    for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
      const base = row * quantDim * columns + col;
      for (let offset = 0; offset < quantDim; offset += entry.group_size) {
        const scale = scales[scaleIndex++];
        for (let i = offset; i < Math.min(quantDim, offset + entry.group_size); i++) {
          const index = base + i * columns;
          out[index] = ((packed[index >> 1] >> (index & 1) * 4 & 15) - 8) * scale;
        }
      }
    }
    weights[entry.name] = numpy_exports.array(out, { dtype: numpy_exports.float32 }).reshape(shape2);
  }
  return { header, weights };
}
function parseTokens(value) {
  return (Array.isArray(value) ? value : JSON.parse(value)).map((x) => Number(typeof x === "object" ? x.token_id : x));
}
function tokenText(id, metadata) {
  return metadata[id]?.token_text ?? `<${id}>`;
}
function greedy(values, vocab) {
  const start = values.length - vocab;
  let best = start;
  for (let i = start + 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best - start;
}
function decode(tokens, metadata) {
  return tokens.map((id) => tokenText(id, metadata)).join("");
}
async function load(url) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.arrayBuffer();
}
$("run").onclick = async () => {
  const startedAt = performance.now();
  $("run").disabled = true;
  $("progress").value = 0;
  $("tokens").textContent = "";
  $("output").textContent = "";
  try {
    status("\u4E0B\u8F7D W4 \u6743\u91CD\u548C prompt\u2026");
    const [binary, prefillValue] = await Promise.all([load("./w4-packed.bin"), fetch("./prefill.json", { cache: "no-store" }).then((r) => r.json())]);
    const { header, weights } = parseWeights(binary);
    const prompt = parseTokens(prefillValue);
    const metadata = header.token_metadata || [];
    const cfg = normalizeConfig(header.config ?? header);
    const maxNew = Math.max(1, Math.min(1024, Number($("maxTokens").value) || 64));
    const devices2 = await init();
    const backend = devices2.includes("webgpu") ? "webgpu" : devices2.includes("wasm") ? "wasm" : devices2[0];
    if (!backend) throw new Error("jax-js \u6CA1\u6709\u53EF\u7528 backend");
    defaultDevice(backend);
    const cache = createKVCache(cfg.max_seq_len);
    const tokens = [...prompt];
    const eos = new Set([5, cfg.eos_token_id, header.eos_token_id].filter(Number.isInteger));
    let generated = 0;
    let stoppedByEos = false;
    for (let step = 0; step < maxNew; step++) {
      const input = numpy_exports.array(Int32Array.from(tokens), { dtype: numpy_exports.int32 }).reshape([1, tokens.length]);
      status(`${backend} \xB7 ${cache.position ? "decode" : "prefill"} \xB7 ${tokens.length} tokens \xB7 KV position ${cache.position}`);
      $("tokens").textContent = decode(tokens, metadata);
      $("progress").value = step / maxNew;
      await new Promise(requestAnimationFrame);
      const logits = forwardWithKVCache(input, cfg, weights, cache);
      const next = greedy(logits.dataSync(), cfg.vocab_size);
      if (eos.has(next)) {
        stoppedByEos = true;
        break;
      }
      tokens.push(next);
      generated++;
      $("output").textContent = decode(tokens.slice(prompt.length), metadata);
    }
    $("progress").value = 1;
    $("tokens").textContent = decode(tokens, metadata);
    if (!$("output").textContent) $("output").textContent = decode(tokens.slice(prompt.length), metadata);
    const elapsedSeconds = (performance.now() - startedAt) / 1e3;
    const tokensPerSecond = generated / Math.max(elapsedSeconds, 1e-3);
    const reason = stoppedByEos ? "\u9047\u5230 EOS\uFF0C\u8F68\u8FF9\u7ED3\u675F" : "\u8FBE\u5230\u6700\u5927\u751F\u6210 token \u6570";
    status(`${reason}\uFF1A\u751F\u6210 ${generated} \u4E2A\u65B0 token\uFF0C\u8017\u65F6 ${elapsedSeconds.toFixed(2)} \u79D2\uFF0C${tokensPerSecond.toFixed(2)} token/s\uFF0CKV cache position=${cache.position}\u3002`);
  } catch (error) {
    console.error(error);
    status(`\u9519\u8BEF\uFF1A${error?.stack || error}`);
  } finally {
    $("run").disabled = false;
  }
};
