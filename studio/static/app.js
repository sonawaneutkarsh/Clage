import { colorFor, layoutNetwork, matchedFrame, metricsCSV } from './math.js';

const $ = identity => document.getElementById(identity);
const actions = ['MOVE', 'TURN LEFT', 'TURN RIGHT', 'EAT'];
const observations = ['Food Δx', 'Food Δy', 'Food density', 'Body density', 'Energy', 'Wall x', 'Wall y', 'Prev MOVE', 'Prev EAT'];
let state = {}, genomes = {}, evolutionLineage = {}, bundle = null, comparison = null, selected = null;
let replayIndex = 0, replayPlaying = false, grid = false, following = false, view = 'ecosystem';
let trail = [], localFrames = [], lastSequence = -1, lastGeneration = -1;
let socket, reconnectTimer, chartDirty = true, events = [], previousBodies = new Map();
let drawingMilliseconds = 0, drawingSamples = 0;
const camera = { zoom: 1, x: 0, y: 0 };
const graphCamera = { x: 0, y: 0, zoom: 1 };
const canvas = $('world'), context = canvas.getContext('2d');
const emptyInspector = $('organism').cloneNode(true);
let transform = { cell: 1, ox: 0, oy: 0 }, fps = 0, countFrames = 0, frameClock = performance.now(), playbackClock = performance.now();

const fieldDefinitions = [
  ['population', 'Founder genomes', 72, 1, 2000, 1, 'Bodies can reproduce within a generation.'],
  ['generations', 'Generations', 8, 1, 50, 1, 'A fresh shared world for every evaluation.'],
  ['ticks', 'Ticks per generation', 320, 1, 3000, 1, 'Not render frames.'],
  ['width', 'World width', 40, 2, 128, 1, 'Walled grid, measured in cells.'],
  ['height', 'World height', 32, 2, 128, 1, 'No wraparound.'],
  ['food', 'Initial / target food', 180, 0, 16384, 1, 'Founders + food must fit.'],
  ['regrowth', 'Food regrowth per tick', 2, 0, 100, 1, 'Grows toward the target, if space exists.'],
  ['metabolism', 'Metabolism per tick', .004, 0, 1, .001, 'Energy cost; zero is permitted.'],
  ['repro_threshold', 'Reproduction threshold', .85, 0, 2, .05, 'Initial/max energy is 1.'],
  ['repro_fraction', 'Energy given to child', .5, 0, 1, .05, 'Asexual, same genome.'],
  ['seed', 'Random seed', 42, 0, 2147483647, 1, 'Engine + derived world seeds.'],
  ['weight_mutation', 'Weight mutation probability', .8, 0, 1, .05, 'Unchanged engine operator.'],
  ['add_node', 'Add-node probability', .03, 0, 1, .01, 'Splits an enabled connection.'],
  ['add_connection', 'Add-connection probability', .08, 0, 1, .01, 'Feed-forward connections only.'],
  ['compatibility', 'Species distance threshold', 3, .01, 100, .01, 'NEAT compatibility distance.']
];

function element(tag, attributes = {}, text = null) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== null) node.textContent = text;
  return node;
}

function svgElement(tag, attributes, text = null) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== null) node.textContent = text;
  return node;
}

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => $('toast').hidden = true, 6000);
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) {
    const message = Array.isArray(data.detail) ? data.detail.map(error => `${error.loc?.slice(1).join('.')}: ${error.msg}`).join('\n') : data.detail;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return data;
}

async function safely(operation) {
  try { return await operation(); } catch (error) { toast(error.message); return null; }
}

function download(data, filename, type = 'application/json') {
  const url = URL.createObjectURL(data instanceof Blob ? data : new Blob([data], { type }));
  const link = element('a', { href: url, download: filename });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function currentFrame() { return bundle ? bundle.frames[replayIndex] : state.frame; }
function currentConfig() { return bundle ? bundle.config : state.config; }
function currentGenomes() { return bundle ? bundle.genomes : genomes; }
function currentSpecies() { return bundle ? bundle.species || {} : state.species || {}; }
function framesForChart() { return bundle ? bundle.frames.slice(0, replayIndex + 1) : localFrames; }

function acceptState(next) {
  if (next.heartbeat) return;
  const runChanged = next.run_id !== state.run_id;
  const historyChanged = (next.history?.length || 0) !== (state.history?.length || 0);
  state = next;
  if (!bundle && Number.isFinite(next.speed) && document.activeElement !== $('speed')) $('speed').value = next.speed;
  if (runChanged) {
    $('network-detail').textContent = 'Select a node or connection to view its recorded metadata.';
    localFrames = []; lastSequence = -1; lastGeneration = -1;
    if (!bundle) { selected = null; following = false; trail = []; previousBodies.clear(); events = []; }
  }
  if (next.error) toast(next.error);
  if (historyChanged && !bundle) safely(loadGenomes);
  if (!bundle && next.frame) {
    const frame = next.frame;
    const changed = frame.sequence !== lastSequence;
    if (frame.sequence !== lastSequence) {
      if (frame.generation !== lastGeneration) {
        selected = null; following = false; trail = []; previousBodies.clear();
        lastGeneration = frame.generation;
        safely(loadGenomes);
      }
      if (frame.sequence < lastSequence) localFrames = [];
      localFrames.push(frame);
      if (localFrames.length > 600) localFrames.shift();
      lastSequence = frame.sequence;
      processEvents(frame);
      chartDirty = true;
    }
    updateUI(changed);
  }
}

function processEvents(frame) {
  const now = performance.now();
  for (const body of frame.organisms) {
    const previous = previousBodies.get(body.id);
    if (!previous && body.parent !== null) events.push({ x: body.x, y: body.y, color: '#a6eccb', time: now });
    if (previous?.alive && !body.alive) events.push({ x: body.x, y: body.y, color: '#e98585', time: now });
    if (previous && body.food_eaten > previous.food_eaten) events.push({ x: body.x, y: body.y, color: '#e7af7d', time: now });
    previousBodies.set(body.id, body);
    if (body.id === selected) {
      trail.push([body.x, body.y]);
      if (trail.length > 60) trail.shift();
    }
  }
  if (events.length > 200) events = events.slice(-200);
}

async function loadGenomes() {
  [genomes, evolutionLineage] = await Promise.all([api('/api/genomes'), api('/api/lineage')]);
  populateGenomeChoices();
  updateInspector();
  drawNetworks();
  if (view === 'evolution') drawHistory();
}

function connect() {
  clearTimeout(reconnectTimer);
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/stream`);
  socket.onopen = () => { $('connection').textContent = 'Local engine connected'; $('connection-dot').style.background = 'var(--accent)'; };
  socket.onmessage = message => acceptState(JSON.parse(message.data));
  socket.onclose = () => {
    $('connection').textContent = 'Disconnected · reconnecting';
    $('connection-dot').style.background = 'var(--red)';
    reconnectTimer = setTimeout(connect, 1500);
  };
  socket.onerror = () => $('connection').textContent = 'Connection unavailable';
}

function updateUI(refreshInspectors = true) {
  const frame = currentFrame(), config = currentConfig();
  if (!frame) return;
  $('population').textContent = frame.metrics.population.toLocaleString();
  $('birth-death').textContent = `${frame.metrics.births} births · ${frame.metrics.deaths} deaths`;
  $('food').textContent = frame.metrics.food;
  $('consumed').textContent = `${frame.metrics.food_eaten} consumed this generation`;
  $('energy').textContent = frame.metrics.mean_energy.toFixed(3);
  $('generation').textContent = String(frame.generation).padStart(2, '0');
  $('tick').textContent = `tick ${frame.tick} / ${config.ticks}`;
  $('dimensions').textContent = `${config.width} × ${config.height}`;
  $('world-seed').textContent = `SEED ${config.seed} · ${config.initialization.toUpperCase()}`;
  const playing = bundle ? replayPlaying : !state.paused;
  $('play').textContent = playing ? 'Ⅱ Pause' : (bundle ? '▶ Play replay' : state.complete ? '✓ Complete' : '▶ Resume');
  $('play').disabled = !bundle && (state.complete || Boolean(state.error));
  $('step').disabled = !bundle && (state.complete || Boolean(state.error));
  $('mode').textContent = bundle ? `REPLAY · ${playing ? 'PLAYING' : 'PAUSED'}` : state.complete ? 'LIVE · COMPLETE' : `LIVE · ${state.paused ? 'PAUSED' : 'RUNNING'}`;
  if (!bundle && state.error) $('mode').textContent = 'LIVE · ERROR';
  $('engine-error').hidden = !state.error || Boolean(bundle);
  $('engine-error').textContent = state.error ? `${state.error} Reset the run to recover.` : '';
  $('sim-rate').textContent = bundle ? 'Recorded playback · engine unchanged' : `${state.paused ? 0 : state.achieved_tps || 0} engine ticks/s`;
  $('recording-window').textContent = bundle ? `${bundle.frames.length} frames · ${bundle.dropped_frames || 0} dropped` : `${state.recording?.frames || 0} recorded · ${state.recording?.dropped || 0} dropped`;
  $('speed-label').textContent = `${$('speed').value} ${bundle ? 'frames/s' : 'ticks/s'}`;
  $('live').hidden = !bundle;
  $('review').hidden = Boolean(bundle);
  $('timeline').hidden = !bundle;
  if (bundle) {
    $('scrub').value = replayIndex;
    $('replay-position').textContent = `Frame ${replayIndex + 1}/${bundle.frames.length} · G${frame.generation}:T${frame.tick}`;
  }
  if (refreshInspectors) {
    updateInspector();
    if (view === 'neural') drawNetworks();
    if (view === 'evolution') drawHistory();
  }
  rememberNavigation(false);
}

function updateInspector() {
  const frame = currentFrame();
  const body = frame?.organisms.find(organism => organism.id === selected);
  const choices = frame?.organisms || [];
  const signature = `${frame?.generation}:${choices.map(item => item.id).join(',')}`;
  if (bodyChoice.dataset.signature !== signature) {
    bodyChoice.replaceChildren(element('option', { value: '' }, 'Select an organism…'),
      ...choices.map(item => element('option', { value: item.id }, `Body ${item.id}`)));
    bodyChoice.dataset.signature = signature;
  }
  bodyChoice.value = body ? String(body.id) : '';
  $('follow').setAttribute('aria-pressed', following);
  $('follow').textContent = following ? 'Following ✓' : 'Follow ↗';
  if (!body) {
    $('organism').replaceChildren(...[...emptyInspector.childNodes].map(node => node.cloneNode(true)));
    renderNetwork($('mini-network'), null, null, 420, 330, false);
    $('decision-label').textContent = '';
    return;
  }
  const host = $('organism');
  host.replaceChildren();
  const title = element('div', { class: 'organism-title' });
  title.append(element('h2', {}, `Organism ${body.id}`), element('span', {}, body.alive ? '● ALIVE' : '○ DEAD'));
  host.append(title);
  const data = element('dl', { class: 'body-data' });
  const final = frame.tick === currentConfig().ticks;
  const genome = currentGenomes()[body.genome];
  const values = [
    ['Genome', body.genome], ['Species', currentSpecies()[body.genome] ?? 'Not evaluated'],
    ['Position', `${body.x}, ${body.y}`], ['Facing', ['North', 'East', 'South', 'West'][body.facing]],
    ['Energy', body.energy.toFixed(4)], ['Age', `${body.age} ticks`],
    ['Last action', actions[body.action] || 'Not acted'], ['Consumed', body.food_eaten],
    ['Offspring', body.offspring], ['Parent body', body.parent ?? 'Founder'],
    ['Body score', `${body.fitness.toFixed(2)} · ${final ? 'final' : 'provisional'}`], ['Generation', frame.generation],
    ['Genome final score', genome?.fitness == null ? 'Not evaluated' : genome.fitness.toFixed(3)],
    ['Evaluation phase', final ? 'Completed world' : 'World in progress']
  ];
  for (const [label, value] of values) {
    const pair = element('div'); pair.append(element('dt', {}, label), element('dd', {}, value)); data.append(pair);
  }
  host.append(data);
  const bar = element('div', { class: 'energy-bar' }), fill = element('span');
  fill.style.width = `${Math.max(0, Math.min(1, body.energy)) * 100}%`;
  bar.append(fill); host.append(bar);
  renderNetwork($('mini-network'), genome, body.inference, 420, 330, false);
  $('decision-label').textContent = body.action === null ? 'No action has been taken yet.' : `Selected: ${actions[body.action]} · pre-action observations at world tick ${body.inference?.world_tick ?? 'not recorded'}`;
}

function resizeCanvas(target) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = target.clientWidth, height = target.clientHeight;
  if (target.width !== Math.round(width * ratio) || target.height !== Math.round(height * ratio)) {
    target.width = Math.round(width * ratio); target.height = Math.round(height * ratio);
  }
  const targetContext = target.getContext('2d');
  targetContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context: targetContext, width, height };
}

function drawWorld(target, frame, config, useCamera = true) {
  const { context: drawing, width, height } = resizeCanvas(target);
  drawing.clearRect(0, 0, width, height);
  if (!frame || !config || width <= 75 || height <= 65) return;
  const cell = Math.min((width - 75) / config.width, (height - 65) / config.height) * (useCamera ? camera.zoom : 1);
  const focus = following && useCamera ? frame.organisms.find(body => body.id === selected) : null;
  const ox = width / 2 - (focus ? focus.x + .5 : config.width / 2) * cell + (useCamera && !focus ? camera.x : 0);
  const oy = height / 2 - (focus ? focus.y + .5 : config.height / 2) * cell + (useCamera && !focus ? camera.y : 0);
  if (useCamera) transform = { cell, ox, oy };
  drawing.save();
  drawing.translate(ox, oy);
  drawing.fillStyle = '#0d191e';
  drawing.fillRect(0, 0, config.width * cell, config.height * cell);
  drawing.strokeStyle = '#355058'; drawing.lineWidth = 1;
  drawing.strokeRect(0, 0, config.width * cell, config.height * cell);
  drawing.beginPath(); drawing.rect(0, 0, config.width * cell, config.height * cell); drawing.clip();
  const layer = $('layer').value;
  if (['food', 'density'].includes(layer)) {
    const occupancy = new Map();
    const sources = layer === 'food' ? frame.food : frame.organisms.filter(body => body.alive).map(body => [body.x, body.y]);
    for (const [sx, sy] of sources) {
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const key = `${sx + dx},${sy + dy}`; occupancy.set(key, (occupancy.get(key) || 0) + 1);
      }
    }
    for (const [key, count] of occupancy) {
      const [x, y] = key.split(',').map(Number);
      drawing.fillStyle = `rgba(${layer === 'food' ? '108,199,155' : '123,163,209'},${Math.min(.65, count / 20)})`;
      drawing.fillRect(x * cell, y * cell, cell, cell);
    }
  }
  if (grid && cell > 5) {
    drawing.beginPath(); drawing.strokeStyle = '#253a4070'; drawing.lineWidth = .5;
    for (let x = 1; x < config.width; x++) { drawing.moveTo(x * cell, 0); drawing.lineTo(x * cell, config.height * cell); }
    for (let y = 1; y < config.height; y++) { drawing.moveTo(0, y * cell); drawing.lineTo(config.width * cell, y * cell); }
    drawing.stroke();
  }
  drawing.fillStyle = '#7bc49d';
  for (const [x, y] of frame.food) {
    const px = (x + .5) * cell, py = (y + .5) * cell, radius = Math.max(1.4, cell * .15);
    drawing.beginPath(); drawing.moveTo(px, py - radius); drawing.lineTo(px + radius, py);
    drawing.lineTo(px, py + radius); drawing.lineTo(px - radius, py); drawing.closePath(); drawing.fill();
  }
  if (useCamera && trail.length > 1) {
    drawing.beginPath(); drawing.strokeStyle = '#d8ede955'; drawing.lineWidth = 1.5;
    trail.forEach(([x, y], index) => index ? drawing.lineTo((x + .5) * cell, (y + .5) * cell) : drawing.moveTo((x + .5) * cell, (y + .5) * cell));
    drawing.stroke();
  }
  const bodyMap = layer === 'lineage' ? new Map(frame.organisms.map(body => [body.id, body])) : null;
  for (const body of frame.organisms) {
    if (!body.alive) continue;
    const px = (body.x + .5) * cell, py = (body.y + .5) * cell;
    if (px + ox < -cell || px + ox > width + cell || py + oy < -cell || py + oy > height + cell) continue;
    let identity = body.genome;
    if (layer === 'species') identity = currentSpecies()[body.genome] ?? 'unknown';
    if (layer === 'lineage') {
      let ancestor = body, steps = 0;
      while (ancestor.parent !== null && bodyMap.has(ancestor.parent) && steps++ < frame.organisms.length) ancestor = bodyMap.get(ancestor.parent);
      identity = ancestor.id;
    }
    drawing.fillStyle = layer === 'energy' ? `hsl(${Math.max(0, Math.min(1, body.energy)) * 140} 48% 64%)` : layer === 'species' && identity === 'unknown' ? '#71838b' : colorFor(identity);
    drawing.save(); drawing.translate(px, py); drawing.rotate(body.facing * Math.PI / 2);
    const radius = Math.max(2.1, cell * .33);
    drawing.beginPath(); drawing.moveTo(0, -radius * 1.3); drawing.lineTo(radius, radius); drawing.quadraticCurveTo(0, radius * .55, -radius, radius); drawing.closePath(); drawing.fill();
    drawing.restore();
    if (body.id === selected && useCamera) {
      drawing.beginPath(); drawing.arc(px, py, Math.max(5, cell * .65), 0, Math.PI * 2);
      drawing.strokeStyle = '#e8fff4'; drawing.lineWidth = 1.5; drawing.stroke();
    }
  }
  if (useCamera && !bundle) {
    const now = performance.now(); events = events.filter(event => now - event.time < 850);
    for (const event of events) {
      const progress = Math.max(0, Math.min(1, (now - event.time) / 850));
      drawing.globalAlpha = 1 - progress; drawing.strokeStyle = event.color; drawing.lineWidth = 1;
      drawing.beginPath(); drawing.arc((event.x + .5) * cell, (event.y + .5) * cell, cell * (.4 + progress), 0, Math.PI * 2); drawing.stroke();
    }
  }
  drawing.restore();
}

function pickBody(event) {
  const frame = currentFrame();
  if (!frame) return null;
  const bounds = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - bounds.left - transform.ox) / transform.cell);
  const y = Math.floor((event.clientY - bounds.top - transform.oy) / transform.cell);
  return frame.organisms.find(body => body.alive && body.x === x && body.y === y);
}

let dragging = null;
canvas.addEventListener('pointerdown', event => {
  dragging = { x: event.clientX, y: event.clientY, cx: camera.x, cy: camera.y, moved: false };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (dragging) {
    const dx = event.clientX - dragging.x, dy = event.clientY - dragging.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) dragging.moved = true;
    if (dragging.moved) { following = false; camera.x = dragging.cx + dx; camera.y = dragging.cy + dy; }
    return;
  }
  const body = pickBody(event);
  $('hover').hidden = !body;
  if (body) $('hover').textContent = `Organism ${body.id} · energy ${body.energy.toFixed(3)} · ${actions[body.action] || 'not acted'}`;
});
canvas.addEventListener('pointerup', event => {
  if (dragging && !dragging.moved) {
    const body = pickBody(event);
    if (body) selectBody(body.id);
  }
  dragging = null;
  rememberNavigation();
});
canvas.addEventListener('pointercancel', () => dragging = null);
canvas.addEventListener('pointerleave', () => $('hover').hidden = true);
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  camera.zoom = Math.max(.35, Math.min(10, camera.zoom * (event.deltaY < 0 ? 1.12 : .89)));
  rememberNavigation();
}, { passive: false });

function selectBody(identity) {
  if (identity !== selected) navigate(view, selected === null ? (view === 'ecosystem' ? 'ecosystem' : 'body lineage') : `organism ${selected}`);
  selected = identity; trail = [];
  const body = currentFrame()?.organisms.find(item => item.id === selected);
  if (body) $('genome-choice').value = body.genome;
  updateInspector(); drawNetworks(); drawHistory();
  rememberNavigation();
}

function fitWorld() { camera.zoom = 1; camera.x = 0; camera.y = 0; following = false; updateUI(); }

function renderNetwork(target, genome, inference, width, height, detailed = true) {
  target.replaceChildren();
  if (!genome) {
    target.append(svgElement('text', { x: width / 2, y: height / 2, fill: '#6f8a94', 'text-anchor': 'middle', 'font-size': detailed ? 18 : 13 }, 'Select an organism or genome'));
    return;
  }
  const positions = layoutNetwork(genome, width, height), group = svgElement('g', {});
  target.append(group);
  for (const edge of genome.connections) {
    const start = positions.get(edge.in), end = positions.get(edge.out);
    if (!start || !end) continue;
    const link = svgElement('path', { d: `M${start.x},${start.y} C${(start.x + end.x) / 2},${start.y} ${(start.x + end.x) / 2},${end.y} ${end.x},${end.y}`,
      fill: 'none', stroke: edge.weight >= 0 ? '#87cdb0' : '#dd9d76', 'stroke-width': Math.min(4, .5 + Math.abs(edge.weight)),
      opacity: edge.enabled ? .46 : .2, 'stroke-dasharray': edge.enabled ? '' : '4 4', class: 'edge' });
    link.append(svgElement('title', {}, `${edge.in} → ${edge.out} · weight ${edge.weight.toFixed(4)} · innovation ${edge.innovation} · ${edge.enabled ? 'enabled' : 'disabled'}`));
    const inspectEdge = () => inspectNetwork({ kind: 'Connection', ...edge,
      source_activation: inference?.values?.[edge.in] ?? 'Not recorded',
      inference_tick: inference?.world_tick ?? 'Not recorded',
      weighted_contribution: inference ? (edge.enabled ? inference.values[edge.in] * edge.weight : 0) : 'Not recorded' });
    if (detailed) makeInspectable(link, `Inspect connection ${edge.in} to ${edge.out}`, inspectEdge, target.id);
    group.append(link);
  }
  const inputs = genome.nodes.filter(node => node.type === 'INPUT').sort((first, second) => first.id - second.id);
  const outputs = genome.nodes.filter(node => node.type === 'OUTPUT').sort((first, second) => first.id - second.id);
  for (const node of genome.nodes) {
    const position = positions.get(node.id), value = inference?.values?.[node.id];
    const color = value === undefined ? '#233940' : value >= 0 ? `hsl(151 35% ${23 + Math.min(1, value) * 48}%)` : `hsl(25 45% ${23 + Math.min(1, -value) * 44}%)`;
    const chosen = node.type === 'OUTPUT' && inference && outputs.indexOf(node) === inference.outputs.indexOf(Math.max(...inference.outputs));
    const circle = svgElement('circle', { cx: position.x, cy: position.y, r: detailed ? 14 : 9, fill: color, stroke: chosen ? '#d5ffe9' : '#8fb4b8', 'stroke-width': chosen ? 3 : 1, class: 'node' });
    const detail = { kind: 'Node', ...node, activation: value ?? 'Not recorded', inference_tick: inference?.world_tick ?? 'Not recorded' };
    circle.append(svgElement('title', {}, JSON.stringify(detail)));
    if (detailed) makeInspectable(circle, `Inspect ${node.type.toLowerCase()} node ${node.id}`, () => inspectNetwork(detail), target.id);
    group.append(circle);
    let label = String(node.id);
    if (node.type === 'INPUT') label = observations[inputs.indexOf(node)] || label;
    if (node.type === 'OUTPUT') label = actions[outputs.indexOf(node)] || label;
    const left = node.type === 'INPUT', right = node.type === 'OUTPUT';
    if (detailed) {
      group.append(svgElement('text', { x: position.x + (left ? -23 : right ? 23 : 0), y: position.y + (left || right ? 4 : -23), fill: '#a4b9c0', 'font-size': 12, 'text-anchor': left ? 'end' : right ? 'start' : 'middle' }, label));
      if (value !== undefined) group.append(svgElement('text', { x: position.x, y: position.y + 30, fill: '#79999f', 'font-size': 9, 'text-anchor': 'middle' }, value.toFixed(3)));
    }
  }
  if (target.id === 'network') group.setAttribute('transform', `translate(${graphCamera.x},${graphCamera.y}) scale(${graphCamera.zoom})`);
}

function populateGenomeChoices() {
  for (const identity of ['genome-choice', 'genome-compare']) {
    const control = $(identity), value = control.value;
    control.replaceChildren();
    if (identity === 'genome-compare') control.append(element('option', { value: '' }, 'No comparison'));
    for (const [key, genome] of Object.entries(currentGenomes())) control.append(element('option', { value: key }, `G${key} · ${genome.nodes.length} nodes · ${genome.connections.length} genes`));
    if (currentGenomes()[value]) control.value = value;
  }
}

function drawNetworks() {
  const focused = document.activeElement?.getAttribute('data-inspection');
  const body = currentFrame()?.organisms.find(item => item.id === selected);
  const key = $('genome-choice').value || body?.genome;
  const genome = currentGenomes()[key];
  $('network-title').textContent = genome ? `GENOME ${key} · ${genome.nodes.length} NODES · ${genome.connections.length} CONNECTIONS` : 'SELECT A GENOME';
  renderNetwork($('network'), genome, body && body.genome === key ? body.inference : null, 800, 620);
  const compareKey = $('genome-compare').value;
  $('network-compare-card').hidden = !compareKey;
  if (compareKey) renderNetwork($('network-compare'), currentGenomes()[compareKey], null, 800, 620);
  if (focused) document.querySelector(`[data-inspection="${focused}"]`)?.focus({ preventScroll: true });
}

function makeInspectable(target, label, inspect, graph) {
  target.setAttribute('tabindex', '0'); target.setAttribute('role', 'button');
  target.setAttribute('aria-label', label);
  target.setAttribute('data-inspection', `${graph}-${label}`);
  target.onclick = inspect;
  target.onkeydown = event => {
    if (event.key === 'Enter' || event.code === 'Space') { event.preventDefault(); event.stopPropagation(); inspect(); }
  };
}

function inspectNetwork(detail) {
  if (!navigation.detail) navigate(view, 'network');
  navigation.detail = true;
  $('network-detail').textContent = JSON.stringify(detail, null, 2);
  $('network-detail').focus({ preventScroll: true });
  rememberNavigation();
}

let graphDrag = null;
$('network').addEventListener('wheel', event => {
  event.preventDefault(); graphCamera.zoom = Math.max(.4, Math.min(4, graphCamera.zoom * (event.deltaY < 0 ? 1.1 : .9))); drawNetworks(); rememberNavigation();
}, { passive: false });
$('network').addEventListener('pointerdown', event => {
  if (event.target.closest('.node, .edge')) return;
  graphDrag = { x: event.clientX, y: event.clientY, cx: graphCamera.x, cy: graphCamera.y };
  $('network').setPointerCapture(event.pointerId);
});
$('network').addEventListener('pointermove', event => {
  if (!graphDrag) return;
  const scale = 800 / $('network').clientWidth;
  graphCamera.x = graphDrag.cx + (event.clientX - graphDrag.x) * scale;
  graphCamera.y = graphDrag.cy + (event.clientY - graphDrag.y) * scale;
  const group = $('network').firstElementChild;
  group?.setAttribute('transform', `translate(${graphCamera.x},${graphCamera.y}) scale(${graphCamera.zoom})`);
});
$('network').addEventListener('pointerup', () => { graphDrag = null; rememberNavigation(); });
$('network').addEventListener('pointercancel', () => graphDrag = null);

function drawChart() {
  const target = $('chart'), { context: drawing, width, height } = resizeCanvas(target);
  drawing.clearRect(0, 0, width, height);
  const frames = framesForChart().slice(-Number($('chart-range').value)), metric = $('metric').value;
  if (!frames.length) return;
  const values = frames.map(frame => frame.metrics[metric]);
  const compareFrames = comparison ? frames.map(frame => matchedFrame(comparison.frames, frame)) : [];
  const compareValues = compareFrames.map(frame => frame?.metrics?.[metric]);
  const maxValue = Math.max(1, ...values, ...compareValues.filter(value => value !== undefined));
  const left = 48, right = width - 20, top = 16, bottom = height - 30;
  drawing.font = '9px -apple-system, sans-serif';
  for (let index = 0; index <= 3; index++) {
    const y = top + (bottom - top) * index / 3;
    drawing.beginPath(); drawing.moveTo(left, y); drawing.lineTo(right, y); drawing.strokeStyle = '#273a41'; drawing.lineWidth = .5; drawing.stroke();
    drawing.fillStyle = '#72909a'; drawing.fillText((maxValue * (1 - index / 3)).toFixed(maxValue > 10 ? 0 : 2), 10, y + 3);
  }
  function trace(series, color, fill = false) {
    drawing.beginPath(); let begun = false, last = null;
    series.forEach((value, index) => {
      if (value === undefined) { begun = false; return; }
      const x = left + index / Math.max(1, series.length - 1) * (right - left), y = bottom - value / maxValue * (bottom - top);
      const sameGeneration = !last || frames[index].generation === last.generation;
      if (!begun || !sameGeneration) drawing.moveTo(x, y); else drawing.lineTo(x, y);
      begun = true; last = frames[index];
    });
    drawing.strokeStyle = color; drawing.lineWidth = 1.8; drawing.stroke();
  }
  trace(values, '#a6eccb');
  if (comparison) trace(compareValues, '#e7af7d');
  drawing.fillStyle = '#72909a';
  drawing.fillText(`G${frames[0].generation}:T${frames[0].tick}`, left, height - 10);
  const last = frames.at(-1);
  drawing.textAlign = 'right'; drawing.fillText(`G${last.generation}:T${last.tick}`, right, height - 10); drawing.textAlign = 'left';
  $('chart-note').textContent = bundle ? 'Actual recorded frames; lines break at generation boundaries. No interpolation of missing ticks.' : 'Live chart uses received snapshots (≤10 Hz), not every engine tick. Generation counters reset per world.';
}

function drawDistribution() {
  const target = $('distribution-chart'), { context: drawing, width, height } = resizeCanvas(target);
  drawing.clearRect(0, 0, width, height);
  const frame = currentFrame(); if (!frame) return;
  const mode = $('distribution').value;
  let counts, labels, note;
  if (mode === 'actions') {
    counts = frame.metrics.actions; labels = ['Move', 'Left', 'Right', 'Eat'];
    note = 'Last action of living bodies that have acted; not cumulative action frequency.';
  } else if (mode === 'fitness') {
    const history = bundle ? bundle.history : state.history || [];
    const row = history.at(-1);
    if (!row) { drawing.fillStyle = '#829ba4'; drawing.font = '11px sans-serif'; drawing.fillText('Complete a generation first', 20, 55); return; }
    const upper = Math.max(1, ...row.fitnesses); counts = Array(8).fill(0);
    for (const value of row.fitnesses) counts[Math.min(7, Math.floor(value / upper * 8))]++;
    labels = counts.map((_, index) => (index * upper / 8).toFixed(1));
    note = `Evaluated founder genomes, world G${row.world_generation}. Range 0–${upper.toFixed(2)}. Max-body score, not learning evidence.`;
  } else {
    counts = Array(8).fill(0);
    for (const body of frame.organisms) if (body.alive) counts[Math.min(7, Math.max(0, Math.floor(body.energy * 8)))]++;
    labels = counts.map((_, index) => (index / 8).toFixed(2)); note = 'Living bodies only. Energy bins from 0 to 1; last bin includes 1.';
  }
  const maximum = Math.max(1, ...counts), span = (width - 45) / counts.length, bottom = height - 25;
  drawing.font = '8px sans-serif'; drawing.textAlign = 'center';
  counts.forEach((value, index) => {
    const x = 25 + index * span, barHeight = value / maximum * (height - 60);
    drawing.fillStyle = index % 2 ? '#91bfc3' : '#8bcbb1'; drawing.fillRect(x, bottom - barHeight, Math.max(2, span - 8), barHeight);
    drawing.fillStyle = '#a4b6bd'; drawing.fillText(String(value), x + (span - 8) / 2, bottom - barHeight - 5);
    drawing.fillStyle = '#738e98'; drawing.fillText(labels[index], x + (span - 8) / 2, bottom + 15);
  });
  drawing.textAlign = 'left'; $('distribution-note').textContent = note;
}

const distributionPanel = element('section', { class: 'analytics distribution-panel' });
const distributionToolbar = element('div', { class: 'panel-toolbar' });
distributionToolbar.append(element('span', {}, 'STATE DISTRIBUTIONS'));
const distributionSelect = element('select', { id: 'distribution', 'aria-label': 'State distribution metric' });
for (const [value, label] of [['energy', 'Living energy'], ['actions', 'Last actions'], ['fitness', 'Evaluated fitness']]) distributionSelect.append(element('option', { value }, label));
distributionToolbar.append(distributionSelect);
distributionPanel.append(distributionToolbar, element('canvas', { id: 'distribution-chart', 'aria-label': 'State distribution histogram' }), element('p', { id: 'distribution-note', class: 'fine-print' }));
document.querySelector('.analytics').after(distributionPanel);
distributionSelect.onchange = () => chartDirty = true;
const engineError = element('p', { id: 'engine-error', class: 'form-error', role: 'alert', hidden: '' });
document.querySelector('.page-heading').after(engineError);

const decisionLabel = element('p', { id: 'decision-label', class: 'fine-print' });
$('mini-network').after(decisionLabel);
document.querySelector('.research-intro p').after(element('p', { class: 'fine-print' }, 'Sensory limitation: the forager uses body-facing information absent from the original neural inputs. It is a privileged reference, not an equal-information learning baseline.'));

function makeTable(headers, rows) {
  const table = element('table'), head = element('thead'), title = element('tr');
  for (const header of headers) title.append(element('th', {}, header));
  head.append(title); table.append(head);
  const body = element('tbody');
  for (const row of rows) {
    const line = element('tr');
    for (const value of row) {
      const cell = element('td'); cell.append(value instanceof Node ? value : document.createTextNode(String(value))); line.append(cell);
    }
    body.append(line);
  }
  table.append(body); return table;
}

function drawHistory() {
  const history = bundle ? bundle.history : state.history || [];
  $('history').replaceChildren();
  if (!history.length) $('history').append(element('p', { class: 'muted' }, 'No evaluated generation yet. Complete a generation to see measured fitness and species.'));
  else {
    const rows = history.map(row => {
      const inspect = element('button', { class: 'text-button' }, `Inspect ${row.champion} ↗`);
      inspect.onclick = () => { navigate('neural', 'evolution'); $('genome-choice').value = row.champion; drawNetworks(); rememberNavigation(); };
      const ancestry = element('button', { class: 'text-button' }, ' · Ancestry ↗');
      ancestry.onclick = () => { navigate('evolution', 'generation history'); $('evolution-genome').value = row.champion; drawEvolutionTree(); rememberNavigation(); };
      const navigation = element('span'); navigation.append(inspect, ancestry);
      return [row.world_generation, row.best_fitness.toFixed(3), row.mean_fitness.toFixed(3), row.species_count,
        row.mean_nodes.toFixed(1), row.mean_connections.toFixed(1), JSON.stringify(row.species), navigation];
    });
    $('history').append(makeTable(['World G', 'Best score', 'Mean score', 'Species', 'Mean nodes', 'Mean genes', 'Species sizes', 'Champion'], rows));
  }
  const selector = $('evolution-genome'), previous = selector.value;
  selector.replaceChildren();
  const records = bundle ? (bundle.version === 2 ? bundle.lineage : {}) : evolutionLineage;
  for (const key of Object.keys(records || {})) selector.append(element('option', { value: key }, `Genome ${key}`));
  if (records?.[previous]) selector.value = previous;
  else if (history.length && records?.[history.at(-1).champion]) selector.value = history.at(-1).champion;
  drawEvolutionTree();
  const frame = currentFrame(), lineage = $('lineage'); lineage.replaceChildren();
  if (!frame || selected === null) { lineage.append(element('p', { class: 'muted' }, 'Select an organism in the ecosystem to trace its ancestors and children.')); return; }
  const bodies = new Map(frame.organisms.map(body => [body.id, body])), ancestry = [], visited = new Set();
  let body = bodies.get(selected);
  while (body && !visited.has(body.id)) { ancestry.unshift(body); visited.add(body.id); body = bodies.get(body.parent); }
  for (const [index, ancestor] of ancestry.entries()) {
    if (index) lineage.append(element('span', { class: 'muted' }, '→'));
    const link = element('button', { class: 'lineage-node' }, `${ancestor.parent === null ? 'Founder' : 'Body'} ${ancestor.id}`);
    link.onclick = () => { selectBody(ancestor.id); }; lineage.append(link);
  }
  const children = frame.organisms.filter(item => item.parent === selected);
  if (children.length) {
    lineage.append(element('span', { class: 'muted' }, '→ children:'));
    for (const child of children.slice(0, 100)) {
      const link = element('button', { class: 'lineage-node' }, `${child.id} · ${child.alive ? 'alive' : 'dead'}`);
      link.onclick = () => selectBody(child.id); lineage.append(link);
    }
  }
}

const evolutionPanel = element('section', { class: 'lineage-panel' });
const evolutionToolbar = element('div', { class: 'panel-toolbar' });
evolutionToolbar.append(element('span', {}, 'EVOLUTIONARY GENOME ANCESTRY · V1'));
const evolutionChoice = element('select', { id: 'evolution-genome', 'aria-label': 'Evolutionary genome focus' });
evolutionToolbar.append(evolutionChoice);
evolutionPanel.append(evolutionToolbar, svgElement('svg', { id: 'evolution-tree', viewBox: '0 0 900 430', 'aria-label': 'Evolutionary genome ancestry tree' }),
  element('pre', { id: 'evolution-detail' }, 'Select a recorded genotype to inspect its reproduction and net mutation deltas.'),
  element('p', { class: 'fine-print' }, 'Up to four ancestral edges. Repeated boxes can refer to the same genotype. Parent selections include crossover, elite clones and champion rescues. Net mutation deltas compare the pre/post-mutation child; they are not a log of every operator invocation. This is separate from body splits within a world.'));
$('history').after(evolutionPanel);
evolutionChoice.onchange = () => { $('evolution-detail').textContent = ''; drawEvolutionTree(); rememberNavigation(); };

function drawEvolutionTree() {
  const records = bundle ? (bundle.version === 2 ? bundle.lineage : {}) : evolutionLineage;
  const focus = $('evolution-genome').value, graph = $('evolution-tree'); graph.replaceChildren();
  if (!records?.[focus]) {
    $('evolution-detail').textContent = 'No evolutionary provenance is available for this recording.';
    graph.append(svgElement('text', { x: 30, y: 70, fill: '#809aa5', 'font-size': 15 }, 'Evolutionary provenance is unavailable in this recording.'));
    return;
  }
  $('evolution-detail').textContent = JSON.stringify({ genome: focus, ...records[focus] }, null, 2);
  const levels = [[{ key: focus, next: null }]];
  for (let depth = 1; depth <= 4; depth++) {
    const entries = [];
    for (const child of levels[depth - 1]) {
      for (const parent of records[child.key]?.parents || []) entries.push({ key: parent, next: child });
    }
    if (!entries.length) break;
    levels.push(entries);
  }
  const height = Math.max(220, Math.max(...levels.map(level => level.length)) * 34 + 80);
  graph.setAttribute('viewBox', `0 0 900 ${height}`);
  levels.forEach((level, depth) => level.forEach((entry, index) => {
    entry.x = 450 + (levels.length - 1) * 90 - depth * 180; entry.y = 40 + (index + .5) / level.length * (height - 80);
  }));
  for (const level of levels) for (const entry of level) {
    if (entry.next) graph.append(svgElement('path', { d: `M${entry.x + 53},${entry.y} C${entry.x + 95},${entry.y} ${entry.next.x - 95},${entry.next.y} ${entry.next.x - 53},${entry.next.y}`,
      fill: 'none', stroke: '#45675f', 'stroke-width': 1.5 }));
  }
  for (const level of levels) for (const entry of level) {
    const group = svgElement('g', { class: 'genome-ancestor', tabindex: 0, role: 'button', 'aria-label': `Inspect genotype ${entry.key ?? 'unknown'}` });
    group.append(svgElement('rect', { x: entry.x - 53, y: entry.y - 14, width: 106, height: 28, rx: 5, fill: '#193029', stroke: '#52786a' }),
      svgElement('text', { x: entry.x, y: entry.y + 4, 'text-anchor': 'middle', fill: '#a6eccb', 'font-size': 11 }, entry.key ?? 'Unknown parent'),
      svgElement('title', {}, records[entry.key]?.kind || 'Unknown'));
    const inspect = () => {
      if (records[entry.key] && entry.key !== $('evolution-genome').value) { navigate('evolution', 'ancestry'); $('evolution-genome').value = entry.key; drawEvolutionTree(); rememberNavigation(); }
    };
    group.onclick = inspect;
    group.onkeydown = event => { if (event.key === 'Enter' || event.code === 'Space') { event.preventDefault(); event.stopPropagation(); inspect(); } };
    graph.append(group);
  }
}

const viewText = {
  ecosystem: ['LIVE ECOSYSTEM', 'Life, in motion', 'Observe a world of evolving neural organisms.'],
  neural: ['NEURAL ATLAS', 'Inside the decision', 'Topology, weights and the last actual inference.'],
  evolution: ['EVOLUTION EXPLORER', 'Across generations', 'Evaluated outcomes, genome ancestry and within-world body lineage.'],
  research: ['RESEARCH WORKBENCH', 'Ask better questions', 'Prespecified baselines, held-out worlds and portable evidence.']
};
const navigationSession = crypto.randomUUID();
let rememberedNavigation = '', lastHistoryWrite = 0;
history.scrollRestoration = 'manual';
let navigation = { session: navigationSession, view: 'ecosystem', nested: false, modal: null };
let livePresentation = null;
let replayIdentity = 0;
const backButton = element('button', { id: 'view-back', class: 'button secondary back-button', hidden: '' }, '← Back');
const bodyChoice = element('select', { id: 'body-choice', 'aria-label': 'Select organism for inspection' });
document.querySelector('.inspector-heading').after(bodyChoice);
bodyChoice.onchange = () => { if (bodyChoice.value !== '') selectBody(Number(bodyChoice.value)); };
document.querySelector('.page-heading').before(backButton);
$('view-title').tabIndex = -1;
$('network-detail').tabIndex = -1;

function presentation() {
  return { context: `${state.run_id}:${bundle ? replayIdentity : 'live'}`, generation: currentFrame()?.generation, selected, following,
    camera: { ...camera }, graphCamera: { ...graphCamera }, trail: trail.map(point => [...point]),
    genome: $('genome-choice').value, compare: $('genome-compare').value,
    ancestor: $('evolution-genome').value, detailText: $('network-detail').textContent,
    scroll: window.scrollY, focus: document.activeElement?.id,
    inspectionFocus: document.activeElement?.getAttribute('data-inspection') };
}

function rememberNavigation(force = true) {
  if (history.state?.session !== navigationSession) return;
  const next = { ...navigation, saved: presentation() }, signature = JSON.stringify(next), now = performance.now();
  if (signature === rememberedNavigation || (!force && now - lastHistoryWrite < 500)) return;
  history.replaceState(next, '', `#${view}`);
  rememberedNavigation = signature; lastHistoryWrite = now;
}

function restorePresentation(saved) {
  if (!saved || saved.context !== `${state.run_id}:${bundle ? replayIdentity : 'live'}`) return;
  selected = saved.generation === currentFrame()?.generation ? saved.selected : null;
  following = selected !== null && saved.following;
  trail = selected !== null ? saved.trail || [] : [];
  Object.assign(camera, saved.camera); Object.assign(graphCamera, saved.graphCamera);
  for (const [identity, value] of [['genome-choice', saved.genome], ['genome-compare', saved.compare], ['evolution-genome', saved.ancestor]]) {
    if ([...$(identity).options].some(option => option.value === value)) $(identity).value = value;
  }
  $('network-detail').textContent = saved.detailText;
  updateUI(); window.scrollTo(0, saved.scroll);
}

function navigate(next, label = null, modal = null) {
  if (!label && next === view && !navigation.nested && !modal) return;
  history.replaceState({ ...navigation, saved: presentation() }, '', `#${view}`);
  navigation = { session: navigationSession, view: next, nested: Boolean(label), label, modal };
  history.pushState(navigation, '', `#${next}`);
  applyNavigation();
}

function applyNavigation(saved = null) {
  for (const dialog of document.querySelectorAll('dialog[open]')) if (dialog.id !== navigation.modal) dialog.close();
  setView(navigation.view);
  restorePresentation(saved);
  backButton.hidden = !navigation.nested || Boolean(navigation.modal);
  backButton.textContent = `← Back to ${navigation.label || 'workspace'}`;
  if (navigation.modal) { if (!$(navigation.modal).open) $(navigation.modal).showModal(); }
  else {
    const focus = saved?.inspectionFocus ? document.querySelector(`[data-inspection="${saved.inspectionFocus}"]`) : $(saved?.focus);
    (focus || $('view-title')).focus({ preventScroll: true });
  }
  rememberNavigation();
}

backButton.onclick = () => { rememberNavigation(); history.back(); };
window.addEventListener('popstate', event => {
  if (event.state?.session === navigationSession) navigation = event.state;
  else {
    const requested = location.hash.slice(1);
    navigation = { session: navigationSession, view: viewText[requested] ? requested : 'ecosystem', nested: false, modal: null };
    history.replaceState(navigation, '', `#${navigation.view}`);
  }
  applyNavigation(navigation.saved);
});
for (const dialog of document.querySelectorAll('dialog')) {
  dialog.addEventListener('cancel', event => { event.preventDefault(); history.back(); });
}
const initialView = location.hash.slice(1);
navigation.view = viewText[initialView] ? initialView : 'ecosystem';
history.replaceState(navigation, '', `#${navigation.view}`);
applyNavigation();
document.querySelector('.brand').onclick = event => { event.preventDefault(); navigate('ecosystem'); };

function setView(next) {
  view = next;
  for (const button of document.querySelectorAll('.nav')) {
    button.classList.toggle('active', button.dataset.view === next);
    if (button.dataset.view === next) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  }
  for (const section of document.querySelectorAll('.view')) section.classList.toggle('active-view', section.id === `${next}-view`);
  const [kicker, title, description] = viewText[next];
  $('view-kicker').textContent = kicker; $('view-title').replaceChildren(document.createTextNode(title), element('span', {}, '.')); $('view-description').textContent = description;
  if (next === 'neural') { populateGenomeChoices(); drawNetworks(); }
  if (next === 'evolution') drawHistory();
  if (next === 'research') safely(loadArtifacts);
  chartDirty = true;
}

for (const button of document.querySelectorAll('.nav')) button.onclick = () => navigate(button.dataset.view);
$('open-neural').onclick = () => navigate('neural', 'organism inspector');
const clearNetworkDetail = () => {
  $('network-detail').textContent = 'Select a node or connection to view its recorded metadata.';
  navigation.detail = false; drawNetworks(); rememberNavigation();
};
$('genome-choice').onchange = clearNetworkDetail;
$('genome-compare').onchange = clearNetworkDetail;
$('network-fit').onclick = () => { Object.assign(graphCamera, { x: 0, y: 0, zoom: 1 }); drawNetworks(); };
$('fit').onclick = fitWorld;
$('grid').onclick = () => { grid = !grid; $('grid').setAttribute('aria-pressed', grid); };
$('follow').onclick = () => { if (selected === null) toast('Select an organism first'); else { following = !following; updateUI(); } };
$('layer').onchange = () => {
  const legends = { genome: 'Colors = genome identity · ◆ Food · triangles face movement direction', species: 'Gray = not evaluated · colors = known species', energy: 'Red = 0 energy → green = 1 energy', food: 'Green opacity = food count in a 5×5 neighborhood (saturates at 13)', density: 'Blue opacity = living body count in a 5×5 neighborhood (saturates at 13)', lineage: 'Colors = founder body ID within this world' };
  $('legend').textContent = legends[$('layer').value];
};

async function control(action) {
  if (bundle && ['step', 'resume', 'pause'].includes(action)) {
    if (action === 'step') { replayPlaying = false; replayIndex = Math.min(bundle.frames.length - 1, replayIndex + 1); }
    else replayPlaying = action === 'resume';
    chartDirty = true; updateUI(); return;
  }
  const next = await api('/api/control', { action });
  if (action === 'reset') {
    bundle = null; comparison = null; localFrames = []; lastSequence = -1; lastGeneration = -1;
    selected = null; trail = []; events = []; previousBodies.clear();
    $('comparison-panel').hidden = true; $('clear-compare').hidden = true;
    document.querySelector('.compare-key').hidden = true;
  }
  acceptState(next);
}
$('play').onclick = () => safely(() => control((bundle ? replayPlaying : !state.paused) ? 'pause' : 'resume'));
$('step').onclick = () => safely(() => control('step'));
$('reset').onclick = () => {
  if (confirm('Reset the live experiment? Unsaved recording will be replaced.')) safely(() => control('reset'));
};
$('speed').oninput = () => $('speed-label').textContent = `${$('speed').value} ${bundle ? 'frames/s' : 'ticks/s'}`;
$('speed').onchange = () => { if (!bundle) safely(() => api('/api/control', { action: 'speed', speed: Number($('speed').value) })); };

function enterReplay(data) {
  if (!bundle) livePresentation = presentation();
  replayIdentity++;
  bundle = data; replayIndex = 0; replayPlaying = false; selected = null; trail = []; events = [];
  $('scrub').max = data.frames.length - 1;
  $('generation-jump').replaceChildren();
  const seen = new Set();
  data.frames.forEach((frame, index) => {
    if (!seen.has(frame.generation)) { seen.add(frame.generation); $('generation-jump').append(element('option', { value: index }, `Generation ${frame.generation}`)); }
  });
  populateGenomeChoices(); chartDirty = true; fitWorld(); updateUI();
  toast(`Loaded ${data.frames.length} actual recorded frames. Playback only; cannot resume engine from replay.`);
}
$('review').onclick = () => safely(async () => {
  await control('pause');
  const data = await api('/api/replay');
  if (!data.frames.length) throw new Error('Recording is disabled or no frames available');
  enterReplay(data);
});
  $('live').onclick = () => {
  bundle = null; comparison = null; replayPlaying = false; selected = null; trail = [];
  $('comparison-panel').hidden = true; $('clear-compare').hidden = true;
  document.querySelector('.compare-key').hidden = true;
  chartDirty = true; populateGenomeChoices(); acceptState(state); updateUI();
  restorePresentation(livePresentation); livePresentation = null;
  $('speed').value = state.speed; updateUI(false);
};
$('scrub').oninput = () => {
  replayIndex = Number($('scrub').value); replayPlaying = false; trail = []; selected = null;
  chartDirty = true; updateUI();
};
$('generation-jump').onchange = () => { replayIndex = Number($('generation-jump').value); replayPlaying = false; selected = null; trail = []; chartDirty = true; updateUI(); };

async function readReplay(file) {
  if (!file || file.size > 24 * 1024 * 1024) throw new Error('Choose a replay file below 24 MiB');
  let stream = file.stream();
  if (file.name.endsWith('.gz')) stream = stream.pipeThrough(new DecompressionStream('gzip'));
  const reader = stream.getReader(), chunks = []; let length = 0;
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    length += value.length;
    if (length > 24 * 1024 * 1024) { await reader.cancel(); throw new Error('Decompressed replay exceeds 24 MiB'); }
    chunks.push(value);
  }
  const response = await fetch('/api/replay/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: new Blob(chunks) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid replay');
  return data;
}
$('import').onchange = () => safely(async () => {
  const file = $('import').files[0]; if (!file) return;
  if (state.frame && !state.paused) await control('pause');
  enterReplay(await readReplay(file)); $('import').value = '';
});
$('compare-import').onchange = () => safely(async () => {
  comparison = await readReplay($('compare-import').files[0]);
  $('comparison-panel').hidden = false; $('clear-compare').hidden = false;
  const config = comparison.config;
  $('comparison-label').textContent = `SECOND RUN · SEED ${config.seed} · FOOD ${config.food} · ${config.width} × ${config.height}`;
  document.querySelector('.compare-key').hidden = false;
  chartDirty = true; $('compare-import').value = '';
  toast('Comparison matches recorded generation/tick. Different configurations are displayed explicitly.');
});
$('clear-compare').onclick = () => { comparison = null; $('comparison-panel').hidden = true; $('clear-compare').hidden = true; document.querySelector('.compare-key').hidden = true; chartDirty = true; };

$('export').onclick = () => safely(async () => {
  if (bundle) download(JSON.stringify(bundle), 'clage-replay.json');
  else {
    const response = await fetch('/api/export');
    if (!response.ok) throw new Error('No live recording to export');
    download(await response.blob(), 'clage-replay.json.gz');
  }
});
$('screenshot').onclick = () => { drawWorld(canvas, currentFrame(), currentConfig()); canvas.toBlob(blob => download(blob, 'clage-world.png', 'image/png')); };
$('chart-export').onclick = () => { drawChart(); $('chart').toBlob(blob => download(blob, 'clage-chart.png', 'image/png')); };
$('csv').onclick = () => download(metricsCSV(framesForChart()), 'clage-metrics.csv', 'text/csv');
$('metric').onchange = () => chartDirty = true;
$('chart-range').onchange = () => chartDirty = true;
$('save').onclick = () => safely(async () => {
  if (bundle) { download(JSON.stringify(bundle), 'clage-replay.json'); toast('Imported replay downloaded; live artifact was not changed.'); return; }
  const result = await api('/api/artifacts', {}); toast(`Saved .studio-runs/${result.filename}`); await loadArtifacts();
});
$('champion-export').onclick = () => safely(async () => {
  if (!bundle) { download(JSON.stringify(await api('/api/champion'), null, 2), 'clage-champion.json'); return; }
  const history = bundle ? bundle.history : state.history || [];
  if (!history.length) return toast('Complete an evaluated generation first');
  const best = history.reduce((first, second) => first.best_fitness >= second.best_fitness ? first : second);
  download(JSON.stringify({ schema: 'clage-champion', version: 1, config: currentConfig(), generation: best.world_generation,
    genome: currentGenomes()[best.champion], metadata: bundle?.metadata || { source: 'current live run' } }, null, 2), 'clage-champion.json');
});

async function loadArtifacts() {
  const records = await api('/api/artifacts'); $('artifacts').replaceChildren();
  for (const record of records) $('artifacts').append(element('a', { class: 'artifact', href: `/api/artifacts/${record.id}`, download: `${record.id}.json.gz` }, `${record.id.slice(0, 8)} · ${(record.bytes / 1024).toFixed(1)} KiB ↓`));
}

let evaluation = null;
$('evaluate').onclick = () => safely(async () => {
  $('evaluate').disabled = true; $('evaluate').textContent = 'Evaluating prespecified worlds…';
  try {
    evaluation = await api('/api/evaluate', {});
    const rows = evaluation.results.map(row => [row.policy, row.seed, row.ticks, row.food_eaten, row.survivors, row.births, row.fitness.toFixed(3)]);
    $('evaluation').replaceChildren(makeTable(['Policy', 'World seed', 'Budget', 'Food consumed', 'Surviving bodies', 'Births', 'Max body fitness'], rows));
    const groups = new Map();
    for (const row of evaluation.results) {
      if (!groups.has(row.policy)) groups.set(row.policy, []); groups.get(row.policy).push(row.food_eaten);
    }
    const summary = element('p', { class: 'fine-print' });
    summary.textContent = [...groups].map(([policy, values]) => {
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      const deviation = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1));
      return `${policy}: consumed ${mean.toFixed(2)} ± ${deviation.toFixed(2)} (sample SD, n=${values.length})`;
    }).join(' · ');
    $('evaluation').prepend(summary);
    $('evaluation-export').hidden = false;
    toast(evaluation.champion ? 'Held-out evaluation complete. No training was performed.' : 'Baseline evaluation complete. No evaluated champion available yet.');
  } finally { $('evaluate').disabled = false; $('evaluate').textContent = 'Run held-out evaluation ↗'; }
});
$('evaluation-export').onclick = () => download(JSON.stringify(evaluation, null, 2), 'clage-heldout-evaluation.json');

for (const [key, label, value, min, max, step, description] of fieldDefinitions) {
  const field = element('div', { class: 'field' });
  field.append(element('label', { for: `config-${key}` }, label), element('input', { id: `config-${key}`, name: key, type: 'number', value, min, max, step, required: '' }), element('small', {}, description));
  $('config-fields').append(field);
}
const initialize = element('div', { class: 'field' }), initializeSelect = element('select', { id: 'config-initialization', name: 'initialization' });
initializeSelect.append(element('option', { value: 'dense-random-v1' }, 'Dense random · v1'), element('option', { value: 'minimal-v1' }, 'Historical minimal · v1'));
initialize.append(element('label', { for: 'config-initialization' }, 'Initialization definition'), initializeSelect, element('small', {}, 'Scientific definition, recorded in artifacts.'));
$('config-fields').append(initialize);
const recording = element('div', { class: 'field' });
recording.append(element('label', { for: 'config-record' }, 'Record recent frames'), element('input', { id: 'config-record', type: 'checkbox', checked: '' }), element('small', {}, '600 frames / 16 MiB JSON budget. Not a checkpoint.'));
$('config-fields').append(recording);

function formConfig() {
  const config = {};
  for (const [key] of fieldDefinitions) config[key] = Number($(`config-${key}`).value);
  config.initialization = $('config-initialization').value; config.record = $('config-record').checked; return config;
}
function fillForm(config) {
  for (const [key] of fieldDefinitions) if (config[key] !== undefined) $(`config-${key}`).value = config[key];
  if (config.initialization) $('config-initialization').value = config.initialization;
  if (config.record !== undefined) $('config-record').checked = config.record;
}
$('configure').onclick = () => { if (state.config) fillForm(state.config); navigate(view, 'workspace', 'config-dialog'); };
const topConfigure = element('button', { class: 'button secondary', id: 'heading-configure', title: 'Configure experiment' }, '＋ Configure');
document.querySelector('.heading-actions').prepend(topConfigure);
topConfigure.onclick = $('configure').onclick;
$('close-config').onclick = () => history.back();
$('close-config').className = 'button secondary back-button';
$('close-config').textContent = '← Back';
$('close-config').setAttribute('aria-label', 'Back to previous view');
$('shortcuts').onclick = () => navigate(view, 'workspace', 'help-dialog');
$('close-help').onclick = () => history.back();
$('close-help').className = 'button secondary back-button';
$('close-help').textContent = '← Back';
$('close-help').setAttribute('aria-label', 'Back to previous view');
$('live').textContent = '← Back to live';
$('preset').onchange = () => {
  const defaults = Object.fromEntries(fieldDefinitions.map(([key, , value]) => [key, value]));
  defaults.initialization = 'dense-random-v1'; defaults.record = true;
  const presets = { scarce: { food: 30, regrowth: 0 }, abundant: { food: 600, regrowth: 6 }, minimal: { initialization: 'minimal-v1' }, large: { population: 512, width: 80, height: 64, food: 500 } };
  fillForm({ ...defaults, ...presets[$('preset').value] });
};
$('preset-save').onclick = () => { localStorage.setItem('clage-preset-v1', JSON.stringify(formConfig())); toast('Preset saved in this browser.'); };
$('preset-load').onclick = () => safely(async () => {
  const saved = localStorage.getItem('clage-preset-v1'); if (!saved) throw new Error('No browser preset saved'); fillForm(JSON.parse(saved));
});
$('config-form').onsubmit = async event => {
  event.preventDefault(); $('config-error').textContent = '';
  const submit = $('config-form').querySelector('[type=submit]'); submit.disabled = true;
  try {
    const next = await api('/api/runs', formConfig());
    bundle = null; comparison = null; localFrames = []; lastSequence = -1; lastGeneration = -1; selected = null;
    previousBodies.clear(); trail = []; events = []; evaluation = null; $('evaluation').replaceChildren(); $('evaluation-export').hidden = true;
    $('comparison-panel').hidden = true; $('clear-compare').hidden = true; document.querySelector('.compare-key').hidden = true;
    $('config-dialog').close(); fitWorld(); acceptState(next); await loadGenomes();
    navigation = { session: navigationSession, view, nested: false, modal: null };
    history.replaceState(navigation, '', `#${view}`); applyNavigation();
    toast('New experiment ready. Configuration frozen. Resume to advance.');
  } catch (error) { $('config-error').textContent = error.message; }
  finally { submit.disabled = false; }
};

document.addEventListener('keydown', event => {
  if (event.defaultPrevented) return;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName) || document.querySelector('dialog[open]')) return;
  if (event.key === 'Escape' && navigation.nested) { event.preventDefault(); history.back(); return; }
  if (event.code === 'Space' && event.target.closest('button, [role="button"]')) return;
  if (event.code === 'Space') { event.preventDefault(); $('play').click(); }
  if (event.key === '.') $('step').click();
  if (event.key.toLowerCase() === 'f') fitWorld();
  if (event.key.toLowerCase() === 'r') $('reset').click();
});

function animate(now) {
  countFrames++;
  if (now - frameClock > 1000) {
    fps = countFrames * 1000 / (now - frameClock); countFrames = 0; frameClock = now;
    $('render-rate').textContent = `${fps.toFixed(0)} render FPS`;
    canvas.dataset.drawMilliseconds = (drawingMilliseconds / Math.max(1, drawingSamples)).toFixed(3);
    drawingMilliseconds = 0; drawingSamples = 0;
  }
  if (bundle && replayPlaying && now - playbackClock >= 1000 / Number($('speed').value)) {
    playbackClock = now;
    if (replayIndex >= bundle.frames.length - 1) replayPlaying = false;
    else replayIndex++;
    chartDirty = true; updateUI();
  }
  if (view === 'ecosystem') {
    const started = performance.now();
    drawWorld(canvas, currentFrame(), currentConfig());
    drawingMilliseconds += performance.now() - started; drawingSamples++;
    if (comparison && currentFrame()) {
      const matched = matchedFrame(comparison.frames, currentFrame());
      drawWorld($('comparison'), matched, comparison.config, false);
      const config = comparison.config;
      $('comparison-label').textContent = matched ? `SECOND RUN · SEED ${config.seed} · FOOD ${config.food} · ${config.width}×${config.height} · ACTUAL G${matched.generation}:T${matched.tick}` : 'SECOND RUN · NO EARLIER RECORDED FRAME IN THIS GENERATION';
    }
    if (chartDirty) { drawChart(); drawDistribution(); chartDirty = false; }
  }
  requestAnimationFrame(animate);
}

window.addEventListener('resize', () => chartDirty = true);
requestAnimationFrame(animate);
await safely(async () => {
  const initial = await api('/api/state');
  if (!initial.frame) acceptState(await api('/api/runs', formConfig())); else acceptState(initial);
  safely(loadGenomes);
});
connect();
