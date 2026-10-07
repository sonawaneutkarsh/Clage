export function colorFor(identity) {
  let hash = 0;
  for (const character of String(identity)) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return `hsl(${(hash * 137.508) % 360} 43% 67%)`;
}

export function layoutNetwork(genome, width = 800, height = 620) {
  const depths = new Map(genome.nodes.map(node => [node.id, 0]));
  const enabled = genome.connections.filter(edge => edge.enabled);
  for (let pass = 0; pass < genome.nodes.length; pass++) {
    let changed = false;
    for (const edge of enabled) {
      const next = depths.get(edge.in) + 1;
      if (next > depths.get(edge.out)) { depths.set(edge.out, next); changed = true; }
    }
    if (!changed) break;
  }
  const hidden = Math.max(0, ...genome.nodes.filter(node => node.type === 'HIDDEN').map(node => depths.get(node.id)));
  for (const node of genome.nodes) if (node.type === 'OUTPUT') depths.set(node.id, hidden + 1);
  const maxDepth = Math.max(1, ...depths.values());
  const layers = new Map();
  for (const node of genome.nodes) {
    const depth = depths.get(node.id);
    if (!layers.has(depth)) layers.set(depth, []);
    layers.get(depth).push(node);
  }
  const positions = new Map();
  for (const [depth, nodes] of layers) {
    nodes.sort((first, second) => first.id - second.id);
    nodes.forEach((node, index) => positions.set(node.id, {
      x: 155 + depth / maxDepth * (width - 310), y: 65 + (index + .5) / nodes.length * (height - 130)
    }));
  }
  return positions;
}

export function matchedFrame(frames, target) {
  let found = null;
  for (const frame of frames) {
    if (frame.generation === target.generation && frame.tick <= target.tick) found = frame;
    if (frame.generation > target.generation) break;
  }
  return found;
}

export function metricsCSV(frames) {
  const keys = ['population', 'food', 'food_eaten', 'mean_energy', 'births', 'deaths'];
  return ['sequence,generation,tick,' + keys.join(','), ...frames.map(frame =>
    [frame.sequence, frame.generation, frame.tick, ...keys.map(key => frame.metrics[key])].join(','))].join('\n');
}
