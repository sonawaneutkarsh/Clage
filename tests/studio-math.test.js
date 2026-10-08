import test from 'node:test';
import assert from 'node:assert/strict';
import { colorFor, layoutNetwork, matchedFrame, metricsCSV } from '../studio/static/math.js';

test('colors are stable for genome identities', () => {
  assert.equal(colorFor('2:14'), colorFor('2:14'));
  assert.notEqual(colorFor('2:14'), colorFor('2:15'));
});
test('layout includes hidden layers and ignores disabled edges', () => {
  const genome = { nodes: [{ id: 0, type: 'INPUT' }, { id: 14, type: 'HIDDEN' }, { id: 10, type: 'OUTPUT' }],
    connections: [{ in: 0, out: 14, enabled: true }, { in: 14, out: 10, enabled: true }, { in: 10, out: 14, enabled: false }] };
  const positions = layoutNetwork(genome);
  assert.ok(positions.get(0).x < positions.get(14).x);
  assert.ok(positions.get(14).x < positions.get(10).x);
});
test('comparison never substitutes another generation or future frame', () => {
  const frames = [{ generation: 0, tick: 1 }, { generation: 0, tick: 3 }, { generation: 1, tick: 0 }];
  assert.equal(matchedFrame(frames, { generation: 0, tick: 2 }), frames[0]);
  assert.equal(matchedFrame(frames, { generation: 2, tick: 5 }), null);
});
test('CSV describes actual sequence and all metric values', () => {
  const csv = metricsCSV([{ sequence: 2, generation: 0, tick: 2, metrics: { population: 10, food: 4, food_eaten: 1, mean_energy: .5, births: 2, deaths: 1 } }]);
  assert.ok(csv.includes('2,0,2,10,4,1,0.5,2,1'));
});
