import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';

test('predeclared paused Canvas workloads: 72, 512, 1000, 2000 bodies', async ({ page, request, browserName }) => {
  const results = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const population of [72, 512, 1000, 2000]) {
    const response = await request.post('/api/runs', { data: { population, generations: 1, ticks: 30,
      width: 96, height: 96, food: 900, seed: 42, repro_threshold: 2.0, record: false } });
    expect(response.ok()).toBe(true);
    await page.goto('/');
    await expect(page.locator('#population')).toHaveText(population.toLocaleString('en-US'));
    await expect(page.locator('#render-rate')).not.toContainText('—');
    const measured = await page.evaluate(() => new Promise(resolve => {
      const times = [];
      const observe = timestamp => {
        times.push(timestamp);
        if (times.length === 121) {
          const intervals = times.slice(1).map((time, index) => time - times[index]).sort((first, second) => first - second);
          const canvas = document.getElementById('world');
          const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
          let opaque = 0;
          for (let index = 3; index < pixels.length; index += 4) if (pixels[index]) opaque++;
          resolve({ fps: 120000 / (times.at(-1) - times[0]), p95_frame_ms: intervals[Math.floor(intervals.length * .95)],
            mean_draw_ms: Number(canvas.dataset.drawMilliseconds), painted_pixels: opaque });
        } else requestAnimationFrame(observe);
      };
      requestAnimationFrame(observe);
    }));
    expect(measured.painted_pixels).toBeGreaterThan(10000);
    expect(measured.fps).toBeGreaterThan(0);
    results.push({ population, world: '96x96', food: 900, layer: 'genome', grid: false, camera: 'fit', paused: true, ...measured });
  }
  await fs.mkdir('docs/studio', { recursive: true });
  await fs.writeFile('docs/studio/browser-benchmark.json', JSON.stringify({ schema: 'clage-render-benchmark', version: 1,
    browser: browserName, viewport: '1440x1100', frames_per_workload: 120,
    limitations: 'Headless Chromium RAF and draw CPU timing on this machine. Paused snapshots, no inference/stream load or density layer. Not a GPU/browser-wide scalability guarantee.', results }, null, 2));
  expect(errors).toEqual([]);
});
