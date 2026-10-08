import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  timeout: 45000,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:8876', viewport: { width: 1440, height: 1100 }, screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  webServer: { command: 'python3 -m studio --port 8876', url: 'http://127.0.0.1:8876/api/state', reuseExistingServer: false, timeout: 30000 },
  reporter: [['list'], ['json', { outputFile: 'test-results/browser-results.json' }]]
});
