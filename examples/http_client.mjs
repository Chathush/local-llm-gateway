// Node 20+. Start llm-gateway serve in the app directory first.
import { readFile } from 'node:fs/promises';

const token = (await readFile('.gateway-token', 'utf8')).trim();
const response = await fetch('http://127.0.0.1:8787/v1/chat/completions', {
  method: 'POST',
  headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: process.argv[2] || 'primary',
    messages: [{ role: 'user', content: 'Describe this local gateway in one sentence.' }],
    stream: false,
  }),
  signal: AbortSignal.timeout(125_000),
});
if (!response.ok) throw new Error(`Gateway HTTP ${response.status}: ${await response.text()}`);
console.log((await response.json()).choices[0].message.content);
