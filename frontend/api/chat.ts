import { proxyToBackend } from './_proxy.js'

export const config = {
  runtime: 'edge',
}

export default function handler(req: Request) {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    })
  }
  return proxyToBackend(req, '/chat', false)
}
