/// <reference types="node" />

export const config = {
  runtime: 'edge',
}

export default async function handler(req: Request) {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const backendUrl =
    process.env.BACKEND_URL || process.env.VITE_API_URL || 'http://localhost:8000'
  const targetUrl = `${backendUrl.replace(/\/$/, '')}/chat/stream`

  try {
    const body = await req.text()
    const apiKey =
      process.env.API_KEY || req.headers.get('x-api-key') || req.headers.get('X-API-Key')

    const response = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey ? { 'X-API-Key': apiKey } : {}),
      },
      body,
    })

    return new Response(response.body, {
      status: response.status,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
      },
    })
  } catch (error) {
    return new Response(
      JSON.stringify({ error: `Backend streaming proxy failure: ${String(error)}` }),
      {
        status: 502,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}
