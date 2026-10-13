/// <reference types="node" />

export const config = {
  runtime: 'edge',
}

export default async function handler(req: Request) {
  const backendUrl =
    process.env.BACKEND_URL || process.env.VITE_API_URL || 'http://localhost:8000'
  const targetUrl = `${backendUrl.replace(/\/$/, '')}/health`

  try {
    const apiKey =
      process.env.API_KEY || req.headers.get('x-api-key') || req.headers.get('X-API-Key')

    const response = await fetch(targetUrl, {
      method: req.method,
      headers: {
        ...(apiKey ? { 'X-API-Key': apiKey } : {}),
      },
    })

    return new Response(response.body, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('content-type') || 'application/json',
      },
    })
  } catch (error) {
    return new Response(
      JSON.stringify({ error: `Backend health proxy failure: ${String(error)}` }),
      {
        status: 502,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}
