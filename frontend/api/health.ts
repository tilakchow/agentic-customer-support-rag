import { proxyToBackend } from './_proxy'

export const config = {
  runtime: 'edge',
}

export default function handler(req: Request) {
  return proxyToBackend(req, '/health', false)
}
