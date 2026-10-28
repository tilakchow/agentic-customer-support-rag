import { proxyToBackend } from '../_proxy.js'

export const config = {
  runtime: 'edge',
}

export default function handler(req: Request) {
  return proxyToBackend(req, '/kb/info', false)
}
