export function sendJson(res, status, body, headers = {}) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  for (const [key, value] of Object.entries(headers)) res.setHeader(key, value);
  res.end(JSON.stringify(body));
}

export async function readJson(req, maxBytes = 1_200_000) {
  if (req.body && typeof req.body === 'object' && !Buffer.isBuffer(req.body)) {
    return req.body;
  }
  if (typeof req.body === 'string' || Buffer.isBuffer(req.body)) {
    if (Buffer.byteLength(req.body) > maxBytes) {
      const error = new Error('요청 데이터가 너무 큽니다.');
      error.statusCode = 413;
      throw error;
    }
    try {
      return JSON.parse(req.body.toString());
    } catch {
      const error = new Error('올바른 JSON 요청이 아닙니다.');
      error.statusCode = 400;
      throw error;
    }
  }

  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) {
      const error = new Error('요청 데이터가 너무 큽니다.');
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    const error = new Error('올바른 JSON 요청이 아닙니다.');
    error.statusCode = 400;
    throw error;
  }
}

export function requireSameOrigin(req) {
  const origin = req.headers.origin;
  const host = req.headers['x-forwarded-host'] || req.headers.host;
  if (!origin || !host) return;

  try {
    if (new URL(origin).host === host) return;
  } catch {
    // Fall through to the rejection below.
  }

  const error = new Error('허용되지 않은 요청 출처입니다.');
  error.statusCode = 403;
  throw error;
}

export function methodNotAllowed(res, allowed) {
  res.setHeader('Allow', allowed.join(', '));
  sendJson(res, 405, { error: '지원하지 않는 요청 방식입니다.' });
}

export function handleError(res, error) {
  const status = Number.isInteger(error.statusCode) ? error.statusCode : 500;
  const message = status >= 500
    ? '서버에서 요청을 처리하지 못했습니다.'
    : error.message;
  if (status >= 500) console.error(error);
  sendJson(res, status, { error: message });
}
