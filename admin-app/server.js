import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createLocalContentStore } from './lib/local-content.js';
import { publishPending } from './lib/git-publisher.js';
import {
  handleError,
  methodNotAllowed,
  readJson,
  requireSameOrigin,
  sendJson,
} from './lib/http.js';
import { validatePostFilename } from './lib/validation.js';

const appRoot = path.dirname(fileURLToPath(import.meta.url));
const blogRoot = path.resolve(appRoot, '..');
const publicRoot = path.join(appRoot, 'public');
const store = createLocalContentStore(blogRoot);
const host = '127.0.0.1';
const port = Number.parseInt(process.env.PORT || '4173', 10);

const staticFiles = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/admin', ['index.html', 'text/html; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
]);

function allowedHost(hostHeader = '') {
  return hostHeader === `127.0.0.1:${port}`
    || hostHeader === `localhost:${port}`
    || hostHeader === '127.0.0.1'
    || hostHeader === 'localhost';
}

async function apiHandler(req, res, url) {
  if (url.pathname === '/api/posts') {
    if (req.method === 'GET') {
      const filename = url.searchParams.get('filename');
      if (!filename) return sendJson(res, 200, { posts: await store.listPosts(), pending: await store.listPending() });
      validatePostFilename(filename);
      return sendJson(res, 200, { post: await store.readPost(filename), pending: await store.listPending() });
    }
    if (req.method === 'PUT') {
      requireSameOrigin(req);
      const post = await store.savePost(await readJson(req));
      return sendJson(res, 200, { post });
    }
    return methodNotAllowed(res, ['GET', 'PUT']);
  }

  if (url.pathname === '/api/upload') {
    if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
    requireSameOrigin(req);
    const image = await store.saveImage(await readJson(req, 4_200_000));
    return sendJson(res, 200, { image });
  }

  if (url.pathname === '/api/publish') {
    if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
    requireSameOrigin(req);
    const result = await publishPending(blogRoot, store);
    return sendJson(res, 200, { result });
  }

  if (url.pathname === '/api/status') {
    if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
    return sendJson(res, 200, { pending: await store.listPending() });
  }

  return sendJson(res, 404, { error: 'API를 찾을 수 없습니다.' });
}

async function requestHandler(req, res) {
  if (!allowedHost(req.headers.host)) return sendJson(res, 403, { error: '로컬 주소로만 접근할 수 있습니다.' });
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https://avatars.githubusercontent.com; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'");

  const url = new URL(req.url, `http://${req.headers.host}`);
  try {
    if (url.pathname.startsWith('/api/')) return await apiHandler(req, res, url);
    if (req.method !== 'GET' && req.method !== 'HEAD') return methodNotAllowed(res, ['GET', 'HEAD']);
    const staticFile = staticFiles.get(url.pathname);
    if (!staticFile) return sendJson(res, 404, { error: '페이지를 찾을 수 없습니다.' });
    const [filename, contentType] = staticFile;
    const content = await readFile(path.join(publicRoot, filename));
    res.statusCode = 200;
    res.setHeader('Content-Type', contentType);
    res.setHeader('Cache-Control', 'no-store');
    return res.end(req.method === 'HEAD' ? undefined : content);
  } catch (error) {
    return handleError(res, error);
  }
}

const server = createServer(requestHandler);
server.listen(port, host, () => {
  console.log(`블로그 관리자: http://${host}:${port}`);
});
