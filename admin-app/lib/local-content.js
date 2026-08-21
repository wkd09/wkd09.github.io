import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, rename, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  decodeImage,
  sanitizeImageFilename,
  validatePost,
  validatePostFilename,
} from './validation.js';

const POST_DIRECTORY = '_posts';
const IMAGE_DIRECTORY = 'assets/images/blog';
const PENDING_FILE = 'admin-app/.admin-pending.json';

function digest(content) {
  return createHash('sha256').update(content).digest('hex');
}

function contentPathAllowed(relativePath) {
  return relativePath.startsWith(`${POST_DIRECTORY}/`)
    || relativePath.startsWith(`${IMAGE_DIRECTORY}/`);
}

function statusError(message, statusCode) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

export function createLocalContentStore(blogRoot) {
  const root = path.resolve(blogRoot);
  const pendingPath = path.join(root, PENDING_FILE);

  async function readPending() {
    try {
      const values = JSON.parse(await readFile(pendingPath, 'utf8'));
      if (!Array.isArray(values)) return [];
      return [...new Set(values.filter((value) => typeof value === 'string' && contentPathAllowed(value)))];
    } catch (error) {
      if (error.code === 'ENOENT' || error instanceof SyntaxError) return [];
      throw error;
    }
  }

  async function writePending(paths) {
    const values = [...new Set(paths)].filter(contentPathAllowed).sort();
    const temporaryPath = `${pendingPath}.${process.pid}.tmp`;
    await writeFile(temporaryPath, `${JSON.stringify(values, null, 2)}\n`, { mode: 0o600 });
    await rename(temporaryPath, pendingPath);
  }

  async function markPending(relativePath) {
    if (!contentPathAllowed(relativePath)) throw statusError('관리 대상이 아닌 경로입니다.', 400);
    const paths = await readPending();
    if (!paths.includes(relativePath)) paths.push(relativePath);
    await writePending(paths);
    return paths.sort();
  }

  async function listPosts() {
    const directory = path.join(root, POST_DIRECTORY);
    const names = (await readdir(directory)).filter((name) => name.endsWith('.md')).sort().reverse();
    return Promise.all(names.map(async (name) => {
      const info = await stat(path.join(directory, name));
      return { name, size: info.size };
    }));
  }

  async function readPost(filename) {
    validatePostFilename(filename);
    const content = await readFile(path.join(root, POST_DIRECTORY, filename), 'utf8').catch((error) => {
      if (error.code === 'ENOENT') throw statusError('글을 찾을 수 없습니다.', 404);
      throw error;
    });
    return { name: filename, sha: digest(content), content };
  }

  async function savePost({ filename, content, sha }) {
    validatePost(filename, content);
    const target = path.join(root, POST_DIRECTORY, filename);
    let existing = null;
    try {
      existing = await readFile(target, 'utf8');
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }

    if (existing !== null && !sha) throw statusError('같은 파일명의 글이 이미 있습니다.', 409);
    if (existing !== null && sha && digest(existing) !== sha) {
      throw statusError('글이 다른 곳에서 변경되었습니다. 목록에서 다시 열어 주세요.', 409);
    }
    if (existing === null && sha) throw statusError('수정하려는 글을 찾을 수 없습니다.', 409);

    const temporaryPath = `${target}.${process.pid}.tmp`;
    await writeFile(temporaryPath, content, 'utf8');
    await rename(temporaryPath, target);
    const pending = await markPending(`${POST_DIRECTORY}/${filename}`);
    return { name: filename, sha: digest(content), pending };
  }

  async function saveImage({ filename, mimeType, base64 }) {
    const safeFilename = sanitizeImageFilename(filename);
    const bytes = decodeImage(base64, mimeType);
    const relativePath = `${IMAGE_DIRECTORY}/${safeFilename}`;
    const target = path.join(root, relativePath);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, bytes, { flag: 'wx' });
    const pending = await markPending(relativePath);
    return { path: `/${relativePath}`, pending };
  }

  return {
    clearPending: () => writePending([]),
    listPending: readPending,
    listPosts,
    readPost,
    saveImage,
    savePost,
  };
}
