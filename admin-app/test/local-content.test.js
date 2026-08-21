import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createLocalContentStore } from '../lib/local-content.js';

async function fixture(context) {
  const root = await mkdtemp(path.join(tmpdir(), 'myblog-content-'));
  await mkdir(path.join(root, '_posts'), { recursive: true });
  await mkdir(path.join(root, 'assets/images/blog'), { recursive: true });
  await mkdir(path.join(root, 'admin-app'), { recursive: true });
  context.after(() => rm(root, { recursive: true, force: true }));
  return { root, store: createLocalContentStore(root) };
}

test('creates and updates a post with optimistic conflict protection', async (context) => {
  const { root, store } = await fixture(context);
  const filename = '2026-08-21-local-editor.md';
  const first = '---\ntitle: "로컬 편집기"\n---\n첫 번째 본문입니다.\n';
  const created = await store.savePost({ filename, content: first });
  assert.equal(created.name, filename);
  assert.deepEqual(created.pending, [`_posts/${filename}`]);

  const loaded = await store.readPost(filename);
  assert.equal(loaded.content, first);
  const second = '---\ntitle: "로컬 편집기"\n---\n수정한 본문입니다.\n';
  await store.savePost({ filename, content: second, sha: loaded.sha });
  assert.equal(await readFile(path.join(root, '_posts', filename), 'utf8'), second);
  await assert.rejects(() => store.savePost({ filename, content: first, sha: loaded.sha }), /다른 곳에서 변경/);
});

test('uploads images and records only managed paths as pending', async (context) => {
  const { root, store } = await fixture(context);
  const image = await store.saveImage({
    filename: 'Diagram.PNG',
    mimeType: 'image/png',
    base64: Buffer.from('image-bytes').toString('base64'),
  });
  assert.match(image.path, /^\/assets\/images\/blog\/\d+-diagram\.png$/);
  assert.deepEqual(await store.listPending(), [image.path.slice(1)]);
  assert.deepEqual(await readFile(path.join(root, image.path)), Buffer.from('image-bytes'));
});
