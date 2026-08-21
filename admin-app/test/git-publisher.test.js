import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createLocalContentStore } from '../lib/local-content.js';
import { publishPending, runGit } from '../lib/git-publisher.js';

test('commits and pushes only files saved by the local editor', async (context) => {
  const container = await mkdtemp(path.join(tmpdir(), 'myblog-publish-'));
  const root = path.join(container, 'blog');
  const remote = path.join(container, 'remote.git');
  await mkdir(path.join(root, '_posts'), { recursive: true });
  await mkdir(path.join(root, 'admin-app'), { recursive: true });
  context.after(() => rm(container, { recursive: true, force: true }));

  await runGit(container, ['init', '--bare', remote]);
  await runGit(root, ['init', '-b', 'main']);
  await runGit(root, ['config', 'user.name', 'Local Editor Test']);
  await runGit(root, ['config', 'user.email', 'local-editor@example.test']);
  await writeFile(path.join(root, 'README.md'), 'fixture\n');
  await runGit(root, ['add', 'README.md']);
  await runGit(root, ['commit', '-m', 'Initial commit']);
  await runGit(root, ['remote', 'add', 'origin', remote]);
  await runGit(root, ['push', '-u', 'origin', 'main']);

  const store = createLocalContentStore(root);
  const filename = '2026-08-21-publish-test.md';
  await store.savePost({ filename, content: '---\ntitle: test\n---\npublish body\n' });
  const result = await publishPending(root, store);

  assert.equal(result.branch, 'main');
  assert.equal(result.committed, true);
  assert.deepEqual(result.files, [`_posts/${filename}`]);
  assert.deepEqual(await store.listPending(), []);
  assert.match((await runGit(root, ['log', '-1', '--pretty=%s'])).stdout, /Update blog content/);
});
