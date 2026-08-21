import test from 'node:test';
import assert from 'node:assert/strict';
import {
  decodeImage,
  sanitizeImageFilename,
  validatePost,
  validatePostFilename,
} from '../lib/validation.js';

test('accepts valid Jekyll post names and content', () => {
  assert.doesNotThrow(() => validatePostFilename('2026-08-21-cache-design.md'));
  assert.doesNotThrow(() => validatePost('2026-08-21-캐시-설계.md', '---\ntitle: test\n---\n본문입니다.'));
});

test('rejects traversal and malformed post names', () => {
  assert.throws(() => validatePostFilename('../config.yml'));
  assert.throws(() => validatePostFilename('cache-design.md'));
  assert.throws(() => validatePost('2026-08-21-test.md', 'front matter 없음'));
});

test('sanitizes and timestamps image names', () => {
  const filename = sanitizeImageFilename('My Diagram (Final).PNG');
  assert.match(filename, /^\d+-my-diagram-final\.png$/);
});

test('accepts supported image bytes and rejects unsupported types', () => {
  assert.deepEqual(decodeImage('aGVsbG8=', 'image/png'), Buffer.from('hello'));
  assert.throws(() => decodeImage('aGVsbG8=', 'text/plain'));
  assert.throws(() => decodeImage('aGVsbG8=', 'image/svg+xml'));
});
