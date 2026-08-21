const POST_FILENAME = /^\d{4}-\d{2}-\d{2}-[a-z0-9가-힣](?:[a-z0-9가-힣-]*[a-z0-9가-힣])?\.md$/u;
const IMAGE_FILENAME = /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.(?:png|jpe?g|gif|webp)$/;

export function validatePost(filename, content) {
  if (typeof filename !== 'string' || !POST_FILENAME.test(filename)) {
    const error = new Error('글 파일명이 올바르지 않습니다. 날짜와 영문·숫자·한글 slug를 사용해 주세요.');
    error.statusCode = 400;
    throw error;
  }
  if (typeof content !== 'string' || content.length < 10 || content.length > 1_000_000) {
    const error = new Error('본문은 10자 이상, 1MB 이하여야 합니다.');
    error.statusCode = 400;
    throw error;
  }
  if (!content.startsWith('---\n') || content.indexOf('\n---\n', 4) < 0) {
    const error = new Error('Jekyll front matter 형식이 올바르지 않습니다.');
    error.statusCode = 400;
    throw error;
  }
}

export function validatePostFilename(filename) {
  if (typeof filename !== 'string' || !POST_FILENAME.test(filename)) {
    const error = new Error('글 파일명이 올바르지 않습니다.');
    error.statusCode = 400;
    throw error;
  }
}

export function sanitizeImageFilename(filename) {
  if (typeof filename !== 'string') throw invalidImageError();
  const normalized = filename.normalize('NFKD').toLowerCase()
    .replace(/[^a-z0-9.]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/-\./g, '.')
    .replace(/^-|-$/g, '');
  if (!IMAGE_FILENAME.test(normalized)) throw invalidImageError();
  return `${Date.now()}-${normalized}`;
}

export function decodeImage(base64, mimeType) {
  const allowedMimeTypes = new Set([
    'image/png', 'image/jpeg', 'image/gif', 'image/webp',
  ]);
  if (!allowedMimeTypes.has(mimeType) || typeof base64 !== 'string') throw invalidImageError();
  const bytes = Buffer.from(base64, 'base64');
  if (bytes.length === 0 || bytes.length > 3_000_000) {
    const error = new Error('이미지는 3MB 이하여야 합니다.');
    error.statusCode = 400;
    throw error;
  }
  return bytes;
}

function invalidImageError() {
  const error = new Error('지원하지 않는 이미지 파일입니다.');
  error.statusCode = 400;
  return error;
}
