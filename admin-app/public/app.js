const elements = Object.fromEntries([
  'editor-view', 'new-button', 'post-search', 'post-list', 'post-form',
  'editing-label', 'editor-title', 'editor-message', 'save-button', 'publish-button',
  'pending-status', 'title',
  'date', 'slug', 'summary', 'categories', 'tags', 'source', 'published',
  'body', 'image-file', 'upload-button', 'filename-preview',
].map((id) => [id, document.getElementById(id)]));

const state = {
  posts: [],
  currentFilename: null,
  currentSha: null,
  dirty: false,
  slugWasEdited: false,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || '요청을 처리하지 못했습니다.');
  }
  return data;
}

function setMessage(element, message, isError = false) {
  element.textContent = message;
  element.hidden = !message;
  element.classList.toggle('error', isError);
}

function todayInSeoul() {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date());
}

function slugify(value) {
  return value.normalize('NFKC').toLowerCase()
    .replace(/[^a-z0-9가-힣]+/gu, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-+/g, '-');
}

function currentFilename() {
  return state.currentFilename || `${elements.date.value}-${elements.slug.value}.md`;
}

function updateFilenamePreview() {
  elements['filename-preview'].textContent = currentFilename();
}

function splitValues(value) {
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

function quoteYaml(value) {
  return JSON.stringify(String(value));
}

function serializePost() {
  const categories = splitValues(elements.categories.value);
  const tags = splitValues(elements.tags.value);
  const lines = [
    '---',
    `title: ${quoteYaml(elements.title.value.trim())}`,
    `date: ${elements.date.value} 00:00:00 +0900`,
    `summary: ${quoteYaml(elements.summary.value.trim())}`,
    'categories:',
    ...(categories.length ? categories : ['study']).map((value) => `  - ${quoteYaml(value)}`),
    'tags:',
    ...(tags.length ? tags : ['기타']).map((value) => `  - ${quoteYaml(value)}`),
  ];
  if (elements.source.value.trim()) lines.push(`source: ${quoteYaml(elements.source.value.trim())}`);
  if (!elements.published.checked) lines.push('published: false');
  lines.push('---', '', elements.body.value.replace(/^\n+/, ''));
  return `${lines.join('\n').replace(/\s+$/, '')}\n`;
}

function unquoteYaml(value) {
  const trimmed = value.trim();
  if (trimmed.startsWith('"')) {
    try { return JSON.parse(trimmed); } catch { return trimmed.slice(1, -1); }
  }
  return trimmed.replace(/^'|'$/g, '');
}

function parsePost(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) throw new Error('글의 front matter를 읽을 수 없습니다.');

  const data = { categories: [], tags: [], published: true, body: match[2] };
  let listKey = null;
  for (const line of match[1].split(/\r?\n/)) {
    const listItem = line.match(/^\s+-\s+(.+)$/);
    if (listItem && listKey) {
      data[listKey].push(unquoteYaml(listItem[1]));
      continue;
    }
    const pair = line.match(/^([A-Za-z_]+):\s*(.*)$/);
    if (!pair) continue;
    const [, key, value] = pair;
    listKey = value ? null : key;
    if (key === 'categories' || key === 'tags') {
      if (value.startsWith('[')) {
        data[key] = value.slice(1, -1).split(',').map(unquoteYaml).filter(Boolean);
      }
    } else if (key === 'published') {
      data.published = value !== 'false';
    } else {
      data[key] = unquoteYaml(value);
    }
  }
  return data;
}

function populateForm(post, filename, sha) {
  state.currentFilename = filename;
  state.currentSha = sha;
  state.slugWasEdited = true;
  elements.title.value = post.title || '';
  elements.date.value = String(post.date || filename.slice(0, 10)).slice(0, 10);
  elements.slug.value = filename.slice(11, -3);
  elements.summary.value = post.summary || '';
  elements.categories.value = (post.categories || []).join(', ');
  elements.tags.value = (post.tags || []).join(', ');
  elements.source.value = post.source || '';
  elements.published.checked = post.published !== false;
  elements.body.value = post.body || '';
  elements['editing-label'].textContent = 'EDIT POST';
  elements['editor-title'].textContent = '글 수정';
  state.dirty = false;
  updateFilenamePreview();
  renderPostList();
}

function startNewPost() {
  if (state.dirty && !window.confirm('저장하지 않은 변경사항을 버리고 새 글을 작성할까요?')) return;
  state.currentFilename = null;
  state.currentSha = null;
  state.slugWasEdited = false;
  elements['post-form'].reset();
  elements.date.value = todayInSeoul();
  elements.published.checked = true;
  elements['editing-label'].textContent = 'NEW POST';
  elements['editor-title'].textContent = '새 글 작성';
  setMessage(elements['editor-message'], '');
  state.dirty = false;
  updateFilenamePreview();
  renderPostList();
  elements.title.focus();
}

async function loadPostList() {
  const { posts, pending } = await api('/api/posts');
  state.posts = posts;
  updatePending(pending);
  renderPostList();
}

function updatePending(paths = []) {
  const count = paths.length;
  elements['publish-button'].disabled = count === 0;
  elements['pending-status'].textContent = count ? `GitHub 반영 대기 ${count}개` : '반영할 변경 없음';
}

function renderPostList() {
  const query = elements['post-search'].value.trim().toLowerCase();
  elements['post-list'].replaceChildren();
  for (const post of state.posts.filter((item) => item.name.toLowerCase().includes(query))) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'post-item';
    button.classList.toggle('active', post.name === state.currentFilename);
    button.textContent = post.name.replace(/\.md$/, '');
    button.addEventListener('click', () => openPost(post));
    elements['post-list'].append(button);
  }
}

async function openPost(post) {
  if (state.dirty && !window.confirm('저장하지 않은 변경사항을 버리고 다른 글을 열까요?')) return;
  setMessage(elements['editor-message'], '글을 불러오는 중입니다.');
  try {
    const { post: loaded } = await api(`/api/posts?filename=${encodeURIComponent(post.name)}`);
    populateForm(parsePost(loaded.content), loaded.name, loaded.sha);
    setMessage(elements['editor-message'], '');
  } catch (error) {
    setMessage(elements['editor-message'], error.message, true);
  }
}

elements['new-button'].addEventListener('click', startNewPost);
elements['post-search'].addEventListener('input', renderPostList);
elements.title.addEventListener('input', () => {
  if (!state.currentFilename && !state.slugWasEdited) elements.slug.value = slugify(elements.title.value);
  updateFilenamePreview();
});
elements.slug.addEventListener('input', () => {
  state.slugWasEdited = true;
  elements.slug.value = slugify(elements.slug.value);
  updateFilenamePreview();
});
elements.date.addEventListener('input', updateFilenamePreview);
elements['post-form'].addEventListener('input', () => { state.dirty = true; });

elements['post-form'].addEventListener('submit', async (event) => {
  event.preventDefault();
  const filename = currentFilename();
  elements['save-button'].disabled = true;
  setMessage(elements['editor-message'], '로컬 파일에 저장하는 중입니다.');
  try {
    const { post } = await api('/api/posts', {
      method: 'PUT',
      body: JSON.stringify({ filename, content: serializePost(), sha: state.currentSha }),
    });
    state.currentFilename = post.name;
    state.currentSha = post.sha;
    state.dirty = false;
    updatePending(post.pending);
    await loadPostList();
    setMessage(elements['editor-message'], '로컬 파일에 저장했습니다. 확인 후 GitHub에 올려 주세요.');
  } catch (error) {
    setMessage(elements['editor-message'], error.message, true);
  } finally {
    elements['save-button'].disabled = false;
  }
});

elements['upload-button'].addEventListener('click', async () => {
  const file = elements['image-file'].files[0];
  if (!file) return setMessage(elements['editor-message'], '업로드할 이미지를 선택해 주세요.', true);
  if (file.size > 3_000_000) return setMessage(elements['editor-message'], '이미지는 3MB 이하여야 합니다.', true);

  elements['upload-button'].disabled = true;
  setMessage(elements['editor-message'], '이미지를 업로드하는 중입니다.');
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = '';
    for (const byte of bytes) binary += String.fromCharCode(byte);
    const { image } = await api('/api/upload', {
      method: 'POST',
      body: JSON.stringify({ filename: file.name, mimeType: file.type, base64: btoa(binary) }),
    });
    const markdown = `![${file.name.replace(/\.[^.]+$/, '')}](${image.path})`;
    const start = elements.body.selectionStart;
    const end = elements.body.selectionEnd;
    elements.body.setRangeText(`\n${markdown}\n`, start, end, 'end');
    elements.body.focus();
    state.dirty = true;
    updatePending(image.pending);
    setMessage(elements['editor-message'], '이미지를 업로드하고 본문에 경로를 넣었습니다.');
  } catch (error) {
    setMessage(elements['editor-message'], error.message, true);
  } finally {
    elements['upload-button'].disabled = false;
  }
});

window.addEventListener('beforeunload', (event) => {
  if (!state.dirty) return;
  event.preventDefault();
});

elements['publish-button'].addEventListener('click', async () => {
  if (!window.confirm('대기 중인 블로그 파일을 커밋하고 GitHub에 push할까요?')) return;
  elements['publish-button'].disabled = true;
  setMessage(elements['editor-message'], 'GitHub에 올리는 중입니다.');
  try {
    const { result } = await api('/api/publish', { method: 'POST', body: '{}' });
    updatePending([]);
    setMessage(elements['editor-message'], `${result.files.length}개 파일을 ${result.branch} 브랜치에 올렸습니다.`);
  } catch (error) {
    setMessage(elements['editor-message'], error.message, true);
    const { pending } = await api('/api/status').catch(() => ({ pending: [] }));
    updatePending(pending);
  }
});

loadPostList()
  .then(startNewPost)
  .catch((error) => setMessage(elements['editor-message'], error.message, true));
