import { spawn } from 'node:child_process';

function gitError(message, statusCode = 500) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

export function runGit(blogRoot, args, acceptedExitCodes = [0]) {
  return new Promise((resolve, reject) => {
    const child = spawn('git', args, {
      cwd: blogRoot,
      env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    const timeout = setTimeout(() => child.kill('SIGTERM'), 60_000);
    child.stdout.on('data', (chunk) => { if (stdout.length < 100_000) stdout += chunk; });
    child.stderr.on('data', (chunk) => { if (stderr.length < 100_000) stderr += chunk; });
    child.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on('close', (code, signal) => {
      clearTimeout(timeout);
      if (signal) return reject(gitError('Git 명령 시간이 초과되었습니다.'));
      const result = { code, stdout: stdout.trim(), stderr: stderr.trim() };
      if (acceptedExitCodes.includes(code)) return resolve(result);
      return reject(gitError(result.stderr || result.stdout || 'Git 명령에 실패했습니다.', 502));
    });
  });
}

export async function publishPending(blogRoot, contentStore) {
  const pending = await contentStore.listPending();
  if (pending.length === 0) throw gitError('GitHub에 올릴 로컬 변경사항이 없습니다.', 400);

  const stagedResult = await runGit(blogRoot, ['diff', '--cached', '--name-only', '-z']);
  const staged = stagedResult.stdout.split('\0').filter(Boolean);
  const unrelated = staged.filter((filename) => !pending.includes(filename));
  if (unrelated.length) {
    throw gitError(`다른 staged 변경사항이 있어 중단했습니다: ${unrelated.join(', ')}`, 409);
  }

  await runGit(blogRoot, ['add', '--', ...pending]);
  const diff = await runGit(blogRoot, ['diff', '--cached', '--quiet'], [0, 1]);
  let committed = false;
  if (diff.code === 1) {
    await runGit(blogRoot, ['commit', '-m', 'Update blog content from local editor']);
    committed = true;
  }

  const branch = (await runGit(blogRoot, ['branch', '--show-current'])).stdout;
  if (!branch) throw gitError('현재 Git 브랜치를 확인할 수 없습니다.', 409);
  await runGit(blogRoot, ['push', 'origin', branch]);
  await contentStore.clearPending();
  return { branch, committed, files: pending };
}
