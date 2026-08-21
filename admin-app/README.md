# 로컬 블로그 관리자

이 Mac에서만 실행되는 Jekyll 글 편집기입니다. 로그인 서비스나 Vercel 없이 `_posts`와 `assets/images/blog`를 직접 수정합니다.

## 실행

```bash
cd /Users/jangjunmin/Documents/code/myblog/admin-app
npm start
```

브라우저에서 [http://127.0.0.1:4173](http://127.0.0.1:4173)을 엽니다.

개발 중 파일 변경을 자동 반영하려면 다음 명령을 사용할 수 있습니다.

```bash
npm run dev
```

## 사용 흐름

1. 기존 글을 선택하거나 새 글을 작성합니다.
2. **로컬에 저장**을 누릅니다.
3. 생성된 Markdown과 이미지를 확인합니다.
4. **GitHub에 올리기**를 누르면 관리자에서 수정한 파일만 commit하고 현재 브랜치로 push합니다.

저장과 push를 분리했기 때문에 글을 로컬에서 확인한 뒤 공개할 수 있습니다. `published`를 끄고 저장하면 Jekyll 비공개 글로 저장됩니다.

## 보안과 Git 처리

- 서버는 `127.0.0.1`에만 바인딩되어 외부 네트워크에서 접근할 수 없습니다.
- 요청의 Host와 Origin을 검사합니다.
- 글 파일명과 이미지 형식을 검증하고 상위 경로 접근을 차단합니다.
- 관리자에서 변경한 파일 목록은 Git에 포함되지 않는 `.admin-pending.json`에 기록합니다.
- 이미 staged된 다른 파일이 있으면 의도하지 않은 commit을 막기 위해 push를 중단합니다.
- `git push`는 현재 컴퓨터에 설정된 기존 GitHub 인증을 사용합니다.

## 테스트

```bash
npm test
```
