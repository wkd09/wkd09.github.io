# 장준민의 기술 블로그

GitHub Pages 기반 개인 기술 블로그입니다.

## Focus

- AI / LLM
- Agent
- 논문리뷰
- 구현 / 프로그래밍
- 공부

## Local Development

Ruby와 Bundler가 설치되어 있어야 합니다.

```bash
bundle install
bundle exec jekyll serve
```

로컬 서버가 실행되면 보통 다음 주소에서 확인할 수 있습니다.

```txt
http://127.0.0.1:4000
```

## Writing

새 글은 `_posts` 디렉터리에 날짜 기반 파일명으로 작성합니다.

```txt
_posts/YYYY-MM-DD-post-slug.md
```

기본 front matter 예시:

```yaml
---
title: "글 제목"
date: 2026-05-28 00:00:00 +0900
categories:
  - engineering
tags:
  - LangGraph
  - AgentOps
---
```

카테고리는 글의 목적 기준으로 하나만 고릅니다.

- `study`: 개념 공부, 이론 정리, 논문 읽기 전 배경지식
- `engineering`: 구현, 시스템 설계, 성능 최적화, 서빙, 배포
- `research`: 논문 리뷰, 실험 결과, 비교 분석

이미지는 `assets/images/blog`에 소문자 kebab-case 파일명으로 넣고 본문에서는 절대 경로를 사용합니다.

```md
![설명](/assets/images/blog/example-image.png)
```

글 작성 템플릿은 `_templates` 디렉터리에 있습니다.

- `_templates/post-template.md`
- `_templates/research-template.md`

## Deployment

이 저장소는 `wkd09.github.io` 사용자 Pages 저장소를 기준으로 설정되어 있습니다.

GitHub 저장소의 Pages 설정에서 Source를 `GitHub Actions`로 선택한 뒤, `main` branch에 push하면 `.github/workflows/pages.yml` workflow가 Jekyll 사이트를 빌드하고 GitHub Pages에 배포합니다.

배포 주소:

```txt
https://wkd09.github.io
```

## Configuration

개인 링크와 사이트 정보는 `_config.yml`에서 수정합니다.

- GitHub: `https://github.com/wkd09`
- Email: `your-email@example.com`
- LinkedIn: optional
- Velog: optional
- Notion: optional

## Theme

이 블로그는 Jekyll과 [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes)를 기반으로 구성했습니다.
Minimal Mistakes의 라이선스와 출처 표기는 theme footer와 의존성 정보를 통해 유지합니다.
