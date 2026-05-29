---
title: "공부"
permalink: /study/
author_profile: true
---

공부하면서 정리한 개념, 메모, 참고 자료를 올리는 공간입니다.

{% assign study_posts = site.categories.study %}

{% if study_posts.size > 0 %}
{% for post in study_posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) <small>{{ post.date | date: "%Y-%m-%d" }}</small>
{% endfor %}
{% else %}
글은 아직 작성하지 않았습니다.
{% endif %}
