---
title: "Troubleshooting"
permalink: /troubleshooting/
layout: single
author_profile: true
---

{% assign troubleshooting_count = site.categories.troubleshooting.size | default: 0 %}
{% assign troubleshooting_posts = site.categories.troubleshooting %}

<section class="blog-intro">
  <p>개발 중 만난 에러, 원인 분석, 해결 과정을 정리하는 공간입니다.</p>
  <span>{{ troubleshooting_count }}개의 글</span>
</section>

{% if troubleshooting_posts.size > 0 %}
<div class="post-list">
  {% for post in troubleshooting_posts %}
    {% include post-list-item.html post=post %}
  {% endfor %}
</div>
{% else %}
<div class="empty-state">
  <p>아직 작성한 트러블 슈팅 글이 없습니다.</p>
  <span>문제 해결 기록을 추가하면 이곳에 최신순으로 표시됩니다.</span>
</div>
{% endif %}
