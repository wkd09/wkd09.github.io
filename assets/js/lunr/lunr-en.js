---
layout: none
---

var idx = lunr(function () {
  this.field("title");
  this.field("excerpt");
  this.field("categories");
  this.field("tags");
  this.ref("id");
  this.pipeline.remove(lunr.trimmer);

  for (var item in store) {
    this.add({
      title: store[item].title,
      excerpt: store[item].excerpt,
      categories: store[item].categories,
      tags: store[item].tags,
      id: item
    });
  }
});

(function ($) {
  var categoryLabels = {
    study: "공부",
    research: "논문",
    engineering: "구현"
  };

  function escapeHtml(value) {
    return $("<div>").text(value || "").html();
  }

  function highlight(value, terms) {
    var escaped = escapeHtml(value);
    var safeTerms = terms
      .filter(Boolean)
      .map(function (term) {
        return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      });

    if (!safeTerms.length) return escaped;
    return escaped.replace(new RegExp("(" + safeTerms.join("|") + ")", "gi"), "<mark>$1</mark>");
  }

  function categoriesFor(entry) {
    return (entry.categories || [])
      .map(function (category) { return categoryLabels[category] || category; })
      .join(" · ");
  }

  function renderResults(resultdiv, results, query) {
    var terms = query.toLowerCase().split(lunr.tokenizer.separator).filter(Boolean);
    resultdiv.empty();

    if (!query.trim()) {
      resultdiv.append(
        '<div class="search-empty"><p>찾고 싶은 주제를 입력해 주세요.</p>' +
        '<span>예: LLM, 캐싱, Transformer</span></div>'
      );
      return;
    }

    if (!results.length) {
      resultdiv.append(
        '<div class="search-empty"><p>검색 결과가 없습니다.</p>' +
        '<span>다른 키워드나 더 짧은 검색어를 사용해 보세요.</span></div>'
      );
      return;
    }

    resultdiv.append('<p class="results__found">' + results.length + '개의 검색 결과</p>');

    results.forEach(function (result) {
      var entry = store[result.ref];
      var excerpt = (entry.excerpt || "").split(/\s+/).slice(0, 24).join(" ");
      var meta = [entry.date, categoriesFor(entry)].filter(Boolean).join(" · ");
      var searchItem =
        '<article class="search-result" itemscope itemtype="https://schema.org/CreativeWork">' +
          '<p class="search-result__meta">' + escapeHtml(meta) + '</p>' +
          '<h2 class="search-result__title" itemprop="headline">' +
            '<a href="' + escapeHtml(entry.url) + '" rel="permalink">' + highlight(entry.title, terms) + '</a>' +
          '</h2>' +
          '<p class="search-result__excerpt" itemprop="description">' + highlight(excerpt, terms) + '…</p>' +
        '</article>';

      resultdiv.append(searchItem);
    });
  }

  $(document).ready(function () {
    var input = $("input#search");
    var resultdiv = $("#results");

    function search() {
      var query = input.val().toLowerCase();
      var results = [];

      if (query.trim()) {
        results = idx.query(function (q) {
          query.split(lunr.tokenizer.separator).forEach(function (term) {
            if (!term) return;
            q.term(term, { boost: 100 });
            q.term(term, { usePipeline: false, wildcard: lunr.Query.wildcard.TRAILING, boost: 10 });
            q.term(term, { usePipeline: false, editDistance: 1, boost: 1 });
          });
        });

        var queryTerms = query.split(lunr.tokenizer.separator).filter(Boolean);
        var matchedRefs = results.reduce(function (refs, result) {
          refs[result.ref] = true;
          return refs;
        }, {});

        store.forEach(function (entry, index) {
          var haystack = [entry.title, entry.excerpt]
            .concat(entry.categories || [], entry.tags || [])
            .join(" ")
            .toLowerCase();
          var ref = String(index);

          if (!matchedRefs[ref] && queryTerms.every(function (term) { return haystack.indexOf(term) !== -1; })) {
            results.push({ ref: ref, score: 0 });
            matchedRefs[ref] = true;
          }
        });
      }

      renderResults(resultdiv, results, query);
    }

    input.on("input", search);
    $(".search-content__close").on("click", function () {
      $(".search__toggle").trigger("click");
    });
    search();
  });
}(jQuery));
