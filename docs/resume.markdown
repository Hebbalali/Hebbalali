---
layout: page
title: Resume
permalink: /resume/

<script src="https://hebbalali.github.io/Hebbalali/assets/web/pdf.js"></script>
<script src="https://hebbalali.github.io/Hebbalali/assets/web/pdf.worker.js"></script>
<link rel="stylesheet" href="https://hebbalali.github.io/Hebbalali/web/build/pdf_viewer.css" />
---

<div id="pdf">
  <canvas id="pdf-canvas"></canvas>
</div>

<script>
  var url = "https://hebbalali.github.io/Hebbalali/assets/Cv_Hebbal_2023.pdf";
  var pdfDoc = null;
  var pageNum = 1;
  var pageRendering = false;
  var pageNumPending = null;
  var canvas = document.getElementById('pdf-canvas');
  var ctx = canvas.getContext('2d');

  function renderPage(num) {
    pageRendering = true;
    pdfDoc.getPage(num).then(function(page) {
      var viewport = page.getViewport({scale: 1});
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      var renderContext = {
        canvasContext: ctx,
        viewport: viewport
      };
      var renderTask = page.render(renderContext);
      renderTask.promise.then(function() {
        pageRendering = false;
        if (pageNumPending !== null) {
          renderPage(pageNumPending);
          pageNumPending = null;
        }
      });
    });
    document.getElementById('page-num').textContent = num;
  }

  pdfjsLib.getDocument(url).promise.then(function(pdfDoc_) {
    pdfDoc = pdfDoc_;
    document.getElementById('page-count').textContent = pdfDoc.numPages;
    renderPage(pageNum);
  });

  document.getElementById('prev-page').addEventListener('click', function() {
    if (pageNum <= 1) {
      return;
    }
    pageNum--;
    renderPage(pageNum);
  });

  document.getElementById('next-page').addEventListener('click', function() {
    if (pageNum >= pdfDoc.numPages) {
      return;
    }
    pageNum++;
    renderPage(pageNum);
  });
</script>