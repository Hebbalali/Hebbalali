---
layout: page
title: Resume
permalink: /resume/
---

<div id="pdf">
  <canvas id="pdf-canvas"></canvas>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.9.359/pdf.min.js" integrity="sha512-iKWpMimpp5Ke5/5nRk5jGnJ/5My3q/iD8kj/nptiN2Q/AwNcUhMa4E4x7V73ZJtiv1CtIyvAGSd7VbYPNSDbcg==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>


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