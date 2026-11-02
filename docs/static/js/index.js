(function () {
  "use strict";

  var CHECK = "\u2713", CROSS = "\u2717", PHI = "\u03A6";

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function preload(srcs) {
    srcs.forEach(function (s) { var im = new Image(); im.src = s; });
  }

  var BENCHES = [
    { key: "VSI-Bench", label: "VSI-Bench", sub: "video" },
    { key: "SPAR-Bench", label: "SPAR-Bench", sub: "multi-view" },
    { key: "MAZE", label: "MAZE", sub: "navigation" }
  ];

  /* =====================================================================
   * Case viewer: steps through a recorded multi-turn rollout turn by turn.
   * ===================================================================== */
  function CaseViewer(root) {
    var group = root.getAttribute("data-group");
    var isFail = group === "outcomeOnly";
    var all = (window.VERSE_CASES && window.VERSE_CASES[group]) || [];
    var state = { bench: isFail ? "MAZE" : "VSI-Bench", idx: 0, turn: 0 };

    root.innerHTML =
      '<div class="cv-tabs"></div><div class="cv-chips"></div>' +
      '<div class="cv-card"><div class="cv-head"></div><div class="cv-body">' +
      '<div class="cv-stage"><div class="cv-stage-main"></div><div class="cv-stage-caption"></div>' +
      '<div class="cv-controls">' +
      '<button class="cv-btn prev" aria-label="Previous turn"><i class="fas fa-backward-step"></i></button>' +
      '<button class="cv-btn next" aria-label="Next turn"><i class="fas fa-forward-step"></i></button>' +
      '<div class="cv-progress"></div><div class="cv-step-label"></div></div></div>' +
      '<div class="cv-transcript"></div></div></div>';

    var tabsEl = root.querySelector(".cv-tabs");
    var chipsEl = root.querySelector(".cv-chips");
    var headEl = root.querySelector(".cv-head");
    var stageEl = root.querySelector(".cv-stage-main");
    var capEl = root.querySelector(".cv-stage-caption");
    var progEl = root.querySelector(".cv-progress");
    var labelEl = root.querySelector(".cv-step-label");
    var transEl = root.querySelector(".cv-transcript");
    var prevBtn = root.querySelector(".cv-btn.prev");
    var nextBtn = root.querySelector(".cv-btn.next");

    function casesOf(bench) { return all.filter(function (c) { return c.bench === bench; }); }
    function current() { return casesOf(state.bench)[state.idx]; }

    function turnsOf(c) {
      var t = [];
      if (c.inputs && c.inputs.length) t.push({ kind: "input" });
      c.steps.forEach(function (s, i) { t.push({ kind: "tool", step: s, n: i + 1 }); });
      t.push({ kind: "answer" });
      return t;
    }

    function correct(c) { return c.score > 0; }

    function frameTag(c, f) {
      if (c.bench === "MAZE") return "";
      return (c.bench === "VSI-Bench" ? "frame " : "image ") + f;
    }

    function renderTabs() {
      tabsEl.innerHTML = BENCHES.filter(function (b) { return casesOf(b.key).length; }).map(function (b) {
        return '<button class="cv-tab' + (b.key === state.bench ? " active" : "") + '" data-bench="' + b.key + '">' +
          esc(b.label) + "<small>" + esc(b.sub) + "</small></button>";
      }).join("");
    }

    function renderChips() {
      chipsEl.innerHTML = casesOf(state.bench).map(function (c, i) {
        return '<button class="cv-chip' + (isFail ? " fail" : "") + (i === state.idx ? " active" : "") + '" data-idx="' + i + '">' +
          esc(c.task) + (isFail ? "" : '<span class="chip-sub">' + esc(c.subtitle) + "</span>") + "</button>";
      }).join("");
    }

    function renderHead(c) {
      var who = isFail ? "Outcome-only" : "VERSE";
      var ok = correct(c);
      var pills = [
        '<span class="pill bench">' + esc(c.bench) + "</span>",
        '<span class="pill">' + esc(c.subtype || c.task) + "</span>",
        '<span class="pill">GT: ' + esc(c.gt) + "</span>",
        '<span class="pill ' + (ok ? "good" : "bad") + '">' + who + ": " + esc(c.pred) + " " + (ok ? CHECK : CROSS) + "</span>",
        '<span class="pill">score ' + Number(c.score).toFixed(1) + (c.problemType === "regression" ? " (MRA)" : "") + "</span>"
      ];
      if (c.nFrames > 1) pills.push('<span class="pill">' + c.nFrames + (c.bench === "VSI-Bench" ? " video frames" : " views") + "</span>");
      var opts = (c.options || []).length
        ? '<div class="cv-options">' + c.options.map(function (o) { return "<span>" + esc(o) + "</span>"; }).join("") + "</div>" : "";
      var note = isFail ? '<div class="cv-failure-note"><b>' + esc(c.task) + ".</b> " + esc(c.subtitle) + "</div>" : "";
      headEl.innerHTML = '<div class="cv-meta">' + pills.join("") + '</div><div class="cv-question">' + esc(c.question) + "</div>" + opts + note;
    }

    function thinkHtml(t) {
      if (!t) return "";
      return '<div class="turn-think"><span class="tag">&lt;think&gt;</span> ' + esc(t) + ' <span class="tag">&lt;/think&gt;</span></div>';
    }

    function renderTranscript(c, turns) {
      transEl.innerHTML = turns.map(function (t, i) {
        var h;
        if (t.kind === "input") {
          var used = c.inputs.filter(function (x) { return x.used; }).map(function (x) { return x.index; });
          h = '<div class="turn-head"><span>Input</span><span class="tool">question + ' + c.inputs.length +
            (c.inputs.length > 1 ? " images" : " image") + "</span></div>" +
            '<div class="turn-input-list">' + (c.inputs.length > 4
              ? "Frames sampled from a scene video. VERSE later draws on frame" + (used.length > 1 ? "s " : " ") + used.join(", ") + "."
              : "The model sees the image" + (c.inputs.length > 1 ? "s" : "") + " and the question above.") + "</div>";
        } else if (t.kind === "tool") {
          h = '<div class="turn-head"><span>Turn ' + t.n + '</span><span class="tool">Action: ' + esc(t.step.tool) + "</span></div>" +
            thinkHtml(t.step.think) + (t.step.json ? '<div class="turn-json">' + esc(t.step.json) + "</div>" : "");
        } else {
          var ok = correct(c);
          h = '<div class="turn-head"><span>Final turn</span><span class="tool answer' + (ok ? "" : " bad") + '">Action: Answer ' +
            (ok ? CHECK : CROSS) + "</span></div>" + thinkHtml(c.final.think) +
            '<span class="turn-answer">&lt;answer&gt; ' + esc(c.final.answer) + " &lt;/answer&gt;</span>" +
            (ok ? "" : ' <span class="pill bad">ground truth: ' + esc(c.gt) + "</span>");
        }
        return '<div class="turn" data-turn="' + i + '">' + h + "</div>";
      }).join("");
    }

    function imgHtml(src, tag) {
      return '<span class="img-wrap"><img src="' + esc(src) + '" alt="">' + (tag ? '<span class="img-tag">' + esc(tag) + "</span>" : "") + "</span>";
    }

    function renderStage(c, turns) {
      var t = turns[state.turn];
      stageEl.className = "cv-stage-main";
      if (t.kind === "input") {
        if (c.inputs.length > 4) {
          stageEl.innerHTML = '<div class="cv-frame-grid">' + c.inputs.map(function (x) {
            return '<div class="fr' + (x.used ? " used" : "") + '"><img src="' + esc(x.src) + '" alt=""><span>' + x.index + "</span></div>";
          }).join("") + "</div>";
          capEl.textContent = c.inputs.length + " input frames. Highlighted frames are the ones VERSE chooses to draw on.";
        } else {
          if (c.inputs.length > 1) stageEl.className += " multi n" + c.inputs.length;
          stageEl.innerHTML = c.inputs.map(function (x) { return imgHtml(x.src, c.inputs.length > 1 ? "image " + x.index : ""); }).join("");
          capEl.textContent = "Input" + (c.inputs.length > 1 ? " views" : " image");
        }
        return;
      }
      var step = t.kind === "tool" ? t.step : c.steps[c.steps.length - 1];
      if (step.images.length === 2) stageEl.className += " two";
      stageEl.innerHTML = step.images.map(function (s, k) {
        var f = step.frames[k] != null ? step.frames[k] : step.frames[0];
        return imgHtml(s, step.images.length > 1 || c.bench === "VSI-Bench" ? frameTag(c, f) : "");
      }).join("");
      if (t.kind === "tool") {
        capEl.textContent = "Rendered observation after turn " + t.n + " (" + step.tool + ")";
      } else {
        var ok = correct(c);
        stageEl.insertAdjacentHTML("beforeend", '<div class="answer-overlay ' + (ok ? "good" : "bad") + '">Answer: <b>' +
          esc(c.final.answer) + "</b> " + (ok ? CHECK : CROSS + " (GT " + esc(c.gt) + ")") + "</div>");
        capEl.textContent = "Final answer, given after the last observation";
      }
    }

    function renderProgress(turns) {
      progEl.innerHTML = turns.map(function (t, i) {
        var cls = i < state.turn ? " done" : i === state.turn ? " current" : "";
        return '<div class="seg' + cls + '"></div>';
      }).join("");
      labelEl.textContent = (state.turn + 1) + " / " + turns.length;
      prevBtn.disabled = state.turn === 0;
      nextBtn.disabled = state.turn === turns.length - 1;
    }

    function highlight() {
      var nodes = transEl.querySelectorAll(".turn");
      for (var i = 0; i < nodes.length; i++) nodes[i].classList.toggle("current", i === state.turn);
      var cur = nodes[state.turn];
      if (cur) transEl.scrollTop = Math.max(0, cur.offsetTop - 10);
    }

    function show() {
      var c = current(), turns = turnsOf(c);
      renderStage(c, turns);
      renderProgress(turns);
      highlight();
    }

    function load() {
      var c = current();
      state.turn = 0;
      renderTabs();
      renderChips();
      renderHead(c);
      renderTranscript(c, turnsOf(c));
      preload(c.steps.reduce(function (a, s) { return a.concat(s.images); }, []));
      show();
    }

    function go(i) {
      var n = turnsOf(current()).length;
      state.turn = Math.max(0, Math.min(n - 1, i));
      show();
    }

    tabsEl.addEventListener("click", function (e) {
      var b = e.target.closest(".cv-tab");
      if (!b) return;
      state.bench = b.getAttribute("data-bench");
      state.idx = 0;
      load();
    });
    chipsEl.addEventListener("click", function (e) {
      var b = e.target.closest(".cv-chip");
      if (!b) return;
      state.idx = +b.getAttribute("data-idx");
      load();
    });
    transEl.addEventListener("click", function (e) {
      var t = e.target.closest(".turn");
      if (t) go(+t.getAttribute("data-turn"));
    });
    prevBtn.addEventListener("click", function () { go(state.turn - 1); });
    nextBtn.addEventListener("click", function () { go(state.turn + 1); });

    if (all.length < 2) {
      tabsEl.style.display = "none";
      chipsEl.style.display = "none";
    }
    load();
  }

  /* =====================================================================
   * Shared helpers for the verifier playground.
   * ===================================================================== */
  function setupCanvas(canvas, w, h) {
    var dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function canvasPoint(canvas, w, e) {
    var r = canvas.getBoundingClientRect();
    var s = w / r.width;
    return { x: (e.clientX - r.left) * s, y: (e.clientY - r.top) * s };
  }

  function scoreRow(label, v, color, cls, text) {
    var pct = Math.max(0, Math.min(1, v)) * 100;
    return '<div class="score-row ' + (cls || "") + '"><span class="lbl">' + label + '</span><div class="bar"><div style="width:' +
      pct.toFixed(1) + "%;background:" + color + '"></div></div><span class="val">' + (text != null ? text : v.toFixed(2)) + "</span></div>";
  }

  function rewardRows(phi, answerCorrect) {
    var verse = 0.1 + 0.7 * (answerCorrect ? 1 : 0) + 0.2 * phi;
    var outcome = 0.1 + 0.9 * (answerCorrect ? 1 : 0);
    return '<div class="score-sep"></div>' +
      scoreRow("r<sup>A</sup> VERSE (w=0.2)", verse, "#5b3fa8", "total") +
      scoreRow("r<sup>A</sup> outcome-only (w=0)", outcome, "#d4a514", "outcome");
  }

  /* =====================================================================
   * Object Mapper playground: drag / resize drawn boxes, live IoU.
   * ===================================================================== */
  function BoxPlayground() {
    var canvas = document.getElementById("pg-box");
    if (!canvas) return;
    var W = 520, H = 320;
    var ctx = setupCanvas(canvas, W, H);
    var out = document.getElementById("pg-box-scores");
    var GT = { sofa: [70, 120, 250, 240], lamp: [350, 60, 440, 205] };
    var PRESETS = {
      paper: { sofa: [113, 143, 293, 263], lamp: [380, 215, 480, 290] },
      faithful: { sofa: [74, 124, 246, 238], lamp: [354, 64, 436, 200] },
      omit: { sofa: [74, 124, 246, 238], lamp: null }
    };
    var drawn = {};
    var drag = null;

    function apply(name) {
      var p = PRESETS[name];
      drawn = { sofa: p.sofa && p.sofa.slice(), lamp: p.lamp && p.lamp.slice() };
      render();
    }

    function iou(a, b) {
      if (!a || !b) return 0;
      var ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
      var iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
      var inter = ix * iy;
      var ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
      return ua > 0 ? inter / ua : 0;
    }

    function rr(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function scene() {
      ctx.fillStyle = "#f3f3f6";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#e7e4dc";
      ctx.fillRect(0, 248, W, H - 248);
      ctx.fillStyle = "#8a8f9c";
      ctx.font = "italic 13px 'Noto Sans', sans-serif";
      ctx.fillText("processed frame k", 12, 22);
      // sofa
      ctx.fillStyle = "#b9a48f";
      rr(78, 140, 164, 60, 12); ctx.fill();
      ctx.fillStyle = "#a48e78";
      rr(74, 185, 172, 50, 10); ctx.fill();
      ctx.fillStyle = "#cdb9a3";
      rr(92, 150, 64, 38, 8); ctx.fill();
      rr(164, 150, 64, 38, 8); ctx.fill();
      ctx.fillStyle = "#7b6957";
      ctx.fillRect(84, 233, 8, 8); ctx.fillRect(228, 233, 8, 8);
      // lamp
      ctx.fillStyle = "#e8c66b";
      ctx.beginPath(); ctx.moveTo(368, 112); ctx.lineTo(422, 112); ctx.lineTo(408, 70); ctx.lineTo(382, 70); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#6f6a60";
      ctx.fillRect(393, 112, 4, 82);
      ctx.fillRect(378, 192, 34, 6);
    }

    function box(b, color, dashed, label, fill) {
      ctx.save();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      if (dashed) ctx.setLineDash([7, 5]);
      if (fill) { ctx.fillStyle = fill; ctx.fillRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); }
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.font = "600 12px 'Noto Sans', sans-serif";
      ctx.fillText(label, b[0] + 2, dashed ? b[3] + 15 : b[1] - 6);
      ctx.restore();
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      scene();
      ["sofa", "lamp"].forEach(function (k) {
        var g = GT[k], d = drawn[k];
        if (d) {
          var ix0 = Math.max(g[0], d[0]), iy0 = Math.max(g[1], d[1]), ix1 = Math.min(g[2], d[2]), iy1 = Math.min(g[3], d[3]);
          if (ix1 > ix0 && iy1 > iy0) { ctx.fillStyle = "rgba(47,158,116,0.22)"; ctx.fillRect(ix0, iy0, ix1 - ix0, iy1 - iy0); }
        }
        box(g, "#2f8a63", false, 'GT "' + k + '"');
      });
      ["sofa", "lamp"].forEach(function (k) {
        var d = drawn[k];
        if (!d) return;
        box(d, "#d2452f", true, 'drawn "' + k + '"');
        ctx.fillStyle = "#d2452f";
        ctx.fillRect(d[2] - 5, d[3] - 5, 10, 10);
      });
      var s = iou(drawn.sofa, GT.sofa), l = iou(drawn.lamp, GT.lamp), phi = (s + l) / 2;
      out.innerHTML =
        scoreRow("IoU(sofa)", s, "#2f8a63") +
        scoreRow("IoU(lamp)", l, "#2f8a63", "", drawn.lamp ? null : "0 (omitted)") +
        scoreRow(PHI + "<sub>Mapper</sub>", phi, "#5b3fa8", "total") +
        rewardRows(phi, true) +
        '<div class="score-note">Assumes the final answer is correct and well formatted. The outcome-only reward is 1.00 no matter where the boxes are.</div>';
    }

    function hit(p) {
      var keys = ["lamp", "sofa"];
      for (var i = 0; i < keys.length; i++) {
        var d = drawn[keys[i]];
        if (!d) continue;
        if (Math.abs(p.x - d[2]) < 10 && Math.abs(p.y - d[3]) < 10) return { key: keys[i], mode: "resize" };
      }
      for (i = 0; i < keys.length; i++) {
        d = drawn[keys[i]];
        if (d && p.x >= d[0] && p.x <= d[2] && p.y >= d[1] && p.y <= d[3]) return { key: keys[i], mode: "move" };
      }
      return null;
    }

    canvas.addEventListener("pointerdown", function (e) {
      var p = canvasPoint(canvas, W, e);
      var h = hit(p);
      if (!h) return;
      drag = { key: h.key, mode: h.mode, start: p, orig: drawn[h.key].slice() };
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = h.mode === "move" ? "grabbing" : "nwse-resize";
    });
    canvas.addEventListener("pointermove", function (e) {
      var p = canvasPoint(canvas, W, e);
      if (!drag) {
        var h = hit(p);
        canvas.style.cursor = h ? (h.mode === "move" ? "grab" : "nwse-resize") : "default";
        return;
      }
      var dx = p.x - drag.start.x, dy = p.y - drag.start.y, o = drag.orig, b;
      if (drag.mode === "move") {
        var w = o[2] - o[0], hh = o[3] - o[1];
        var x0 = Math.max(0, Math.min(W - w, o[0] + dx)), y0 = Math.max(0, Math.min(H - hh, o[1] + dy));
        b = [x0, y0, x0 + w, y0 + hh];
      } else {
        b = [o[0], o[1], Math.max(o[0] + 16, Math.min(W, o[2] + dx)), Math.max(o[1] + 16, Math.min(H, o[3] + dy))];
      }
      drawn[drag.key] = b;
      render();
    });
    function end() { drag = null; canvas.style.cursor = "grab"; }
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);

    document.querySelectorAll("[data-box-preset]").forEach(function (btn) {
      btn.addEventListener("click", function () { apply(btn.getAttribute("data-box-preset")); });
    });
    apply("paper");
  }

  /* =====================================================================
   * Path Tracer playground: click cells to draw a route on a 4x4 maze.
   * ===================================================================== */
  function MazePlayground() {
    var canvas = document.getElementById("pg-maze");
    if (!canvas) return;
    var S = 360, N = 4, M = 20, C = (S - 2 * M) / N;
    var ctx = setupCanvas(canvas, S, S);
    var out = document.getElementById("pg-maze-scores");
    var showGT = document.getElementById("pg-show-gt");
    function id(r, c) { return r * N + c; }
    var START = id(0, 0), GOAL = id(3, 3);
    var PASSAGES = [[0, 1], [1, 2], [2, 3], [3, 7], [7, 6], [6, 5], [5, 9], [9, 10], [10, 11], [11, 15],
                    [0, 4], [4, 8], [8, 12], [12, 13], [13, 14]];
    var open = {};
    PASSAGES.forEach(function (p) { open[p[0] + "-" + p[1]] = open[p[1] + "-" + p[0]] = true; });
    function isOpen(a, b) { return !!open[a + "-" + b]; }
    function adjacent(a, b) {
      var ra = Math.floor(a / N), ca = a % N, rb = Math.floor(b / N), cb = b % N;
      return Math.abs(ra - rb) + Math.abs(ca - cb) === 1;
    }

    var GT = (function bfs() {
      var prev = {}, q = [START], seen = {};
      seen[START] = true;
      while (q.length) {
        var u = q.shift();
        if (u === GOAL) break;
        for (var v = 0; v < N * N; v++) {
          if (!seen[v] && adjacent(u, v) && isOpen(u, v)) { seen[v] = true; prev[v] = u; q.push(v); }
        }
      }
      var path = [GOAL];
      while (path[0] !== START) path.unshift(prev[path[0]]);
      return path;
    })();

    var PRESETS = {
      faithful: GT.slice(),
      shortcut: [0, 4, 8, 12, 13, 14, 15],
      partial: GT.slice(0, GT.length - 1),
      reset: [START]
    };
    var path = PRESETS.faithful.slice();

    function center(cell) {
      return { x: M + (cell % N + 0.5) * C, y: M + (Math.floor(cell / N) + 0.5) * C };
    }

    function lcs(a, b) {
      var dp = [];
      for (var i = 0; i <= a.length; i++) { dp.push(new Array(b.length + 1).fill(0)); }
      for (i = 1; i <= a.length; i++) {
        for (var j = 1; j <= b.length; j++) {
          dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
      return dp[a.length][b.length];
    }

    function polyline(cells, color, width, alpha) {
      ctx.save();
      ctx.globalAlpha = alpha || 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      cells.forEach(function (cell, i) {
        var p = center(cell);
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.restore();
    }

    function render() {
      ctx.clearRect(0, 0, S, S);
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, S, S);
      ctx.strokeStyle = "#e3e5ea";
      ctx.lineWidth = 1;
      for (var k = 0; k <= N; k++) {
        ctx.beginPath(); ctx.moveTo(M + k * C, M); ctx.lineTo(M + k * C, S - M); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(M, M + k * C); ctx.lineTo(S - M, M + k * C); ctx.stroke();
      }
      ctx.strokeStyle = "#2b2f38";
      ctx.lineWidth = 6;
      ctx.lineCap = "square";
      for (var r = 0; r < N; r++) {
        for (var c = 0; c < N; c++) {
          var a = id(r, c);
          if (c < N - 1 && !isOpen(a, id(r, c + 1))) {
            ctx.beginPath(); ctx.moveTo(M + (c + 1) * C, M + r * C); ctx.lineTo(M + (c + 1) * C, M + (r + 1) * C); ctx.stroke();
          }
          if (r < N - 1 && !isOpen(a, id(r + 1, c))) {
            ctx.beginPath(); ctx.moveTo(M + c * C, M + (r + 1) * C); ctx.lineTo(M + (c + 1) * C, M + (r + 1) * C); ctx.stroke();
          }
        }
      }
      ctx.strokeRect(M, M, S - 2 * M, S - 2 * M);

      if (showGT.checked) polyline(GT, "#2f8a63", 12, 0.35);

      var s = center(START), g = center(GOAL);
      ctx.fillStyle = "#2f8a63";
      ctx.beginPath(); ctx.arc(s.x, s.y, 13, 0, 2 * Math.PI); ctx.fill();
      ctx.font = "600 12px 'Noto Sans', sans-serif";
      ctx.fillText("start", s.x - 14, s.y - 18);
      ctx.strokeStyle = "#4363d8";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(g.x, g.y, 15, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = "#4363d8";
      ctx.fillText("goal", g.x - 13, g.y + 30);

      var crossings = 0;
      for (var i = 1; i < path.length; i++) {
        var p0 = center(path[i - 1]), p1 = center(path[i]);
        var wall = !isOpen(path[i - 1], path[i]);
        ctx.save();
        ctx.strokeStyle = "#d2452f";
        ctx.lineWidth = 4;
        ctx.lineCap = "round";
        if (wall) { ctx.setLineDash([6, 6]); crossings++; }
        ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
        ctx.restore();
        if (wall) {
          var mx = (p0.x + p1.x) / 2, my = (p0.y + p1.y) / 2;
          ctx.save();
          ctx.fillStyle = "#fff";
          ctx.beginPath(); ctx.arc(mx, my, 10, 0, 2 * Math.PI); ctx.fill();
          ctx.strokeStyle = "#d2452f"; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(mx - 5, my - 5); ctx.lineTo(mx + 5, my + 5); ctx.moveTo(mx + 5, my - 5); ctx.lineTo(mx - 5, my + 5); ctx.stroke();
          ctx.restore();
        }
      }
      path.forEach(function (cell, i) {
        var p = center(cell);
        ctx.fillStyle = i === path.length - 1 ? "#4363d8" : "#d2452f";
        ctx.beginPath(); ctx.arc(p.x, p.y, i === path.length - 1 ? 7 : 4.5, 0, 2 * Math.PI); ctx.fill();
      });

      var l = lcs(path, GT) / GT.length;
      var endHit = path[path.length - 1] === GOAL ? 1 : 0;
      var phi = 0.6 * l + 0.4 * endHit;
      out.innerHTML =
        scoreRow("LCS / |P*|", l, "#2f8a63") +
        scoreRow("endpoint hit", endHit, "#2f8a63", "", endHit ? "1" : "0") +
        scoreRow(PHI + "<sub>Tracer</sub>", phi, "#5b3fa8", "total") +
        rewardRows(phi, !!endHit) +
        '<div class="score-note">The answer counts as correct iff the route ends at the goal. Walls crossed: <b>' + crossings +
        "</b>. " + PHI + "<sub>Tracer</sub> penalizes them implicitly, because such routes share little with the engine path.</div>";
    }

    canvas.addEventListener("click", function (e) {
      var p = canvasPoint(canvas, S, e);
      var c = Math.floor((p.x - M) / C), r = Math.floor((p.y - M) / C);
      if (c < 0 || r < 0 || c >= N || r >= N) return;
      var cell = id(r, c), last = path[path.length - 1];
      var at = path.indexOf(cell);
      if (cell === last && path.length > 1) path.pop();
      else if (at >= 0) path = path.slice(0, at + 1);
      else if (adjacent(last, cell)) path.push(cell);
      render();
    });
    showGT.addEventListener("change", render);
    document.querySelectorAll("[data-maze-preset]").forEach(function (btn) {
      btn.addEventListener("click", function () { path = PRESETS[btn.getAttribute("data-maze-preset")].slice(); render(); });
    });
    render();
  }

  /* =====================================================================
   * Result tables (numbers from the paper) and the overview bar chart.
   * ===================================================================== */
  var VSI_COLS = ["Overall", "Obj. Count", "Abs. Dist.", "Obj. Size", "Room Size", "Rel. Dist.", "Rel. Dir.", "Route Plan", "Appr. Order"];
  var SPAR_COLS = ["Overall", "Low", "Medium", "High"];
  var MAIN_ROWS = [
    ["Qwen2.5-VL-3B", [29.02, 36.31, 20.30, 17.22, 28.02, 34.65, 43.39, 27.84, 24.43, 28.62, 24.07, 27.16, 32.90, 26.70]],
    ["Qwen2.5-VL-7B", [33.68, 43.62, 14.59, 38.64, 33.99, 40.14, 38.43, 27.32, 32.69, 34.12, 25.57, 30.14, 42.95, 33.50]],
    ["Kimi-VL-A3B-Instruct (16B)", [34.14, 32.25, 29.15, 47.74, 41.25, 35.35, 37.29, 29.38, 20.71, 33.75, 27.74, 30.03, 40.33, 49.30]],
    ["LLaVA-OneVision-7B", [31.60, 45.22, 16.29, 40.63, 19.69, 39.58, 33.68, 29.38, 28.32, 30.19, 20.96, 24.33, 40.34, 30.95]],
    ["SpatialThinker-7B", [32.42, 37.66, 22.99, 38.86, 24.06, 37.04, 39.05, 30.93, 28.80, 33.36, 24.08, 25.52, 44.22, 33.00]],
    ["SpatialEvo-7B", [40.06, 53.96, 28.18, 51.45, 36.70, 41.97, 43.70, 26.29, 38.19, 34.94, 28.40, 21.87, 45.11, 32.80]],
    ["SpaceR-7B", [38.43, 42.99, 25.68, 54.46, 33.75, 40.42, 42.25, 26.29, 41.59, 35.33, 28.93, 28.35, 43.35, 35.90]],
    ["Qwen2.5-VL-3B + VERSE", [40.15, 49.84, 29.57, 52.82, 33.58, 35.77, 44.21, 30.41, 44.98, 33.51, 28.23, 28.50, 39.77, 87.50], 0],
    ["Qwen2.5-VL-7B + VERSE", [40.52, 45.10, 30.14, 54.56, 34.69, 44.37, 44.21, 28.87, 42.23, 36.01, 29.00, 32.01, 43.55, 90.80], 1]
  ];

  function rankCols(rows, ncol) {
    var rank = [];
    for (var j = 0; j < ncol; j++) {
      var vals = rows.map(function (r) { return r[1][j]; });
      var uniq = vals.filter(function (v, i) { return vals.indexOf(v) === i; }).sort(function (a, b) { return b - a; });
      rank.push({ best: uniq[0], second: uniq[1] });
    }
    return rank;
  }

  function cell(v, rk, gain, extraCls) {
    var cls = v === rk.best ? "best" : v === rk.second ? "second" : "";
    var g = gain != null ? '<span class="gain">' + (gain >= 0 ? "+" : "") + gain.toFixed(2) + "</span>" : "";
    return '<td class="' + (extraCls || "") + '"><span class="' + cls + '">' + v.toFixed(2) + "</span>" + g + "</td>";
  }

  function buildMainTable() {
    var t = document.getElementById("table-main");
    if (!t) return;
    var rank = rankCols(MAIN_ROWS, 14);
    var h = '<thead><tr><th class="th-left" rowspan="2">Method</th><th class="group divider" colspan="9">VSI-Bench (video)</th>' +
      '<th class="group divider" colspan="4">SPAR-Bench (multi-view)</th><th class="divider" rowspan="2">MAZE</th></tr><tr>' +
      VSI_COLS.map(function (c, i) { return '<th class="' + (i === 0 ? "divider" : "") + '">' + c.replace(" ", "<br>") + "</th>"; }).join("") +
      SPAR_COLS.map(function (c, i) { return '<th class="' + (i === 0 ? "divider" : "") + '">' + c + "</th>"; }).join("") + "</tr></thead><tbody>";
    MAIN_ROWS.forEach(function (row, ri) {
      var ours = row.length > 2;
      var base = ours ? MAIN_ROWS[row[2]][1] : null;
      h += '<tr class="' + (ours ? "ours" : "") + (ri === 7 ? " sep" : "") + '"><td>' + esc(row[0]) + "</td>" +
        row[1].map(function (v, j) {
          return cell(v, rank[j], ours ? v - base[j] : null, j === 0 || j === 9 || j === 13 ? "divider" : "");
        }).join("") + "</tr>";
    });
    t.innerHTML = h + "</tbody>";
  }

  function buildAblationTable() {
    var t = document.getElementById("table-ablation");
    if (!t) return;
    var rows = [
      ["Qwen2.5-VL-7B (untuned base)", [33.68, 43.62, 14.59, 38.64, 33.99, 40.14, 38.43, 27.32, 32.69]],
      ["VERSE (full framework, w = 0.2)", [40.52, 45.10, 30.14, 54.56, 34.69, 44.37, 44.21, 28.87, 42.23], true],
      ["(1) w/o self-evolution (reflective SFT only)", [34.93, 43.71, 22.83, 46.47, 30.58, 39.21, 38.67, 23.79, 34.18]],
      ["(2) w/o ground-truth engine (unverified proposer)", [38.61, 44.71, 27.54, 51.83, 33.08, 42.49, 42.06, 26.02, 41.14]],
      ["(3) w/o process rewards (outcome-only RL, w = 0)", [38.89, 44.59, 28.19, 52.46, 32.94, 42.36, 42.73, 26.84, 40.97]]
    ];
    var rank = rankCols(rows, 9);
    var h = '<thead><tr><th class="th-left">Configuration</th>' + VSI_COLS.map(function (c) { return "<th>" + c.replace(" ", "<br>") + "</th>"; }).join("") + "</tr></thead><tbody>";
    rows.forEach(function (r, i) {
      h += '<tr class="' + (r[2] ? "ours" : "") + (i === 2 ? " sep" : "") + '"><td>' + esc(r[0]) + "</td>" +
        r[1].map(function (v, j) { return cell(v, rank[j]); }).join("") + "</tr>";
    });
    t.innerHTML = h + "</tbody>";
  }

  function simpleTable(id, head, rows, oursIdx) {
    var t = document.getElementById(id);
    if (!t) return;
    t.innerHTML = "<thead><tr>" + head.map(function (c, i) { return "<th" + (i === 0 ? ' class="th-left"' : "") + ">" + c + "</th>"; }).join("") + "</tr></thead><tbody>" +
      rows.map(function (r, i) {
        return '<tr class="' + (oursIdx.indexOf(i) >= 0 ? "ours" : "") + '">' + r.map(function (c) { return "<td>" + c + "</td>"; }).join("") + "</tr>";
      }).join("") + "</tbody>";
  }

  function buildOtherTables() {
    simpleTable("table-vilasr",
      ["Method", "Stage-3 RL source", "RL inputs", "VSI-Bench", "SPAR-Bench", "MAZE", "Mean"],
      [["ViLaSR-7B", "Annotated QA instances", "38,610", "<b>41.91</b>", "35.44", "<b>91.50</b>", "<b>56.28</b>"],
       ["VERSE-7B", "Unlabeled contexts", "<b>1,464</b>", "40.52", "<b>36.01</b>", "90.80", "55.78"]], [1]);
    simpleTable("table-tools",
      ["Protocol", "VSI Overall", "Abs. Dist.", "Obj. Size", "Room Size", "SPAR", "MAZE"],
      [["Direct (no tools)", "33.68", "14.59", "38.64", "33.99", "34.12", "33.50"],
       ["Tools, untrained", "17.00", "4.06", "9.24", "3.13", "25.94", "29.00"],
       ["Tools + VERSE", "<b>40.52</b>", "<b>30.14</b>", "<b>54.56</b>", "<b>34.69</b>", "<b>36.01</b>", "<b>90.80</b>"]], [2]);
    simpleTable("table-behavior",
      ["", "Tools, untrained", "Tools + VERSE"],
      [["No tool call (%) &darr;", "41.3 / 59.0 / 31.8", "<b>0.0 / 0.0 / 0.0</b>"],
       ["Tool calls / trajectory", "2.51 / 1.78 / 3.74", "4.68 / 2.83 / 9.89"],
       ["No final answer (%) &darr;", "18.7 / 11.2 / 4.6", "<b>0.7 / 0.3 / 0.2</b>"]], []);
    simpleTable("table-grounding",
      ["Self-evolution stage", "Mean max-IoU &uarr;", "Hit Rate@0.5 &uarr;", "Coverage (%) &uarr;"],
      [["First 20% of updates", "0.318", "0.364", "60.9"],
       ["Last 20% of updates", "<b>0.411</b>", "<b>0.467</b>", "<b>68.1</b>"],
       ["Relative change", "+29.2%", "+28.3%", "+11.8%"]], [1]);
  }

  function buildBarChart() {
    var host = document.getElementById("bar-chart");
    if (!host) return;
    var series = [
      { name: "Qwen2.5-VL-3B", color: "#cfd3dc", v: [29.02, 28.62, 26.70] },
      { name: "Qwen2.5-VL-3B + VERSE", color: "#a993e0", v: [40.15, 33.51, 87.50] },
      { name: "Qwen2.5-VL-7B", color: "#9aa1b0", v: [33.68, 34.12, 33.50] },
      { name: "Qwen2.5-VL-7B + VERSE", color: "#5b3fa8", v: [40.52, 36.01, 90.80] }
    ];
    var groups = ["VSI-Bench", "SPAR-Bench", "MAZE"];
    var W = 900, H = 290, L = 44, R = 10, T = 14, B = 34;
    var gw = (W - L - R) / groups.length, bw = 44, gap = 10;
    var y = function (v) { return T + (H - T - B) * (1 - v / 100); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Accuracy of base models and VERSE on three benchmarks">';
    [0, 25, 50, 75, 100].forEach(function (t) {
      s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + y(t) + '" y2="' + y(t) + '" stroke="#eceef3"/>' +
        '<text x="' + (L - 8) + '" y="' + (y(t) + 4) + '" font-size="12" text-anchor="end" fill="#8a8f9c">' + t + "</text>";
    });
    groups.forEach(function (g, gi) {
      var x0 = L + gi * gw + (gw - (series.length * bw + (series.length - 1) * gap)) / 2;
      series.forEach(function (se, si) {
        var v = se.v[gi], x = x0 + si * (bw + gap);
        s += '<rect x="' + x + '" y="' + y(v) + '" width="' + bw + '" height="' + (y(0) - y(v)) + '" rx="4" fill="' + se.color + '"/>' +
          '<text x="' + (x + bw / 2) + '" y="' + (y(v) - 5) + '" font-size="11.5" text-anchor="middle" fill="#3a4050" font-weight="' +
          (si % 2 ? 700 : 400) + '">' + v.toFixed(1) + "</text>";
      });
      s += '<text x="' + (L + gi * gw + gw / 2) + '" y="' + (H - 10) + '" font-size="14" text-anchor="middle" fill="#1f2430" font-weight="600">' + g + "</text>";
    });
    s += "</svg>";
    host.innerHTML = '<div class="legend-row">' + series.map(function (se) {
      return '<span><i style="background:' + se.color + '"></i>' + se.name + "</span>";
    }).join("") + "</div>" + s;
  }

  /* ===================================================================== */
  function init() {
    document.querySelectorAll(".case-viewer").forEach(function (el) { CaseViewer(el); });
    BoxPlayground();
    MazePlayground();
    buildBarChart();
    buildMainTable();
    buildAblationTable();
    buildOtherTables();

    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        ignoredClasses: ["case-viewer", "code-block"]
      });
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
