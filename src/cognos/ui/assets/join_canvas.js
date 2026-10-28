/* The join canvas (panels.join_canvas): draws each measured link between its two tables, puts
   the link's chip on the line, and lets the developer drag a table to arrange the canvas.
   Arranging is only a view: what the decision reads are the join fields the chips and the
   tiles' buttons write (app.py, clientside callbacks). Positions are kept per table for the
   page's life, so a refresh of the stage page does not undo them. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const placed = {};

  function tile(cv, name) {
    return cv.querySelector('.jc-tile[data-table="' + CSS.escape(name) + '"]');
  }
  function box(el) {
    return { x: el.offsetLeft + el.offsetWidth / 2, y: el.offsetTop + el.offsetHeight / 2,
             w: el.offsetWidth / 2, h: el.offsetHeight / 2 };
  }
  function edge(from, to) {
    const dx = to.x - from.x, dy = to.y - from.y;
    const k = Math.min(from.w / Math.abs(dx || 1e-6), from.h / Math.abs(dy || 1e-6));
    return [from.x + dx * k, from.y + dy * k];
  }
  function set(el, name, value) {
    if (el.getAttribute(name) !== value) el.setAttribute(name, value);
  }

  function draw(cv) {
    let svg = cv.querySelector("svg.jc-links");
    if (!svg) {
      svg = document.createElementNS(NS, "svg");
      svg.setAttribute("class", "jc-links");
      svg.setAttribute("aria-hidden", "true");
      cv.querySelector(".jc-svg-host").appendChild(svg);
    }
    const baseTile = cv.querySelector(".jc-tile.base");
    const base = baseTile ? baseTile.dataset.table : null;
    cv.querySelectorAll(".jc-chip").forEach(chip => {
      const a = tile(cv, chip.dataset.a), b = tile(cv, chip.dataset.b);
      if (!a || !b) return;
      /* a link that neither touches the base table nor is in the join is not a step anyone
         would take: keep the canvas to the ones that are */
      const other = chip.dataset.a === base ? chip.dataset.b : chip.dataset.b === base ? chip.dataset.a : null;
      const held = c => c !== chip && (c.classList.contains("chosen") || c.classList.contains("proposed"))
                        && (c.dataset.a === other || c.dataset.b === other) && (c.dataset.a === base || c.dataset.b === base);
      const lead = chip.classList.contains("chosen") || chip.classList.contains("proposed");
      /* one lane per table: a merely measured alternative steps aside for the link in use */
      const shown = lead || (other !== null && ![...cv.querySelectorAll(".jc-chip")].some(held));
      chip.hidden = !shown;
      const A = box(a), B = box(b), p = edge(A, B), q = edge(B, A);
      const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
      let path = svg.querySelector('path[data-link="' + chip.dataset.link + '"]');
      if (!path) {
        path = document.createElementNS(NS, "path");
        path.setAttribute("data-link", chip.dataset.link);
        svg.appendChild(path);
      }
      set(path, "d", `M${p[0]},${p[1]} C${p[0]},${my} ${q[0]},${my} ${q[0]},${q[1]}`);
      const kind = ["chosen", "proposed", "measured"].find(k => chip.classList.contains(k)) || "measured";
      set(path, "class", "jc-line " + kind + (shown ? "" : " off"));
      chip.style.left = mx + "px";
      chip.style.top = my + "px";
    });
    unstack([...cv.querySelectorAll(".jc-chip")].filter(c => !c.hidden));
  }

  /* two links can meet in the middle: move a chip down along the canvas until it is clear */
  function unstack(chips) {
    const taken = [];
    chips.sort((a, b) => a.offsetTop - b.offsetTop).forEach(chip => {
      const w = chip.offsetWidth, h = chip.offsetHeight, x = parseFloat(chip.style.left);
      const y0 = parseFloat(chip.style.top);
      let y = y0;
      for (let guard = 0; guard < 20; guard++) {
        const hit = taken.find(r => Math.abs(r.x - x) < (r.w + w) / 2 + 4 && Math.abs(r.y - y) < (r.h + h) / 2 + 3);
        if (!hit) break;
        y = hit.y + (hit.h + h) / 2 + 4;
      }
      taken.push({ x, y, y0, w, h, chip });
    });
    /* chips that started at one height and were stacked: centre the stack on that height */
    const groups = {};
    taken.forEach(r => (groups[Math.round(r.y0 / 12)] = groups[Math.round(r.y0 / 12)] || []).push(r));
    Object.values(groups).forEach(g => {
      const lift = (Math.max(...g.map(r => r.y)) - Math.min(...g.map(r => r.y))) / 2;
      g.forEach(r => { r.chip.style.top = (r.y - lift) + "px"; });
    });
  }

  function drag(cv, t) {
    let sx = 0, sy = 0, ox = 0, oy = 0, on = false;
    t.addEventListener("pointerdown", e => {
      if (e.target.closest("button")) return;
      on = true; t.setPointerCapture(e.pointerId); t.classList.add("dragging");
      sx = e.clientX; sy = e.clientY; ox = t.offsetLeft; oy = t.offsetTop;
    });
    t.addEventListener("pointermove", e => {
      if (!on) return;
      const x = Math.max(0, Math.min(cv.offsetWidth - t.offsetWidth, ox + e.clientX - sx));
      const y = Math.max(0, Math.min(cv.offsetHeight - t.offsetHeight, oy + e.clientY - sy));
      t.style.left = x + "px"; t.style.top = y + "px";
      placed[t.dataset.table] = [x, y];
      draw(cv);
    });
    const stop = () => { on = false; t.classList.remove("dragging"); };
    t.addEventListener("pointerup", stop);
    t.addEventListener("pointercancel", stop);
  }

  function setup(cv) {
    if (cv.dataset.ready) return;
    cv.dataset.ready = "1";
    cv.querySelectorAll(".jc-tile").forEach(t => {
      const at = placed[t.dataset.table];
      if (at) { t.style.left = at[0] + "px"; t.style.top = at[1] + "px"; }
      drag(cv, t);
    });
    /* a chip changes class when the join changes; the line follows it */
    new MutationObserver(() => draw(cv)).observe(cv, { attributes: true, subtree: true,
                                                       attributeFilter: ["class"] });
    draw(cv);
  }

  const scan = () => document.querySelectorAll(".jc-canvas").forEach(setup);
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("resize", () => document.querySelectorAll(".jc-canvas").forEach(draw));
  scan();
})();
