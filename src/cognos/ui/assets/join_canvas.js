/* The join canvas (panels.join_canvas): draws each measured link between its two tables and
   puts the link's chip on the line. The developer can
   - click a chip to open the link's settings: in the join or not, what to do when a key
     matches several rows, and base rows with no match;
   - drag from a column of one table to a column of another to draw a link: the server
     measures it (/api/measure-link) before it appears, and it joins like any other;
   - make a table the base table, and drag a table by its name to arrange the canvas.
   The join as the canvas states it is kept in the data gate's hidden join_spec field
   ({base, links, rules, drawn}), which the decision reads. Arranging is only a view;
   positions are kept per table for the page's life. */
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
      const lead = ["chosen", "proposed", "drawn", "sel"].some(k => chip.classList.contains(k));
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
      set(path, "class", "jc-line " + kind + (chip.classList.contains("drawn") ? " drawn" : "")
                        + (chip.classList.contains("sel") ? " sel" : "") + (shown ? "" : " off"));
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
      if (!cv.classList.contains("locked") && e.target.closest("li[data-col]")) return;
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
    if (!cv.classList.contains("locked")) edit(cv);
    draw(cv);
  }

  /* ---- editing: the join as the canvas states it ---- */
  const FIELD = { type: "gate-field", gate: "gate_data", field: "join_spec" };

  function spec(cv) {
    if (!cv._spec) {
      const field = cv.closest(".jc").querySelector("input[type=text], input:not([type])");
      try { cv._spec = JSON.parse(field.value); } catch (e) { cv._spec = null; }
      cv._spec = cv._spec || { base: (cv.querySelector(".jc-tile.base") || {}).dataset?.table,
                               links: [], rules: {}, drawn: [] };
    }
    return cv._spec;
  }
  function commit(cv) {
    const value = JSON.stringify(spec(cv));
    if (window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props(FIELD, { value: value });
    }
  }
  function chipState(cv, chip) {
    const on = spec(cv).links.includes(chip.dataset.link);
    chip.classList.toggle("chosen", on);
    if (!on && !chip.classList.contains("proposed") && !chip.classList.contains("drawn")) {
      chip.classList.add("measured");
    } else if (on) {
      chip.classList.remove("measured");
    }
  }
  function el(tag, attrs, kids) {
    const e = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === "text") e.textContent = v; else if (k === "cls") e.className = v; else e.setAttribute(k, v);
    });
    (kids || []).forEach(k => k && e.appendChild(k));
    return e;
  }
  function select(label, options, value, onchange) {
    const s = el("select", {}, options.map(([v, t]) => {
      const o = el("option", { value: v, text: t });
      if (v === value) o.selected = true;
      return o;
    }));
    s.addEventListener("change", () => onchange(s.value));
    return el("label", { cls: "jc-field" }, [el("span", { text: label }), s]);
  }

  function inspect(cv, chip, error) {
    const host = cv.closest(".jc");
    let box = host.querySelector(".jc-insp");
    if (box) box.remove();
    cv.querySelectorAll(".jc-chip.sel").forEach(c => c.classList.remove("sel"));
    if (!chip) return;
    chip.classList.add("sel");
    const id = chip.dataset.link, st = spec(cv), rule = st.rules[id] || {};
    const on = st.links.includes(id);
    const close = el("button", { cls: "jc-insp-x", text: "×", "aria-label": "Close" });
    close.addEventListener("click", () => inspect(cv, null));
    const kids = [
      el("div", { cls: "jc-insp-head" }, [el("b", { text: chip.dataset.a + " ↔ " + chip.dataset.b }), close]),
      el("div", { cls: "jc-insp-on", text: "on " + chip.dataset.on }),
      el("div", { cls: "jc-insp-m", text: chip.dataset.match + " of rows match · " + chip.dataset.rel }),
    ];
    if (chip.dataset.by) kids.push(el("div", { cls: "jc-insp-m", text: "drawn by " + chip.dataset.by }));
    if (chip.dataset.low === "1") kids.push(el("div", { cls: "jc-insp-warn", text: "Fewer than half the rows match: check the columns." }));
    if (error) kids.push(el("div", { cls: "jc-insp-warn", text: error }));
    const tick = el("input", { type: "checkbox" });
    tick.checked = on;
    tick.addEventListener("change", () => {
      st.links = tick.checked ? st.links.concat([id]) : st.links.filter(v => v !== id);
      chipState(cv, chip); commit(cv); inspect(cv, chip);
    });
    kids.push(el("label", { cls: "jc-check" }, [tick, el("span", { text: "In the join" })]));
    if (on) {
      if (chip.dataset.rel.includes("many")) {
        kids.push(select("When a key matches several rows",
          [["aggregate", "Average them (numbers)"], ["first", "Keep the first row"]],
          rule.many || chip.dataset.many || "aggregate",
          v => { st.rules[id] = Object.assign({}, st.rules[id], { many: v }); commit(cv); }));
      }
      const now = chip.dataset.unmatched;
      kids.push(select("Base rows with no match",
        [["", "As recommended" + (now ? " (" + now + ")" : "")],
         ["keep", "Keep them, with missing values"], ["drop", "Drop them (counted)"]],
        rule.unmatched || "",
        v => { st.rules[id] = Object.assign({}, st.rules[id], { unmatched: v || undefined }); commit(cv); }));
    }
    box = el("div", { cls: "jc-insp", role: "dialog", "aria-label": "Link settings" }, kids);
    host.appendChild(box);
  }

  function edit(cv) {
    const st = spec(cv);
    cv.querySelectorAll(".jc-chip").forEach(chip => wireChip(cv, chip));
    cv.querySelectorAll(".jc-make-base").forEach(btn => btn.addEventListener("click", () => {
      const t = btn.closest(".jc-tile");
      st.base = t.dataset.table;
      cv.querySelectorAll(".jc-tile").forEach(x => {
        x.classList.toggle("base", x === t);
        x.querySelector(".jc-make-base").textContent = x === t ? "base table" : "Make base";
      });
      commit(cv); draw(cv);
    }));
    /* draw a link: from a column of one table to a column of another */
    cv.querySelectorAll(".jc-tile li[data-col]").forEach(li => li.addEventListener("pointerdown", e => {
      e.preventDefault();
      const from = li.closest(".jc-tile"), r0 = cv.getBoundingClientRect();
      const svg = cv.querySelector("svg.jc-links");
      const line = document.createElementNS(NS, "path");
      line.setAttribute("class", "jc-line drawing");
      svg.appendChild(line);
      li.classList.add("jc-from");
      const lr = li.getBoundingClientRect();
      const sx = lr.right - r0.left, sy = lr.top + lr.height / 2 - r0.top;
      let over = null;
      const move = ev => {
        const x = ev.clientX - r0.left, y = ev.clientY - r0.top;
        line.setAttribute("d", `M${sx},${sy} C${(sx + x) / 2},${sy} ${(sx + x) / 2},${y} ${x},${y}`);
        const hit = document.elementFromPoint(ev.clientX, ev.clientY);
        const to = hit && hit.closest && hit.closest(".jc-tile li[data-col]");
        if (over && over !== to) over.classList.remove("jc-to");
        over = to && to.closest(".jc-tile") !== from ? to : null;
        if (over) over.classList.add("jc-to");
      };
      const up = () => {
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", up);
        line.remove(); li.classList.remove("jc-from");
        if (over) { over.classList.remove("jc-to"); propose(cv, li, over); }
      };
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", up);
    }));
  }

  function wireChip(cv, chip) {
    chip.addEventListener("click", e => {
      e.stopPropagation();
      inspect(cv, chip.classList.contains("sel") ? null : chip);
    });
  }

  function propose(cv, fromLi, toLi) {
    const st = spec(cv);
    const a = fromLi.closest(".jc-tile").dataset.table, b = toLi.closest(".jc-tile").dataset.table;
    const a_on = [fromLi.dataset.col], b_on = [toLi.dataset.col];
    const same = st.drawn.find(d => d.a === a && d.b === b && d.a_on[0] === a_on[0] && d.b_on[0] === b_on[0]);
    if (same) { inspect(cv, cv.querySelector('.jc-chip[data-link="' + same.id + '"]')); return; }
    const run = (location.pathname.split("/run/")[1] || "").split("/")[0];
    const id = "D" + (Number(cv.dataset.drawn || 0) + st.drawn.length + 1);
    const chip = el("button", { cls: "jc-chip drawn measuring", text: "measuring…", type: "button",
                                "data-a": a, "data-b": b, "data-link": id,
                                "data-on": a_on[0] === b_on[0] ? a_on[0] : a_on[0] + " = " + b_on[0],
                                "data-by": "you", "data-match": "", "data-rel": "" });
    cv.appendChild(chip);
    draw(cv);
    fetch("/api/measure-link", { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify({ run, a, a_on, b, b_on }) })
      .then(r => r.json())
      .then(res => {
        if (!res.ok) { chip.remove(); draw(cv); flash(cv, res.error || "The link cannot be measured."); return; }
        chip.classList.remove("measuring");
        chip.textContent = res.chip;
        chip.dataset.match = res.match; chip.dataset.rel = res.rel; chip.dataset.low = res.low ? "1" : "";
        st.drawn.push({ id, a, a_on, b, b_on });
        st.links.push(id);
        chipState(cv, chip);
        wireChip(cv, chip);
        commit(cv); draw(cv); inspect(cv, chip);
      })
      .catch(() => { chip.remove(); draw(cv); flash(cv, "The server did not answer; try again."); });
  }

  function flash(cv, text) {
    const host = cv.closest(".jc");
    const old = host.querySelector(".jc-flash");
    if (old) old.remove();
    const note = el("div", { cls: "jc-flash", role: "status", text });
    host.appendChild(note);
    setTimeout(() => note.remove(), 6000);
  }

  const scan = () => document.querySelectorAll(".jc-canvas").forEach(setup);
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("resize", () => document.querySelectorAll(".jc-canvas").forEach(draw));
  scan();
})();
