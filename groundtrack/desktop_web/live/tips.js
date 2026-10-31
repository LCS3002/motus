/* Live view: a short explanation beside any control held under the pointer for a moment (30 Oct,
   the user: "when i hover long over it explain shortly what it does"). The text is the control's
   title, moved to data-tip as it is set (so the browser's own tooltip never shows as well); a
   label's title explains the checkbox or slider inside it. Hidden by a click, a key, the wheel. */

const DELAY = 650;           // ms under the pointer before it shows
const GAP = 8;               // px between the control and the tip

const tip = document.createElement("div");
tip.className = "tip";
tip.setAttribute("role", "tooltip");
tip.hidden = true;
document.body.appendChild(tip);

function take(el) {
  if (el.nodeType !== 1 || !el.hasAttribute("title")) return;
  const t = el.getAttribute("title");
  if (t) el.dataset.tip = t; else delete el.dataset.tip;
  el.removeAttribute("title");
  if (el === cur && !tip.hidden) tip.textContent = t;          // a live text (Play / Pause) follows
}

function takeAll(root) {
  if (root.nodeType !== 1) return;
  take(root);
  root.querySelectorAll("[title]").forEach(take);
}

let cur = null, timer = 0;

function hide() {
  clearTimeout(timer);
  tip.hidden = true;
  tip.classList.remove("on");
}

function show(el) {
  if (!el.isConnected || !el.dataset.tip) return;
  tip.textContent = el.dataset.tip;
  tip.hidden = false;
  const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
  const W = innerWidth, H = innerHeight;
  let x = r.left, y = r.bottom + GAP;
  if (y + h > H - GAP) y = r.top - GAP - h;                      // no room below: above
  if (y < GAP) {                                                 // nor above (a tall row): beside, left
    y = Math.min(H - GAP - h, Math.max(GAP, r.top));
    x = r.left - GAP - w;
  }
  tip.style.left = `${Math.round(Math.max(GAP, Math.min(W - GAP - w, x)))}px`;
  tip.style.top = `${Math.round(y)}px`;
  requestAnimationFrame(() => tip.classList.add("on"));
}

export function startTips() {
  takeAll(document.body);
  new MutationObserver((ms) => {
    for (const m of ms) {
      if (m.type === "attributes") take(m.target);
      else m.addedNodes.forEach(takeAll);
    }
  }).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["title"] });
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest ? e.target.closest("[data-tip]") : null;
    if (el === cur) return;
    hide();
    cur = el;
    if (el && e.pointerType !== "touch") timer = setTimeout(() => show(el), DELAY);
  });
  document.addEventListener("pointerleave", () => { hide(); cur = null; });
  for (const ev of ["pointerdown", "keydown", "wheel"]) window.addEventListener(ev, hide, { capture: true, passive: true });
  window.addEventListener("blur", hide);
}
