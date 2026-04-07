"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let uploadedFile   = null;   // File from upload tab
let generatedExpr  = null;   // Expression string from generate tab
let currentMode    = "auto"; // OCR mode
let currentCat     = "Арифметика";
let stepsVisible   = true;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dropZone       = $("dropZone");
const fileInput      = $("fileInput");
const uploadPreview  = $("uploadPreview");
const modeSegment    = $("modeSegment");
const solveBtnUpload = $("solveBtnUpload");
const categoryChips  = $("categoryChips");
const exprInput      = $("exprInput");
const randomBtn      = $("randomBtn");
const generateBtn    = $("generateBtn");
const genPreviewWrap = $("genPreviewWrap");
const genPreview     = $("genPreview");
const solveBtnGen    = $("solveBtnGen");
const loader         = $("loader");
const result         = $("result");
const errorBox       = $("errorBox");
const successBox     = $("successBox");
const exprDisplay    = $("exprDisplay");
const answerDisplay  = $("answerDisplay");
const verifiedBadge  = $("verifiedBadge");
const stepsList      = $("stepsList");
const toggleSteps    = $("toggleSteps");
const exampleBtns    = $("exampleBtns");

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => {
      p.classList.remove("active");
      p.classList.add("hidden");
    });
    btn.classList.add("active");
    const panel = $("tab-" + btn.dataset.tab);
    panel.classList.remove("hidden");
    panel.classList.add("active");
    hideResult();
  });
});

// ── OCR mode segmented control ────────────────────────────────────────────────
modeSegment.querySelectorAll(".seg-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    modeSegment.querySelectorAll(".seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.value;
  });
});

// ── Upload tab ────────────────────────────────────────────────────────────────
function setUploadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  uploadedFile = file;
  uploadPreview.src = URL.createObjectURL(file);
  uploadPreview.classList.remove("hidden");
  solveBtnUpload.disabled = false;
  hideResult();
}

fileInput.addEventListener("change", () => setUploadFile(fileInput.files[0]));

dropZone.addEventListener("click", e => {
  if (e.target !== fileInput) fileInput.click();
});
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  setUploadFile(e.dataTransfer.files[0]);
});

solveBtnUpload.addEventListener("click", async () => {
  if (!uploadedFile) return;
  const fd = new FormData();
  fd.append("image", uploadedFile);
  fd.append("mode", currentMode);
  await runSolve(() => fetch("/solve", { method: "POST", body: fd }));
});

// ── Generate tab — categories ─────────────────────────────────────────────────
const EXAMPLES = {
  "Арифметика":  ["15 + 28", "100 - 47", "12 * 8", "144 / 12", "(18 + 7) * 3"],
  "Алгебра":     ["2*x + 3 = 7", "x**2 - 5*x + 6 = 0", "3*x + 1 = 10"],
  "Вычисления":  ["integrate(x**2, x)", "diff(x**3 + 2*x, x)", "integrate(2*x + 1, x)"],
  "Упрощение":   ["(x + 1)**2", "(x - 3)**2", "(x + 2) * (x - 2)"],
};

function renderExamples(cat) {
  exampleBtns.innerHTML = "";
  (EXAMPLES[cat] || []).forEach(expr => {
    const btn = document.createElement("button");
    btn.className = "example-btn";
    btn.textContent = expr;
    btn.addEventListener("click", () => {
      exprInput.value = expr;
      exprInput.focus();
    });
    exampleBtns.appendChild(btn);
  });
}

categoryChips.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    categoryChips.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    currentCat = chip.dataset.cat;
    renderExamples(currentCat);
  });
});
renderExamples(currentCat);  // initial

// ── Random expression ─────────────────────────────────────────────────────────
randomBtn.addEventListener("click", async () => {
  randomBtn.disabled = true;
  try {
    const res  = await fetch("/random");
    const data = await res.json();
    exprInput.value = data.expression;
    // Switch category chip if needed
    categoryChips.querySelectorAll(".chip").forEach(c => {
      c.classList.toggle("active", c.dataset.cat === data.category);
    });
    currentCat = data.category;
    renderExamples(currentCat);
  } catch {
    // silently ignore
  } finally {
    randomBtn.disabled = false;
  }
});

// ── Generate image ────────────────────────────────────────────────────────────
generateBtn.addEventListener("click", async () => {
  const expr = exprInput.value.trim();
  if (!expr) {
    exprInput.focus();
    exprInput.classList.add("shake");
    setTimeout(() => exprInput.classList.remove("shake"), 500);
    return;
  }

  generateBtn.disabled = true;
  generateBtn.textContent = "Генерирую…";

  const fd = new FormData();
  fd.append("expression", expr);
  fd.append("width", 520);
  fd.append("height", 100);

  try {
    const res = await fetch("/generate", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json();
      showError(err.detail || "Ошибка генерации");
      return;
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    genPreview.src = url;
    genPreviewWrap.classList.remove("hidden");
    solveBtnGen.classList.remove("hidden");
    generatedExpr = expr;
    hideResult();
  } catch (e) {
    showError("Не удалось подключиться к серверу: " + e.message);
  } finally {
    generateBtn.disabled = false;
    generateBtn.innerHTML = '<span class="btn-icon">🖼</span> Сгенерировать картинку';
  }
});

// ── Solve generated image ─────────────────────────────────────────────────────
solveBtnGen.addEventListener("click", async () => {
  if (!generatedExpr) return;
  const fd = new FormData();
  fd.append("expression", generatedExpr);
  await runSolve(() => fetch("/solve-expression", { method: "POST", body: fd }));
});

// ── Core solve runner ─────────────────────────────────────────────────────────
async function runSolve(fetchFn) {
  showLoader();
  try {
    const res  = await fetchFn();
    const data = await res.json();

    if (!res.ok) { showError(data.detail || "Ошибка сервера"); return; }
    if (data.error) { showError(data.error); return; }

    showSuccess(data);
  } catch (e) {
    showError("Не удалось подключиться к серверу: " + e.message);
  } finally {
    hideLoader();
  }
}

// ── Toggle steps ──────────────────────────────────────────────────────────────
toggleSteps.addEventListener("click", () => {
  stepsVisible = !stepsVisible;
  stepsList.classList.toggle("collapsed", !stepsVisible);
  toggleSteps.textContent = stepsVisible ? "Скрыть ▲" : "Показать ▼";
});

// ── UI helpers ────────────────────────────────────────────────────────────────
function showLoader() {
  loader.classList.remove("hidden");
  result.classList.add("hidden");
  solveBtnUpload.disabled = true;
  solveBtnGen.disabled    = true;
}
function hideLoader() {
  loader.classList.add("hidden");
  solveBtnUpload.disabled = !uploadedFile;
  solveBtnGen.disabled    = false;
}
function hideResult() {
  result.classList.add("hidden");
  errorBox.classList.add("hidden");
  successBox.classList.add("hidden");
}

function showError(msg) {
  result.classList.remove("hidden");
  errorBox.classList.remove("hidden");
  successBox.classList.add("hidden");
  errorBox.textContent = "Ошибка: " + msg;
}

function showSuccess(data) {
  result.classList.remove("hidden");
  errorBox.classList.add("hidden");
  successBox.classList.remove("hidden");

  exprDisplay.textContent = data.expression || "—";
  answerDisplay.textContent = data.answer || "—";

  if (data.verified === true) {
    verifiedBadge.textContent = "✓ Проверено";
    verifiedBadge.className = "badge ok";
  } else if (data.verified === false) {
    verifiedBadge.textContent = "✗ Не проверено";
    verifiedBadge.className = "badge fail";
  } else {
    verifiedBadge.textContent = "";
    verifiedBadge.className = "badge";
  }

  stepsList.innerHTML = "";
  (data.steps || []).forEach(step => {
    const li = document.createElement("li");
    li.textContent = step;
    stepsList.appendChild(li);
  });
  stepsList.classList.remove("collapsed");
  stepsVisible = true;
  toggleSteps.textContent = "Скрыть ▲";

  result.scrollIntoView({ behavior: "smooth", block: "start" });

  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise([successBox]).catch(console.warn);
  }
}
