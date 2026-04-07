"use strict";

const API_URL = "/solve";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone   = document.getElementById("dropZone");
const fileInput  = document.getElementById("fileInput");
const preview    = document.getElementById("preview");
const modeSelect = document.getElementById("modeSelect");
const solveBtn   = document.getElementById("solveBtn");
const loader     = document.getElementById("loader");
const result     = document.getElementById("result");
const errorBox   = document.getElementById("errorBox");
const successBox = document.getElementById("successBox");
const exprDisplay   = document.getElementById("exprDisplay");
const answerDisplay = document.getElementById("answerDisplay");
const verifiedBadge = document.getElementById("verifiedBadge");
const stepsList     = document.getElementById("stepsList");

let selectedFile = null;

// ── File selection ────────────────────────────────────────────────────────────
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.classList.remove("hidden");
  solveBtn.disabled = false;
  hideResult();
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

// Click on drop zone opens file picker
dropZone.addEventListener("click", (e) => {
  if (e.target !== fileInput) fileInput.click();
});

// Drag & drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  setFile(e.dataTransfer.files[0]);
});

// ── Solve ─────────────────────────────────────────────────────────────────────
solveBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  showLoader();

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("mode", modeSelect.value);

  try {
    const res = await fetch(API_URL, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || "Сталася помилка сервера");
      return;
    }

    if (data.error) {
      showError(data.error);
      return;
    }

    showSuccess(data);
  } catch (err) {
    showError("Не вдалося з'єднатися з сервером: " + err.message);
  } finally {
    hideLoader();
  }
});

// ── UI helpers ────────────────────────────────────────────────────────────────
function showLoader() {
  loader.classList.remove("hidden");
  result.classList.add("hidden");
  solveBtn.disabled = true;
}

function hideLoader() {
  loader.classList.add("hidden");
  solveBtn.disabled = false;
}

function hideResult() {
  result.classList.add("hidden");
}

function showError(msg) {
  result.classList.remove("hidden");
  errorBox.classList.remove("hidden");
  successBox.classList.add("hidden");
  errorBox.textContent = "Помилка: " + msg;
}

function showSuccess(data) {
  result.classList.remove("hidden");
  errorBox.classList.add("hidden");
  successBox.classList.remove("hidden");

  exprDisplay.textContent = data.expression || "—";

  answerDisplay.textContent = data.answer || "—";

  if (data.verified === true) {
    verifiedBadge.textContent = "Перевірено ✓";
    verifiedBadge.className = "badge ok";
  } else if (data.verified === false) {
    verifiedBadge.textContent = "Не перевірено ✗";
    verifiedBadge.className = "badge fail";
  } else {
    verifiedBadge.textContent = "";
    verifiedBadge.className = "badge";
  }

  stepsList.innerHTML = "";
  (data.steps || []).forEach((step) => {
    const li = document.createElement("li");
    li.textContent = step;
    stepsList.appendChild(li);
  });

  // Re-render MathJax if available
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([successBox]).catch(console.warn);
  }
}
