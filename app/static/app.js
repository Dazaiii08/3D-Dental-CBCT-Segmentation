// ---------- Element refs ----------
const fileInput = document.getElementById('fileInput');
const fileLabel = document.getElementById('fileLabel');
const fileLabelText = document.getElementById('fileLabelText');
const fileValidationError = document.getElementById('fileValidationError');
const submitBtn = document.getElementById('submitBtn');

const progressWrap = document.getElementById('progressWrap');
const progressFill = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');

const checklistCard = document.getElementById('checklistCard');
const checklist = document.getElementById('checklist');
const inferencePct = document.getElementById('inferencePct');

const errorCard = document.getElementById('errorCard');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');

const resultsCard = document.getElementById('resultsCard');
const statShape = document.getElementById('statShape');
const statVoxels = document.getElementById('statVoxels');
const statTime = document.getElementById('statTime');
const vizFrame = document.getElementById('vizFrame');
const reportImg = document.getElementById('reportImg');
const downloadNii = document.getElementById('downloadNii');
const downloadReport = document.getElementById('downloadReport');
const resetBtn = document.getElementById('resetBtn');

const offlineBanner = document.getElementById('offlineBanner');
const deviceReadout = document.getElementById('deviceReadout');

const uploadCard = document.getElementById('uploadCard');

let selectedFile = null;
let pollTimer = null;

const STEP_ORDER = ['loading', 'inference', 'postprocess', 'report', 'visualization', 'saving'];

// ---------- Health check ----------
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('bad status');
    const data = await res.json();
    offlineBanner.classList.add('hidden');
    if (data.device) {
      deviceReadout.textContent = data.device.toUpperCase();
    }
    return true;
  } catch (err) {
    offlineBanner.classList.remove('hidden');
    return false;
  }
}

checkHealth();
setInterval(checkHealth, 8000);

// ---------- File selection + validation ----------
function isValidNiiName(name) {
  const lower = name.toLowerCase();
  return lower.endsWith('.nii') || lower.endsWith('.nii.gz');
}

fileInput.addEventListener('change', () => {
  fileValidationError.classList.add('hidden');
  if (fileInput.files.length === 0) return;

  const file = fileInput.files[0];

  if (!isValidNiiName(file.name)) {
    fileValidationError.textContent = `"${file.name}" isn't a .nii or .nii.gz file. Please choose a valid CBCT scan.`;
    fileValidationError.classList.remove('hidden');
    submitBtn.disabled = true;
    selectedFile = null;
    fileLabelText.textContent = 'Choose a .nii CBCT scan';
    return;
  }

  if (file.size === 0) {
    fileValidationError.textContent = 'That file is empty (0 bytes). Please choose a valid scan.';
    fileValidationError.classList.remove('hidden');
    submitBtn.disabled = true;
    selectedFile = null;
    return;
  }

  selectedFile = file;
  fileLabelText.textContent = `${file.name} (${(file.size / 1e6).toFixed(1)} MB)`;
  submitBtn.disabled = false;
});

submitBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  startUpload(selectedFile);
});

retryBtn.addEventListener('click', resetToUpload);
resetBtn.addEventListener('click', resetToUpload);

// ---------- UI state transitions ----------
function resetToUpload() {
  clearInterval(pollTimer);
  errorCard.classList.add('hidden');
  resultsCard.classList.add('hidden');
  checklistCard.classList.add('hidden');
  progressWrap.classList.add('hidden');
  uploadCard.classList.remove('hidden');

  fileInput.value = '';
  selectedFile = null;
  fileLabelText.textContent = 'Choose a .nii CBCT scan';
  fileValidationError.classList.add('hidden');
  submitBtn.disabled = true;
  progressFill.style.width = '0%';
  progressLabel.textContent = 'Uploading… 0%';

  resetChecklist();
  window.scrollTo({ top: uploadCard.offsetTop - 20, behavior: 'smooth' });
}

function resetChecklist() {
  [...checklist.children].forEach(li => {
    li.classList.remove('active', 'done');
  });
  inferencePct.textContent = '';
}

function showError(msg) {
  clearInterval(pollTimer);
  checklistCard.classList.add('hidden');
  progressWrap.classList.add('hidden');
  errorMessage.textContent = msg;
  errorCard.classList.remove('hidden');
  errorCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ---------- Upload (tracked via XHR for real progress %) ----------
function startUpload(file) {
  errorCard.classList.add('hidden');
  resultsCard.classList.add('hidden');
  progressWrap.classList.remove('hidden');
  progressFill.style.width = '0%';
  progressLabel.textContent = 'Uploading… 0%';
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/predict', true);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      progressFill.style.width = pct + '%';
      progressLabel.textContent = `Uploading… ${pct}%`;
    }
  };

  xhr.onload = () => {
    submitBtn.disabled = false;
    if (xhr.status >= 200 && xhr.status < 300) {
      const data = JSON.parse(xhr.responseText);
      progressWrap.classList.add('hidden');
      beginChecklist(data.job_id);
    } else {
      let msg = `Upload failed (status ${xhr.status}).`;
      try {
        const errData = JSON.parse(xhr.responseText);
        if (errData.detail) msg = errData.detail;
      } catch (_) {}
      showError(msg);
    }
  };

  xhr.onerror = () => {
    submitBtn.disabled = false;
    showError("Couldn't reach the backend. Make sure the FastAPI server is still running.");
  };

  xhr.send(formData);
}

// ---------- Real progress polling ----------
function beginChecklist(jobId) {
  resetChecklist();
  checklistCard.classList.remove('hidden');
  checklistCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

  pollTimer = setInterval(() => pollStatus(jobId), 700);
  pollStatus(jobId);
}

async function pollStatus(jobId) {
  try {
    const res = await fetch(`/status/${jobId}`);
    if (!res.ok) throw new Error('status check failed');
    const job = await res.json();
    renderChecklist(job);

    if (job.status === 'done') {
      clearInterval(pollTimer);
      setTimeout(() => showResults(job.result), 400);
    } else if (job.status === 'error') {
      clearInterval(pollTimer);
      showError(job.error || 'Processing failed unexpectedly.');
    }
  } catch (err) {
    clearInterval(pollTimer);
    showError("Lost connection to the backend while processing. It may still be running — check the terminal.");
  }
}

function renderChecklist(job) {
  const currentIndex = STEP_ORDER.indexOf(job.step);

  [...checklist.children].forEach((li) => {
    const step = li.dataset.step;
    const stepIndex = STEP_ORDER.indexOf(step);
    li.classList.remove('active', 'done');

    if (stepIndex < currentIndex) {
      li.classList.add('done');
    } else if (stepIndex === currentIndex) {
      li.classList.add('active');
    }
  });

  if (job.step === 'inference' && typeof job.progress === 'number') {
    inferencePct.textContent = `${job.progress}%`;
  } else if (STEP_ORDER.indexOf('inference') < currentIndex) {
    inferencePct.textContent = '';
  }
}

// ---------- Results + animated counters ----------
function animateCount(el, target, suffix = '', duration = 700) {
  const start = 0;
  const startTime = performance.now();

  function tick(now) {
    const progress = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = Math.round(start + (target - start) * eased);
    el.textContent = value.toLocaleString() + suffix;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function showResults(result) {
  checklistCard.classList.add('hidden');
  uploadCard.classList.add('hidden');

  statShape.textContent = result.volume_shape.join(' × ');
  animateCount(statVoxels, result.tooth_voxels);
  animateCount(statTime, Math.round(result.processing_seconds), 's');

  vizFrame.src = result.html_url;
  reportImg.src = result.report_url;
  downloadNii.href = result.download_nii_url;
  downloadReport.href = result.download_report_url;

  resultsCard.classList.remove('hidden');
  resultsCard.scrollIntoView({ behavior: 'smooth' });
}
