const form = document.getElementById("upload-form");
const input = document.getElementById("video-input");
const conditionSelect = document.getElementById("condition-select");
const classifierSelect = document.getElementById("classifier-select");
const liveFeedToggle = document.getElementById("live-feed-toggle");
const statusEl = document.getElementById("status");
const pipelineBadge = document.getElementById("pipeline-badge");
const resultsEl = document.getElementById("results");
const submitButton = document.getElementById("submit-button");
const annotatedVideo = document.getElementById("annotated-video");
const analysisSummary = document.getElementById("analysis-summary");
const summaryGrid = document.getElementById("summary-grid");
const conditionAnomalies = document.getElementById("condition-anomalies");
const modelStack = document.getElementById("model-stack");
const timeline = document.getElementById("timeline");
const analystChat = document.getElementById("analyst-chat");
const notes = document.getElementById("notes");
const downloadAnnotated = document.getElementById("download-annotated");
const downloadTimeline = document.getElementById("download-timeline");
const openAnnotated = document.getElementById("open-annotated");
const annotatedVideoStatus = document.getElementById("annotated-video-status");
const selectedFileName = document.getElementById("selected-file-name");

const formatSeconds = (value) => `${value.toFixed(2)}s`;
const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

if (input && selectedFileName) {
  input.addEventListener("change", () => {
    selectedFileName.textContent = input.files.length
      ? input.files[0].name
      : "No file selected";
  });
}

function updateVideoStatus(element, message = "", isError = false) {
  element.textContent = message;
  element.classList.toggle("error", Boolean(message && isError));
}

function prepareVideoPlayer(videoEl, statusEl, url, label) {
  updateVideoStatus(statusEl, `Loading ${label.toLowerCase()}...`);
  videoEl.pause();
  videoEl.removeAttribute("src");
  videoEl.load();
  videoEl.src = url;
  videoEl.load();
}

annotatedVideo.addEventListener("loadedmetadata", () => {
  updateVideoStatus(annotatedVideoStatus, "Annotated video ready.");
  annotatedVideo.currentTime = 0;
});

annotatedVideo.addEventListener("error", () => {
  updateVideoStatus(
    annotatedVideoStatus,
    "This browser could not render the annotated video inline. Use the Open file or Download link above.",
    true
  );
});

async function submitVideo(fileLike) {
  const formData = new FormData();
  formData.append("file", fileLike);
  formData.append("condition", conditionSelect?.value || "");
  formData.append("classifier_type", classifierSelect?.value || "adaptive_hmm");

  const response = await fetch("/api/analysis/upload-video", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Analysis failed.");
  }
  return payload;
}

async function submitVideoStream(fileLike) {
  const formData = new FormData();
  formData.append("file", fileLike);
  formData.append("condition", conditionSelect?.value || "");
  formData.append("classifier_type", classifierSelect?.value || "adaptive_hmm");

  const response = await fetch("/api/analysis/upload-video-stream", {
    method: "POST",
    body: formData,
  });

  if (!response.ok || !response.body) {
    let detail = "Streaming analysis failed.";
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);

      if (event.type === "progress") {
        statusEl.textContent = `Analyzing... processed ${event.processed_frames} frames (${formatSeconds(event.timestamp_s)})`;
      } else if (event.type === "status") {
        const pipelineMode = event.pipeline_mode ? event.pipeline_mode.replaceAll("_", " ") : "updating";
        pipelineBadge.textContent = `Pipeline: ${pipelineMode}. ${event.message}`;
      } else if (event.type === "event") {
        appendAnalystMessage({
          speaker: "AI Observer",
          timestamp_s: event.timestamp_s,
          message: `${event.label} detected with confidence ${event.confidence.toFixed(2)}. ${event.message}`,
        });
      } else if (event.type === "result") {
        return event.payload;
      } else if (event.type === "error") {
        throw new Error(event.detail || "Streaming analysis failed.");
      }
    }
  }

  throw new Error("Streaming analysis ended without a final result.");
}

function appendAnalystMessage(entry) {
  const empty = analystChat.querySelector(".status");
  if (empty) empty.remove();

  const item = document.createElement("article");
  item.className = "timeline-item";
  item.innerHTML = `<strong>${entry.speaker}</strong> at ${formatSeconds(entry.timestamp_s)}<br />${entry.message}`;
  analystChat.appendChild(item);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!input.files.length) {
    statusEl.textContent = "Choose a video first.";
    return;
  }

  submitButton.disabled = true;
  statusEl.textContent = "Uploading and analyzing video. This can take a minute for longer clips.";
  pipelineBadge.textContent = "Pipeline: initializing";
  analystChat.innerHTML = "<p class='status'>AI observer is waiting for the first event.</p>";
  updateVideoStatus(annotatedVideoStatus, "");

  try {
    const payload = liveFeedToggle.checked
      ? await submitVideoStream(input.files[0])
      : await submitVideo(input.files[0]);

    renderResults(payload);
    statusEl.textContent = "Analysis complete.";
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    submitButton.disabled = false;
  }
});

function renderResults(payload) {
  resultsEl.classList.remove("hidden");

  const cacheBust = `t=${Date.now()}`;
  const annotatedUrl = `${payload.annotated_video_url}?${cacheBust}`;
  prepareVideoPlayer(annotatedVideo, annotatedVideoStatus, annotatedUrl, "Annotated video");

  downloadAnnotated.href = payload.annotated_video_url;
  downloadTimeline.href = payload.timeline_url;
  openAnnotated.href = payload.annotated_video_url;

  const summaryItems = [
    ["Analysis ID", payload.analysis_id],
    ["Pipeline", payload.pipeline_mode.replaceAll("_", " ")],
    ["Classifier", payload.classifier_mode.replaceAll("_", " ")],
    ["Duration", formatSeconds(payload.duration_s)],
    ["FPS", payload.fps.toFixed(2)],
    ["Frames", String(payload.frame_count)],
    ["Dominant", payload.dominant_behavior],
    ["Review", payload.review_priority],
  ];
  if (payload.condition) {
    summaryItems.push(["Condition", payload.condition]);
  }

  analysisSummary.textContent = payload.analysis_summary;
  pipelineBadge.textContent = `Pipeline: ${payload.pipeline_mode.replaceAll("_", " ")}`;

  summaryGrid.innerHTML = summaryItems
    .map(
      ([label, value]) => `
        <div class="summary-card">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `
    )
    .join("");

  if (conditionAnomalies) {
    const anomalies = payload.condition_anomalies || [];
    conditionAnomalies.innerHTML = anomalies.length
      ? anomalies
          .map(
            (anomaly) => `
              <span class="chip ${anomaly.severity === "high" ? "stereotypy_candidate" : "hesitation"}">
                ${anomaly.behavior}: ${anomaly.direction.replaceAll("_", " ")} (${formatPercent(anomaly.observed)})
              </span>
            `
          )
          .join("")
      : payload.condition
        ? `<span class="chip exploration">No condition anomalies vs ${payload.condition} norms</span>`
        : "";
  }

  behaviorShare.innerHTML = Object.entries(payload.behavior_share)
    .map(
      ([label, share]) => `
        <span class="chip ${label}">${label}: ${formatPercent(share)}</span>
      `
    )
    .join("");

  modelStack.innerHTML = payload.model_stack
    .map((item) => `<span class="chip">${item.replaceAll("_", " ")}</span>`)
    .join("");

  notes.innerHTML = payload.notes
    .concat(
      payload.condition_report
        ? [
            `condition summary: ${payload.condition_report.condition} (${payload.condition_report.anomaly_count} anomalies)`,
          ]
        : []
    )
    .concat(
      (payload.condition_anomalies || []).map(
        (anomaly) =>
          `condition anomaly: ${anomaly.behavior} ${anomaly.direction.replaceAll("_", " ")} (${formatPercent(anomaly.observed)} observed vs ${formatPercent(anomaly.expected_range[0])}-${formatPercent(anomaly.expected_range[1])})`
      )
    )
    .concat(
      Object.entries(payload.selected_models || {}).map(
        ([role, model]) => `${role.replaceAll("_", " ")}: ${model}`
      )
    )
    .map((note) => `<li>${note}</li>`)
    .join("");

  timeline.innerHTML = payload.bouts.length
    ? payload.bouts
        .map(
          (bout) => `
            <article class="timeline-item ${bout.label}">
              <strong>${bout.label}</strong><br />
              ${formatSeconds(bout.start_s)} to ${formatSeconds(bout.end_s)}<br />
              Duration ${formatSeconds(bout.duration_s)} | Confidence ${bout.confidence.toFixed(2)} | Mean speed ${bout.mean_speed_px_s.toFixed(1)} px/s
            </article>
          `
        )
        .join("")
    : "<p class='status'>No strong bouts crossed the current thresholds in this clip.</p>";

  analystChat.innerHTML = payload.analyst_messages.length
    ? payload.analyst_messages
        .map(
          (entry) => `
            <article class="timeline-item">
              <strong>${entry.speaker}</strong> at ${formatSeconds(entry.timestamp_s)}<br />
              ${entry.message}
            </article>
          `
        )
        .join("")
    : "<p class='status'>No analyst messages were generated for this clip.</p>";
}
