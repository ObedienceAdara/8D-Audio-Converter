const form = document.getElementById("convert-form");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");
const metricsBox = document.getElementById("metrics");
const download = document.getElementById("download");

function showStatus(message) {
  statusBox.hidden = false;
  statusBox.textContent = message;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultBox.hidden = true;
  const body = new FormData(form);
  showStatus("Uploading and queueing job…");
  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to create job.");
    await poll(data.status_url);
  } catch (error) {
    showStatus(`Error: ${error.message}`);
  }
});

async function poll(url) {
  for (;;) {
    const response = await fetch(url);
    const data = await response.json();
    if (data.status === "queued") {
      showStatus("Job queued…");
    } else if (data.status === "processing") {
      showStatus("Processing audio…");
    } else if (data.status === "failed") {
      showStatus(`Conversion failed: ${data.error}`);
      return;
    } else if (data.status === "completed") {
      showStatus("Conversion complete.");
      resultBox.hidden = false;
      metricsBox.textContent = JSON.stringify(data.metrics, null, 2);
      download.href = data.download_url;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
}
