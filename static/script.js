import { Chart } from "@/components/ui/chart"
// CHILYAN PhishGuard - Client-side detection handler

function switchTab(tabName) {
  // Hide all tabs
  document.querySelectorAll(".tab-content").forEach((tab) => {
    tab.classList.remove("active")
  })

  // Deactivate all nav buttons
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.remove("active")
  })

  // Show selected tab
  document.getElementById(tabName).classList.add("active")
  event.target.classList.add("active")

  // Load dashboard data if dashboard tab
  if (tabName === "dashboard") {
    loadDashboard()
  }
}

function selectType(type) {
  // Hide all input sections
  document.querySelectorAll(".input-section").forEach((section) => {
    section.classList.remove("active")
  })

  // Deactivate all type buttons
  document.querySelectorAll(".type-btn").forEach((btn) => {
    btn.classList.remove("active")
  })

  // Show selected type
  document.getElementById(type + "-input").classList.add("active")
  event.target.classList.add("active")
}

async function detectThreat(type) {
  let content = ""

  if (type === "url") {
    content = document.getElementById("urlInput").value
    if (!content) {
      alert("Please enter a URL")
      return
    }
  } else if (type === "email") {
    content = document.getElementById("emailInput").value
    if (!content) {
      alert("Please enter email content")
      return
    }
  } else if (type === "sms") {
    content = document.getElementById("smsInput").value
    if (!content) {
      alert("Please enter SMS content")
      return
    }
  }

  // Show loading
  document.getElementById("loading").style.display = "block"
  document.getElementById("result").style.display = "none"

  try {
    const response = await fetch("/api/detect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        type: type,
        content: content,
      }),
    })

    const result = await response.json()
    displayResult(result)
  } catch (error) {
    console.error("Error:", error)
    displayError("Error analyzing threat. Please try again.")
  } finally {
    document.getElementById("loading").style.display = "none"
  }
}

function displayResult(result) {
  const resultDiv = document.getElementById("result")
  const resultContent = document.getElementById("resultContent")

  let colorClass = "success"
  if (result.threat_level === "MALICIOUS") {
    colorClass = "danger"
  } else if (result.threat_level === "SUSPICIOUS") {
    colorClass = "warning"
  }

  let icon = "✓"
  if (result.threat_level === "MALICIOUS") {
    icon = "⛔"
  } else if (result.threat_level === "SUSPICIOUS") {
    icon = "⚠️"
  }

  const confidencePercent = Math.round(result.confidence * 100)

  let html = `
        <div class="threat-card ${colorClass}">
            <div class="threat-header">
                <div class="threat-level">${icon}</div>
                <div class="threat-info">
                    <h3>${result.threat_level}</h3>
                    <p>${
                      result.threat_level === "MALICIOUS"
                        ? "High-confidence phishing threat detected"
                        : result.threat_level === "SUSPICIOUS"
                          ? "Potential phishing attempt - verify before proceeding"
                          : "Content appears safe"
                    }</p>
                </div>
            </div>
            
            <div>
                <strong>Confidence Score:</strong> ${confidencePercent}%
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                </div>
            </div>
            
            <div class="explanation">
                <h4>Threat Indicators:</h4>
                <ul>
    `

  if (result.explanation && result.explanation.length > 0) {
    result.explanation.forEach((exp) => {
      html += `<li>${exp}</li>`
    })
  } else {
    html += `<li>No specific threat indicators detected</li>`
  }

  html += `
                </ul>
            </div>
            
            <div>
                <strong>Recommended Action:</strong> ${result.recommended_action}
                <button class="action-button" onclick="performAction('${result.recommended_action}')">
                    Execute Action
                </button>
            </div>
        </div>
    `

  resultContent.innerHTML = html
  resultDiv.classList.add("active")
  resultDiv.style.display = "block"
}

function displayError(message) {
  const resultDiv = document.getElementById("result")
  const resultContent = document.getElementById("resultContent")

  resultContent.innerHTML = `<div class="threat-card danger"><h3>Error</h3><p>${message}</p></div>`
  resultDiv.classList.add("active")
  resultDiv.style.display = "block"
}

async function loadDashboard() {
  try {
    const statsResponse = await fetch("/api/stats")
    const stats = await statsResponse.json()

    document.getElementById("totalDetections").textContent = stats.total_detections
    document.getElementById("maliciousCount").textContent = stats.malicious
    document.getElementById("suspiciousCount").textContent = stats.suspicious
    document.getElementById("legitimateCount").textContent = stats.legitimate

    // Threat distribution chart
    const ctx = document.getElementById("threatChart")
    if (ctx && ctx.parentElement.offsetWidth > 0) {
      new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: ["Malicious", "Suspicious", "Legitimate"],
          datasets: [
            {
              data: [stats.malicious, stats.suspicious, stats.legitimate],
              backgroundColor: ["#ff4444", "#ffaa00", "#44aa44"],
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              labels: { color: "#ffffff" },
            },
          },
        },
      })
    }

    // Load recent detections
    const historyResponse = await fetch("/api/history?limit=10")
    const history = await historyResponse.json()

    const recentDiv = document.getElementById("recentDetections")
    if (history.history.length > 0) {
      recentDiv.innerHTML = history.history
        .reverse()
        .map(
          (item) => `
                <div class="detection-item">
                    <strong>${item.threat_level}</strong> - ${item.input_type}<br>
                    <small>${new Date(item.timestamp).toLocaleString()} | Confidence: ${Math.round(item.confidence * 100)}%</small>
                </div>
            `,
        )
        .join("")
    } else {
      recentDiv.innerHTML = "<p>No detections yet</p>"
    }
  } catch (error) {
    console.error("Error loading dashboard:", error)
  }
}

function performAction(action) {
  alert(`Executing action: ${action}`)
  // In production, this would trigger actual response mechanisms
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  selectType("url")
})
