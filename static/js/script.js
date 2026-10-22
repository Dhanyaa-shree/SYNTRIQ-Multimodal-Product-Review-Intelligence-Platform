let currentAnalysisData = null; 

function runAnalysis() {
    const form = document.getElementById('analysisForm');
    const formData = new FormData(form);
    
    const btn = document.querySelector('.btn-primary');
    const originalText = btn.innerText;
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    fetch('/api/analyze', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
        if (data.error) { alert(data.error); return; }
        
        currentAnalysisData = data; 
        
        document.getElementById('resultsPanel').style.display = 'block';
        document.getElementById('res-image').src = data.image_url;
        
        const setText = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.innerText = text;
        };

        setText('res-confidence', data.confidence);
        setText('res-consistency', data.consistency);
        setText('res-summary', data.summary);
        setText('res-aspect', data.aspect);
        setText('res-intent', data.intent || "Unknown");
        setText('res-severity', data.severity);
        setText('res-defect', data.defect_type || "None");
        setText('res-explanation', data.explanation || "No explanation provided.");
        setText('res-action', data.action || "No action recommended.");

        const sentEl = document.getElementById('res-sentiment');
        const confValue = parseFloat(data.confidence);
        const warningEl = document.getElementById('low-confidence-warning');

        if (sentEl) {
            sentEl.innerText = data.sentiment;
            sentEl.className = 'badge ' + (data.sentiment === 'Positive' ? 'green' : data.sentiment === 'Negative' ? 'red' : 'gray');
        }

        if (confValue < 60.0) {
            warningEl.style.display = 'block';
        } else {
            warningEl.style.display = 'none';
        }

        // --- DRAW BOUNDING BOX ---
        const defectBox = document.getElementById('defect-box');
        
        // We draw the box if the backend sent coordinates
        if (data.box) {
            defectBox.style.top = `${data.box.top}%`;
            defectBox.style.left = `${data.box.left}%`;
            defectBox.style.width = `${data.box.width}%`;
            defectBox.style.height = `${data.box.height}%`;
            defectBox.style.display = 'block';
            
            defectBox.innerHTML = `<span style="background: red; color: white; font-size: 10px; padding: 2px 5px; position: absolute; top: -20px; left: -3px; font-weight: bold;">${data.box.label.toUpperCase()}</span>`;
        } else {
            defectBox.style.display = 'none';
        }
    })
    .catch(err => {
        console.error("Fetch Error:", err);
        alert("Error: " + err);
    })
    .finally(() => {
        btn.innerText = originalText;
        btn.disabled = false;
    });
}

function downloadReport() {
    if (!currentAnalysisData) {
        alert("Please analyze a review first.");
        return;
    }

    const data = currentAnalysisData;
    const dateStr = new Date().toLocaleString();

    const reportContent = `
==================================================
SYNTRIQ - MULTIMODAL PRODUCT REVIEW REPORT
Date: ${dateStr}
==================================================

AI PREDICTIONS
--------------------------------------------------
Sentiment: ${data.sentiment}
Confidence Score: ${data.confidence}
Evidence Match: ${data.consistency}

LLM BUSINESS INSIGHTS
--------------------------------------------------
Summary: ${data.summary}
Aspect: ${data.aspect}
Intent: ${data.intent}
Issue Severity: ${data.severity}
Defect Type: ${data.defect_type}

EXPLANATION
--------------------------------------------------
${data.explanation}

RECOMMENDED ACTION
--------------------------------------------------
${data.action}

==================================================
Generated automatically by SYNTRIQ AI Engine.
    `.trim();

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `SYNTRIQ_Report_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
}