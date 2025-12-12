document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictForm");
    const resultBox = document.getElementById("result");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const jsonData = {};

        // Convert form fields into correct JSON (numbers vs strings)
        formData.forEach((value, key) => {
            const trimmed = value.trim();
            if (!isNaN(trimmed) && trimmed !== "") {
                jsonData[key] = Number(trimmed);
            } else {
                jsonData[key] = trimmed;
            }
        });

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(jsonData)
            });

            const data = await response.json();

            if (data.status === "success") {
                const label = data.predicted_label;
                const probs = data.probabilities || {};
                const riskScore = data.risk_score;
                const riskLevel = data.risk_level || "Unknown";

                // Determine risk pill class
                let riskClass = "low";
                if (riskLevel === "Medium") riskClass = "medium";
                if (riskLevel === "High") riskClass = "high";

                const riskScoreText =
                    typeof riskScore === "number"
                        ? `${(riskScore * 100).toFixed(1)}%`
                        : "N/A";

                // Order classes in a sensible way
                const orderedClasses = [
                    "Fatal injury",
                    "Serious Injury",
                    "Slight Injury"
                ];

                let probListHTML = "";
                orderedClasses.forEach((cls) => {
                    const p = probs[cls] != null ? probs[cls] : 0;
                    const pct = (p * 100).toFixed(2);
                    probListHTML += `
                        <div class="prob-row">
                            <div class="prob-label-row">
                                <span>${cls}</span>
                                <span>${pct}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                            </div>
                        </div>
                    `;
                });

                resultBox.innerHTML = `
                    <div class="result-title">
                        Predicted Severity:
                        <span>${label}</span>
                    </div>

                    <div class="risk-section">
                        <div class="risk-label">Risk Assessment</div>
                        <div class="risk-pill ${riskClass.toLowerCase()}">
                            ${riskLevel} Risk
                        </div>
                        <div class="risk-score-text">
                            Combined Fatal + Serious probability: ${riskScoreText}
                        </div>
                    </div>

                    <div class="prob-header">Class Probabilities</div>
                    <div class="prob-list">
                        ${probListHTML}
                    </div>
                `;

                resultBox.classList.remove("hidden");

            } else {
                resultBox.innerHTML = `
                    <div class="error-box">
                        <strong>Error:</strong> ${data.message || "Unknown error from server."}
                    </div>
                `;
                resultBox.classList.remove("hidden");
            }

        } catch (err) {
            resultBox.innerHTML = `
                <div class="error-box">
                    <strong>Error:</strong> ${err.message}
                </div>
            `;
            resultBox.classList.remove("hidden");
        }
    });
});
