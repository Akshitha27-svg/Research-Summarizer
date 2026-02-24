// Load papers dynamically
window.onload = async function () {

    const response = await fetch("http://127.0.0.1:8000/papers");
    const data = await response.json();

    const select = document.getElementById("paperSelect");

    data.papers.forEach(paper => {
        const option = document.createElement("option");
        option.value = paper;
        option.textContent = paper;
        select.appendChild(option);
    });

    updatePDF();
};

document.getElementById("paperSelect").addEventListener("change", updatePDF);

function updatePDF() {
    const selectedPaper = document.getElementById("paperSelect").value;
    document.getElementById("pdfViewer").src =
        `http://127.0.0.1:8000/pdfs/${selectedPaper}`;
}

async function askQuestion() {

    const question = document.getElementById("question").value;
    const selectedPaper = document.getElementById("paperSelect").value;
    const resultDiv = document.getElementById("result");

    resultDiv.innerHTML = "Generating...";
    document.getElementById("confidenceContainer").style.display = "none";

    const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            question: question,
            selected_paper: selectedPaper
        })
    });

    const data = await response.json();

    if (data.out_of_context) {
        resultDiv.innerHTML =
            `<p style="color:red; font-weight:bold;">
            ${data.message}
            </p>`;
        return;
    }

    resultDiv.innerHTML = `<h3>Generated Answer</h3><p>${data.summary}</p>`;

    // Animate confidence bar
    const confidenceBar = document.getElementById("confidenceBar");
    const confidenceText = document.getElementById("confidenceText");
    const container = document.getElementById("confidenceContainer");

    container.style.display = "block";

    setTimeout(() => {
        confidenceBar.style.width = data.confidence + "%";
    }, 100);

    confidenceText.innerHTML =
        `Confidence Score: ${data.confidence}%`;
}