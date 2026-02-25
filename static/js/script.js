document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const resultDiv = document.getElementById("result");
  const previewImage = document.getElementById("previewImage");
  const diseaseLabel = document.getElementById("diseaseLabel");
  const confidence = document.getElementById("confidence");
  const treatmentText = document.getElementById("treatmentText");

  uploadBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("language", document.getElementById("language").value); // Add selected language

    // ✅ Instead of removing child elements, just update text
    resultDiv.style.display = "block";
    diseaseLabel.textContent = "Analyzing...";
    confidence.textContent = "";
    treatmentText.textContent = "";
    previewImage.src = "";

    try {
      const response = await fetch("/predict", { method: "POST", body: formData });
      const data = await response.json();
      
      // ---- after displaying prediction ----
      if (data.error) {
        diseaseLabel.textContent = "Error";
        confidence.textContent = "";
        treatmentText.textContent = data.error;
      } else {
        // ✅ Safely update existing DOM elements
        previewImage.src = `data:image/jpeg;base64,${data.image_data}`;
        diseaseLabel.textContent = data.label;
        confidence.textContent = `${data.confidence}`;
        
        console.log("Treatment string received by frontend:", data.treatment); // Debugging line
        // Split the treatment text by newline and create a list
        const treatmentPoints = data.treatment.split('\n').filter(point => point.trim() !== '');
        let treatmentHtml = '';
        if (treatmentPoints.length > 0) {
          treatmentHtml = '<ul>' + treatmentPoints.map(point => `<li>${point.trim()}</li>`).join('') + '</ul>';
        } else {
          treatmentHtml = data.treatment; // Fallback if no newlines are present
        }
        treatmentText.innerHTML = treatmentHtml;
      }
    } catch (err) {
      diseaseLabel.textContent = "Error";
      confidence.textContent = "";
      treatmentText.textContent = err.message;
    }
  });

  // Language selector change event listener
  document.getElementById("language").addEventListener("change", function() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (file) {
      document.getElementById("uploadBtn").click(); // Re-trigger prediction with new language
    }
  }); 
});
