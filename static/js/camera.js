// const video = document.getElementById("camera");
// const canvas = document.getElementById("snapshot");
// const captureBtn = document.getElementById("captureBtn");
// const resultDiv = document.getElementById("result");

// // Start webcam
// navigator.mediaDevices.getUserMedia({ video: true })
//   .then(stream => video.srcObject = stream)
//   .catch(err => alert("Unable to access camera: " + err));

// captureBtn.addEventListener("click", async () => {
//   // Capture current frame
//   const context = canvas.getContext("2d");
//   canvas.width = video.videoWidth;
//   canvas.height = video.videoHeight;
//   context.drawImage(video, 0, 0, canvas.width, canvas.height);

//   // Convert frame to image blob
//   canvas.toBlob(async (blob) => {
//     const formData = new FormData();
//     formData.append("file", blob, "snapshot.jpg");

//     resultDiv.style.display = "block";
//     resultDiv.innerHTML = "<b>Analyzing image...</b>";

//     try {
//       const response = await fetch("/predict", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await response.json();

//       if (data.error) {
//         resultDiv.innerHTML = `<span style='color:red;'>${data.error}</span>`;
//       } else {
//         resultDiv.innerHTML = `
//           <h3>Prediction: ${data.label}</h3>
//           <p><b>Confidence:</b> ${data.confidence}%</p>
//           <p><b>Treatment:</b> ${data.treatment}</p>
//         `;
//       }
//     } catch (err) {
//       resultDiv.innerHTML = `<span style='color:red;'>Error: ${err.message}</span>`;
//     }
//   }, "image/jpeg");
// });

document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("camera");
  const canvas = document.getElementById("snapshot");
  const captureBtn = document.getElementById("captureBtn");
  const resultDiv = document.getElementById("result");
  const previewImage = document.getElementById("previewImage");
  const diseaseLabel = document.getElementById("diseaseLabel");
  const confidence = document.getElementById("confidence");
  const treatmentText = document.getElementById("treatmentText");

  // ✅ Ask for camera access
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then(stream => (video.srcObject = stream))
    .catch(err => {
      alert("Camera access denied: " + err.message);
    });

  // ✅ Capture photo & send to backend
  captureBtn.addEventListener("click", async () => {
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageBlob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));
    const formData = new FormData();
    formData.append("file", imageBlob, "capture.jpg");

    resultDiv.style.display = "block";
    diseaseLabel.textContent = "Analyzing...";
    confidence.textContent = "";
    treatmentText.textContent = "";

    try {
      const response = await fetch("/predict-camera", { method: "POST", body: formData });
      const data = await response.json();

      if (data.error) {
        diseaseLabel.textContent = "Error";
        treatmentText.textContent = data.error;
      } else {
        previewImage.src = `data:image/jpeg;base64,${data.image_data}`;
        diseaseLabel.textContent = data.label;
        confidence.textContent = data.confidence;
        treatmentText.textContent = data.treatment;
      }
    } catch (err) {
      diseaseLabel.textContent = "Error";
      treatmentText.textContent = err.message;
    }
  });
});
