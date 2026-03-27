const form = document.getElementById('upload-form');
const fileInput = document.getElementById('image-input');
const spinner = document.getElementById('spinner');
const resultDiv = document.getElementById('result');
const labelP = document.getElementById('label');
const confP = document.getElementById('confidence');
const previewImg = document.getElementById('preview');

let selectedDisease = null;

// Disease selection
function selectDisease(type, buttonElement) {
  // Normalize disease type to lowercase without spaces
  selectedDisease = type.trim().toLowerCase();
  document.getElementById('disease-type').value = selectedDisease;

  // Remove active from all buttons
  document.querySelectorAll('.disease-btn').forEach(btn => {
    btn.classList.remove('active');
  });

  // Add active to clicked button
  buttonElement.classList.add('active');
}

// Image preview
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      previewImg.src = event.target.result;
      previewImg.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }
});

// Form submit
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  if (!selectedDisease) {
    alert("Please select a disease first.");
    return;
  }

  if (!fileInput.files[0]) return;

  // Normalize again before sending
  const normalizedDisease = selectedDisease.trim().toLowerCase();

  const formData = new FormData();
  formData.append('image', fileInput.files[0]);
  formData.append('disease_type', normalizedDisease);

  spinner.classList.remove('hidden');
  resultDiv.classList.add('hidden');
  previewImg.classList.add('hidden');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    labelP.textContent = data.label;
    confP.textContent = data.confidence + '%';

    const confidenceFill = document.getElementById('confidence-fill');
    confidenceFill.style.width = data.confidence + '%';

    // Reset class
    labelP.className = 'diagnosis-text';

    // Styling logic based on disease label
      if (
        data.label === 'NORMAL' ||
        data.label === 'NOT_FRACTURED' ||
        data.label === 'UNINFECTED' ||
        data.label === 'NOTUMOR') {      
        labelP.classList.add('normal');
    } else if (data.label === 'PNEUMONIA') {
      labelP.classList.add('pneumonia');
    } else if (data.label === 'TUBERCULOSIS') {
      labelP.classList.add('tuberculosis');
    } else if (data.label === 'FRACTURED') {
      labelP.classList.add('fracture');
    }
      else if (data.label === 'PARASITIZED') {
      labelP.classList.add('malaria');
    }
    else if (
      data.label === 'GLIOMA' ||
      data.label === 'MENINGIOMA' ||
      data.label === 'PITUITARY'
    ) {
      labelP.classList.add('brain-tumor'); // reuse red danger style
    }
    else if (data.label === 'NORMAL' || data.label === 'GLAUCOMA') {
      if (data.label === 'NORMAL') {
        labelP.classList.add('normal');
      } 
      else {
        labelP.classList.add('glaucoma');
      }
    }
    resultDiv.classList.remove('hidden');

  } catch (err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    spinner.classList.add('hidden');
  }
});

// Reset
function resetForm() {
  form.reset();
  previewImg.classList.add('hidden');
  resultDiv.classList.add('hidden');
  labelP.className = 'diagnosis-text';
  selectedDisease = null;
}