document.getElementById('userForm').addEventListener('submit', function (event) {
  event.preventDefault();

  // Clear all error messages
  clearErrors();

  // Validation checks
  let valid = true;

  // Name validation
  const name = document.getElementById('name').value;
  if (!name.trim()) {
    displayError('nameError');
    valid = false;
  }

  // Email validation
  const email = document.getElementById('email').value;
  const emailRegex = /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/;
  if (!emailRegex.test(email)) {
    displayError('emailError');
    valid = false;
  }

  // Password validation
  const password = document.getElementById('password').value;
  const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
  if (!passwordRegex.test(password)) {
    displayError('passwordError');
    valid = false;
  }

  // Age validation
  const age = document.getElementById('age').value;
  if (age < 18 || age > 100) {
    displayError('ageError');
    valid = false;
  }

  if (valid) {
    alert('Form submitted successfully!');
  }
});

// Live validation for email
document.getElementById('email').addEventListener('input', function () {
  const email = document.getElementById('email').value;
  const emailRegex = /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/;
  if (emailRegex.test(email)) {
    hideError('emailError');
  } else {
    displayError('emailError');
  }
});

// Live validation for password
document.getElementById('password').addEventListener('input', function () {
  const password = document.getElementById('password').value;

  // Check for each password requirement
  updatePasswordCriteria(password.length >= 8, 'passwordLength');
  updatePasswordCriteria(/[A-Z]/.test(password), 'passwordUpper');
  updatePasswordCriteria(/[a-z]/.test(password), 'passwordLower');
  updatePasswordCriteria(/\d/.test(password), 'passwordNumber');
  updatePasswordCriteria(/[@$!%*?&]/.test(password), 'passwordSpecial');

  // Check password strength and update the strength meter
  updatePasswordStrength(password);
});

// Function to update criteria (show red/green based on whether it is satisfied)
function updatePasswordCriteria(isValid, elementId) {
  const element = document.getElementById(elementId);

  // Reset the text to its original state (without checkmarks)
  const originalText = elementId === 'passwordLength' ? 'At least 8 characters' :
                       elementId === 'passwordUpper' ? 'At least one uppercase letter' :
                       elementId === 'passwordLower' ? 'At least one lowercase letter' :
                       elementId === 'passwordNumber' ? 'At least one number' :
                       'At least one special character (@$!%*?&)';

  // Reset the content of the element
  element.innerHTML = originalText;

  // Add checkmark if the condition is met
  if (isValid) {
    element.classList.remove('text-gray-600');
    element.classList.add('text-green-600');
    element.innerHTML += ' &#10003;'; // Append a single checkmark
  } else {
    element.classList.remove('text-green-600');
    element.classList.add('text-gray-600');
  }
}

// Function to update the password strength meter
function updatePasswordStrength(password) {
  const strengthBar = document.getElementById('passwordStrengthFill');
  const strengthText = document.getElementById('passwordStrengthText');

  // If password is empty, reset the bar and text
  if (password === "") {
    strengthBar.style.width = '0%';
    strengthBar.style.backgroundColor = 'transparent';
    strengthText.innerHTML = '';
    return;
  }

  let strength = 0;

  // Increment strength score for satisfying each condition
  if (password.length >= 8) strength++;
  if (/[A-Z]/.test(password)) strength++;
  if (/[a-z]/.test(password)) strength++;
  if (/\d/.test(password)) strength++;
  if (/[@$!%*?&]/.test(password)) strength++;

  // Determine password strength
  if (strength <= 2) {
    // Weak password
    strengthBar.style.width = '33%';
    strengthBar.style.backgroundColor = 'red';
    strengthText.innerHTML = 'Weak';
    strengthText.classList.remove('text-orange-500', 'text-green-500');
    strengthText.classList.add('text-red-500');
  } else if (strength === 3 || strength === 4) {
    // Moderate password
    strengthBar.style.width = '66%';
    strengthBar.style.backgroundColor = 'orange';
    strengthText.innerHTML = 'Moderate';
    strengthText.classList.remove('text-red-500', 'text-green-500');
    strengthText.classList.add('text-orange-500');
  } else if (strength === 5) {
    // Strong password
    strengthBar.style.width = '100%';
    strengthBar.style.backgroundColor = 'green';
    strengthText.innerHTML = 'Strong';
    strengthText.classList.remove('text-red-500', 'text-orange-500');
    strengthText.classList.add('text-green-500');
  }
}

function displayError(errorId) {
  document.getElementById(errorId).classList.remove('hidden');
}

function hideError(errorId) {
  document.getElementById(errorId).classList.add('hidden');
}

function clearErrors() {
  document.querySelectorAll('.text-red-600').forEach(function (element) {
    element.classList.add('hidden');
  });
}
