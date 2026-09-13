// =======================================
// EXAMGUARD AI REGISTER JAVASCRIPT
// =======================================


// =======================================
// SHOW OTHER BRANCH
// =======================================

function showOtherBranch() {

    const department = document.getElementById("department");
    const other = document.getElementById("otherBranch");

    if (!department || !other) return;

    if (department.value === "Others") {

        other.style.display = "block";
        other.required = true;

    }

    else {

        other.style.display = "none";
        other.required = false;
        other.value = "";

    }

}


// =======================================
// PASSWORD SHOW / HIDE
// =======================================

function togglePassword(id, button) {

    const input = document.getElementById(id);

    if (!input) return;

    const icon = button.querySelector("i");

    if (input.type === "password") {

        input.type = "text";

        if (icon) {

            icon.classList.remove("bi-eye");
            icon.classList.add("bi-eye-slash");

        }

    }

    else {

        input.type = "password";

        if (icon) {

            icon.classList.remove("bi-eye-slash");
            icon.classList.add("bi-eye");

        }

    }

}


// =======================================
// DOM LOADED
// =======================================

document.addEventListener("DOMContentLoaded", function () {


// =======================================
// PASSWORD STRENGTH
// =======================================

const password = document.getElementById("password");
const strengthBar = document.getElementById("strengthBar");
const strengthText = document.getElementById("strengthText");

if (password && strengthBar && strengthText) {

    password.addEventListener("input", function () {

        let strength = 0;

        if (password.value.length >= 8) strength++;

        if (/[A-Z]/.test(password.value)) strength++;

        if (/[0-9]/.test(password.value)) strength++;

        if (/[!@#$%^&*]/.test(password.value)) strength++;

        switch (strength) {

            case 0:

                strengthBar.style.width = "10%";
                strengthText.innerHTML = "Very Weak";

                break;

            case 1:

                strengthBar.style.width = "30%";
                strengthText.innerHTML = "Weak";

                break;

            case 2:

                strengthBar.style.width = "55%";
                strengthText.innerHTML = "Medium";

                break;

            case 3:

                strengthBar.style.width = "80%";
                strengthText.innerHTML = "Strong";

                break;

            case 4:

                strengthBar.style.width = "100%";
                strengthText.innerHTML = "Very Strong";

                break;

        }

    });

}


// =======================================
// CONFIRM PASSWORD
// =======================================

const confirmPassword = document.getElementById("confirmPassword");

if (confirmPassword && password) {

    confirmPassword.addEventListener("input", function () {

        if (confirmPassword.value === password.value) {

            confirmPassword.style.borderColor = "#22c55e";

        }

        else {

            confirmPassword.style.borderColor = "#ef4444";

        }

    });

}


// =======================================
// EMAIL VALIDATION
// =======================================

const email = document.getElementById("email");
const emailError = document.getElementById("emailError");

if (email && emailError) {

    email.addEventListener("input", function () {

        const regex =
        /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

        if (regex.test(email.value)) {

            email.style.borderColor = "#22c55e";
            emailError.innerHTML = "";

        }

        else {

            email.style.borderColor = "#ef4444";
            emailError.style.color = "red";
            emailError.innerHTML = "Enter Valid Email Address";

        }

    });

}


// =======================================
// APPLICATION ID
// =======================================

const application = document.getElementById("applicationId");

if (application) {

    application.innerHTML = "Generating...";

}


// =======================================
// BUTTON ANIMATION
// =======================================

const button = document.querySelector(".register-btn");

if (button) {

    button.addEventListener("mouseenter", function () {

        button.style.transform = "translateY(-3px)";

    });

    button.addEventListener("mouseleave", function () {

        button.style.transform = "translateY(0px)";

    });
}
// =======================================
// CAMERA ELEMENTS
// =======================================

const form = document.querySelector("form");

const openCamera = document.getElementById("openCamera");

const cameraBox = document.getElementById("cameraBox");

const captureBtn = document.getElementById("captureFace");

const video = document.getElementById("video");

const capturedImage = document.getElementById("capturedImage");

const faceImage = document.getElementById("face_image");

const faceStatusText = document.getElementById("faceStatusText");

const captureSuccessBadge = document.getElementById("captureSuccessBadge");

const captureMessage = document.getElementById("captureMessage");

let faceStatusInterval = null;


// =======================================
// FACE STATUS POLLING (Registration only)
// =======================================
// Shows a clean "Face detected" / "No face detected" status next to
// the button while the camera is open. This is unrelated to exam
// warnings/scoring -- it only reflects whether a face is currently
// visible, read from the same face-detection loop.

function pollFaceStatus() {

    fetch("/face_status")

        .then(response => response.json())

        .then(data => {

            if (!faceStatusText) {
                return;
            }

            if (data.face_detected) {

                faceStatusText.textContent = "Face detected";
                faceStatusText.classList.remove("not-detected");
                faceStatusText.classList.add("detected");

            } else {

                faceStatusText.textContent = "No face detected";
                faceStatusText.classList.remove("detected");
                faceStatusText.classList.add("not-detected");

            }

        })

        .catch(() => {});

}


function stopFaceStatusPolling() {

    if (faceStatusInterval) {

        clearInterval(faceStatusInterval);
        faceStatusInterval = null;

    }

}


// =======================================
// OPEN CAMERA
// =======================================

if (openCamera) {

    openCamera.addEventListener("click", function (e) {

        e.preventDefault();

        // Extra safety on top of the disabled attribute: never
        // reopen the camera once a face has already been captured
        // for this registration.
        if (openCamera.disabled || (faceImage && faceImage.value)) {
            return;
        }

        if (cameraBox) {
            cameraBox.style.display = "block";
        }

        if (video) {
            video.style.display = "block";
            video.src = "/video_feed";
        }

        if (faceStatusText) {
            faceStatusText.style.display = "block";
            faceStatusText.textContent = "Position your face clearly";
            faceStatusText.classList.remove("detected", "not-detected");
        }

        if (captureMessage) {
            captureMessage.style.display = "none";
            captureMessage.classList.remove("success", "error");
        }

        stopFaceStatusPolling();
        faceStatusInterval = setInterval(pollFaceStatus, 800);

    });

}
// ===============================
// CAPTURE FACE
// ===============================

if (captureBtn) {

    captureBtn.onclick = function (e) {

        e.preventDefault();

        fetch("/capture", {
            method: "POST"
        })

        .then(response => response.json())

        .then(data => {

            console.log(data);

            if (data.success) {

                stopFaceStatusPolling();

                // Tell the server to release the webcam now that we
                // have the photo, then close the browser side of the
                // stream too -- otherwise the img tag keeps the
                // MJPEG connection open (and the camera on) in the
                // background even though it's hidden.
                fetch("/stop_camera", { method: "POST" }).catch(() => {});

                // Hide only the live feed -- keep cameraBox (and
                // everything inside it: the button, the badge, the
                // message, the captured image) visible.
                if (video) {
                    video.style.display = "none";
                    video.src = "";
                }

                if (faceStatusText) {
                    faceStatusText.style.display = "none";
                }

                // A face has now been captured -- don't let the
                // camera be reopened for this registration.
                if (openCamera) {
                    openCamera.disabled = true;
                }

                if (capturedImage) {

                    // Guard against the photo preview failing to load
                    // (e.g. a slow write to disk) by retrying a few
                    // times with a fresh cache-busting timestamp
                    // before giving up and telling the user.
                    let attempts = 0;

                    const loadCapturedImage = function () {

                        attempts++;

                        capturedImage.onload = function () {
                            capturedImage.style.display = "block";
                        };

                        capturedImage.onerror = function () {
                            if (attempts < 5) {
                                setTimeout(loadCapturedImage, 400);
                            } else if (captureMessage) {
                                captureMessage.textContent =
                                    "Face captured, but the preview couldn't load. " +
                                    "You can still continue registering.";
                                captureMessage.classList.remove("success");
                                captureMessage.classList.add("error");
                                captureMessage.style.display = "block";
                            }
                        };

                        capturedImage.src =
                            "/static/photos/candidate.jpg?" + Date.now();
                    };

                    loadCapturedImage();
                }

                if (faceImage) {
                    faceImage.value = "candidate.jpg";
                }

                if (captureBtn) {
                    captureBtn.textContent = "✓ Face Captured";
                    captureBtn.classList.add("captured");
                    captureBtn.disabled = true;
                }

                if (captureSuccessBadge) {
                    captureSuccessBadge.style.display = "inline-flex";
                }

                if (captureMessage) {
                    captureMessage.textContent =
                        "Face captured successfully.";
                    captureMessage.classList.remove("error");
                    captureMessage.classList.add("success");
                    captureMessage.style.display = "block";
                }

            } else {

                if (captureMessage) {
                    captureMessage.textContent =
                        data.message ||
                        "No face detected. Please position your face in the frame and try again.";
                    captureMessage.classList.remove("success");
                    captureMessage.classList.add("error");
                    captureMessage.style.display = "block";
                }

            }

        })

        .catch(error => {

            console.error(error);

        });

    };

}

// =======================================
// FORM SUBMIT
// =======================================

if (form) {

    form.addEventListener("submit", function (e) {

        if (faceImage && faceImage.value === "") {

            e.preventDefault();

            alert("Please Capture Your Face Before Registering.");

            return false;

        }

    });

}
// =======================================
// CLEAR FORM AFTER SUCCESSFUL REGISTRATION
// =======================================

if (form) {

    form.addEventListener("submit", function () {

        // Wait for form submission to complete
        setTimeout(function () {

            form.reset();

            stopFaceStatusPolling();

            if (capturedImage) {

                capturedImage.style.display = "none";
                capturedImage.src = "";

            }

            if (faceImage) {

                faceImage.value = "";

            }

            if (cameraBox) {

                cameraBox.style.display = "none";

            }

            if (video) {

                video.src = "";
                video.style.display = "block";

            }

            if (faceStatusText) {

                faceStatusText.style.display = "block";
                faceStatusText.textContent = "Position your face clearly";
                faceStatusText.classList.remove("detected", "not-detected");

            }

            if (captureBtn) {

                captureBtn.textContent = "Capture Face";
                captureBtn.classList.remove("captured");
                captureBtn.disabled = false;

            }

            if (openCamera) {

                openCamera.disabled = false;

            }

            if (captureSuccessBadge) {

                captureSuccessBadge.style.display = "none";

            }

            if (captureMessage) {

                captureMessage.style.display = "none";
                captureMessage.classList.remove("success", "error");

            }

            // Hide Other Branch textbox

            const otherBranch = document.getElementById("otherBranch");

            if (otherBranch) {

                otherBranch.style.display = "none";
                otherBranch.required = false;
                otherBranch.value = "";

            }

            // Reset password strength

            if (strengthBar) {

                strengthBar.style.width = "0%";

            }

            if (strengthText) {

                strengthText.innerHTML = "";

            }

            // Application ID placeholder

            const application = document.getElementById("applicationId");

            if (application) {

                application.innerHTML = "Generating...";

            }

        }, 500);

    });

}


// =======================================
// CLOSE DOMCONTENTLOADED
// =======================================

});