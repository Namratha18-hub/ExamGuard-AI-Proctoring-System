document.addEventListener("DOMContentLoaded", function () {

    const fileInput = document.getElementById("profilePictureInput");
    const previewImg = document.getElementById("avatarPreview");
    const currentAvatar = document.getElementById("currentAvatar");
    const saveBtn = document.getElementById("savePictureBtn");
    const errorBox = document.getElementById("uploadClientError");

    if (!fileInput) {
        return;
    }

    const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

    fileInput.addEventListener("change", function () {

        const file = fileInput.files[0];

        if (errorBox) {
            errorBox.style.display = "none";
            errorBox.textContent = "";
        }

        if (!file) {
            return;
        }

        if (!ALLOWED_TYPES.includes(file.type)) {

            if (errorBox) {
                errorBox.textContent =
                    "Only JPG, JPEG, and PNG files are allowed.";
                errorBox.style.display = "block";
            }

            fileInput.value = "";

            if (previewImg) {
                previewImg.style.display = "none";
            }

            if (currentAvatar) {
                currentAvatar.style.display = "";
            }

            if (saveBtn) {
                saveBtn.style.display = "none";
            }

            return;

        }

        const reader = new FileReader();

        reader.onload = function (event) {

            if (previewImg) {
                previewImg.src = event.target.result;
                previewImg.style.display = "block";
            }

            if (currentAvatar) {
                currentAvatar.style.display = "none";
            }

            if (saveBtn) {
                saveBtn.style.display = "inline-flex";
            }

        };

        reader.readAsDataURL(file);

    });

    // =====================================
    // EDIT PROFILE TOGGLE
    // =====================================

    const editBtn = document.getElementById("editProfileBtn");
    const cancelBtn = document.getElementById("cancelEditBtn");
    const detailsCard = document.getElementById("profileDetailsCard");

    if (editBtn && detailsCard) {

        editBtn.addEventListener("click", function () {
            detailsCard.classList.add("editing");
        });

    }

    if (cancelBtn && detailsCard) {

        cancelBtn.addEventListener("click", function () {

            detailsCard.classList.remove("editing");

            const inputs = detailsCard.querySelectorAll(".detail-input");

            inputs.forEach(function (input) {
                input.value = input.defaultValue;
            });

        });

    }

});