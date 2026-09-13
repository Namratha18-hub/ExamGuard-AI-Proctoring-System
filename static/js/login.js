// ===============================
// LOGIN PASSWORD SHOW / HIDE
// ===============================

document.addEventListener("DOMContentLoaded", function () {

    const password = document.getElementById("password");
    const togglePassword = document.getElementById("togglePassword");
    const toggleIcon = document.getElementById("toggleIcon");

    if (password && togglePassword && toggleIcon) {

        togglePassword.addEventListener("click", function () {

            if (password.type === "password") {

                password.type = "text";

                toggleIcon.classList.remove("bi-eye");
                toggleIcon.classList.add("bi-eye-slash");

            } else {

                password.type = "password";

                toggleIcon.classList.remove("bi-eye-slash");
                toggleIcon.classList.add("bi-eye");

            }

        });

    }

});