// =======================================
// CAMERA CAPTURE JAVASCRIPT
// =======================================

document.addEventListener("DOMContentLoaded", function () {


    console.log("camera.js loaded");


    const captureBtn = document.getElementById("captureBtn");

    const statusText = document.getElementById("statusText");


    if(captureBtn){


        captureBtn.addEventListener("click", function () {


            console.log("Capture button clicked");


            statusText.innerHTML = "Capturing...";


            fetch("/capture", {

                method: "POST"

            })


            .then(response => response.json())


            .then(data => {


                console.log(data);


        


            })


            .catch(error=>{


                console.error(error);


                statusText.innerHTML =
                "Camera Error";
if(data.success)
{

    let message = document.getElementById("captureMessage");

    message.style.display = "block";

    message.style.background = "#dcfce7";
    message.style.color = "#166534";

    message.innerHTML =
    "✔ Face captured successfully";


    document.getElementById("face_image").value =
    "candidate.jpg";


}
else
{

    let message = document.getElementById("captureMessage");

    message.style.display = "block";

    message.style.background = "#fee2e2";
    message.style.color = "#991b1b";

    message.innerHTML =
    "✘ Face capture failed";

}

            });



        });


    }



});