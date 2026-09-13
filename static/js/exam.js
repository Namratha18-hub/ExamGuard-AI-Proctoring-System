let currentQuestion = 0; // 0 = no question shown yet

// Read from the data-total-questions attribute on #examRoot (set by
// Jinja in exam.html) instead of an inline <script> variable -- see
// the comment above the exam.js <script> tag in exam.html for why.
let totalQuestions = parseInt(
    document.getElementById("examRoot").dataset.totalQuestions,
    10
);

const nextBtn = document.getElementById("nextBtn");
const prevBtn = document.getElementById("prevBtn");

const examForm = document.getElementById("examForm");

const timer = document.getElementById("timer");

const questionPalette = document.getElementById("questionPalette");


// ================================
// GUARDED SUBMIT (Milestone 2)
// ================================
// Timer expiry, tab-switch termination, fullscreen termination, and
// the session-status poll can all try to end the exam. This makes
// sure only the first one actually submits the form.

let examSubmitted = false;

function forceSubmitExam(message){

    if(examSubmitted){
        return;
    }

    examSubmitted = true;

    if(message){
        alert(message);
    }

    examForm.submit();

}


// ================================
// QUESTION PALETTE STATE
// ================================
// Every palette button starts plain white (see CSS + no color class
// in the template). State only ever changes here, at runtime, based
// on what the candidate actually does -- never preassigned from the
// database or any server-rendered value.
//
// Four mutually-exclusive color states (class names match the CSS
// above): none="not visited" (white), "visited" (blue), "skipped"
// (yellow), "answered" (green). Plus a separate "current" class for
// the highlighted border, which can sit on top of any of the four
// colors above.

function isAnswered(number){

    let questionBox = document.getElementById("question"+number);

    if(!questionBox){
        return false;
    }

    return questionBox.querySelector("input[type=radio]:checked") !== null;

}

function setPaletteColorState(number, state){

    let btn = document.getElementById("paletteBtn"+number);

    if(!btn){
        return;
    }

    // Answered is sticky -- once green, never downgraded back to
    // visited/skipped just because the candidate revisits it.
    if(btn.classList.contains("answered") && state !== "answered"){
        return;
    }

    btn.classList.remove("visited","skipped","answered");

    if(state){
        btn.classList.add(state);
    }

}

function markPaletteCurrent(number){

    document.querySelectorAll(".palette-btn.current").forEach(function(btn){
        btn.classList.remove("current");
    });

    let btn = document.getElementById("paletteBtn"+number);

    if(btn){
        btn.classList.add("current");
    }

}

// Delegated listener: catches every radio selection inside the exam
// form without touching the radio inputs' name/value attributes, so
// the existing scoring/database logic in submit_exam() is untouched.
examForm.addEventListener("change", function(e){

    if(e.target.type !== "radio"){
        return;
    }

    let questionBox = e.target.closest(".question-box");

    if(!questionBox){
        return;
    }

    let number = parseInt(questionBox.id.replace("question", ""), 10);

    if(!isNaN(number)){
        setPaletteColorState(number, "answered");
    }

});


// ================================
// UPDATE QUESTION NUMBER
// ================================

function updateHeading(){

    document.getElementById("questionNumber").innerHTML =
    "Question " + currentQuestion + " of " + totalQuestions;

}


// ================================
// HIDE ALL QUESTIONS
// ================================

function hideAllQuestions(){

    for(let i=1;i<=totalQuestions;i++){

        let q = document.getElementById("question"+i);

        if(q){

            q.style.display="none";

        }

    }

}



// ================================
// SHOW QUESTION
// ================================

function showQuestion(number){


    if(number < 1 || number > totalQuestions){

        return;

    }


    // The question being left behind: if it was never answered, it
    // becomes "skipped" (yellow) instead of staying "visited" (blue).
    if(currentQuestion && currentQuestion !== number){

        if(!isAnswered(currentQuestion)){
            setPaletteColorState(currentQuestion, "skipped");
        }

    }


    hideAllQuestions();


    currentQuestion = number;


    let question =
    document.getElementById("question"+number);



    if(question){

        question.style.display="block";

    }


    updateHeading();

    // The question being opened: mark it "visited" (blue) unless it's
    // already answered (green), then move the current-question
    // highlight border here.
    if(!isAnswered(number)){
        setPaletteColorState(number, "visited");
    }

    markPaletteCurrent(number);


}



window.showQuestion = showQuestion;



// ================================
// QUESTION PALETTE CLICKS
// ================================
// Replaces the old inline onclick="showQuestion({{ loop.index }})"
// (which mixed Jinja into an HTML event-handler attribute and caused
// the reported syntax errors). Each button now just carries a plain
// data-question-number="{{ loop.index }}" attribute, and a single
// delegated listener here reads it -- identical navigation behavior,
// just wired up in JS instead of inline HTML.

if(questionPalette){

    questionPalette.addEventListener("click", function(e){

        let btn = e.target.closest(".palette-btn");

        if(!btn){
            return;
        }

        let number = parseInt(btn.dataset.questionNumber, 10);

        if(!isNaN(number)){
            showQuestion(number);
        }

    });

}



// ================================
// NEXT BUTTON
// ================================

nextBtn.addEventListener("click",function(){


    if(currentQuestion < totalQuestions){

        showQuestion(currentQuestion + 1);

    }


});




// ================================
// PREVIOUS BUTTON
// ================================

prevBtn.addEventListener("click",function(){


    if(currentQuestion > 1){

        showQuestion(currentQuestion - 1);

    }


});



// ================================
// INITIAL LOAD
// ================================

showQuestion(1);



// ================================
// FULLSCREEN MODE
// ================================


document.documentElement.requestFullscreen()
.catch(()=>{});



document.addEventListener(
"fullscreenchange",
function(){


    if(!document.fullscreenElement){


        fetch("/update_warning",{

            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body: JSON.stringify({
                event_type:"fullscreen_exit"
            })

        })

        .then(response=>response.json())

        .then(data=>{

            if(data.warnings !== undefined){

                document.getElementById(
                "warningCount"
                ).innerHTML =
                data.warnings;

                document.getElementById(
                "deductedMarks"
                ).innerHTML =
                data.deducted;

            }

            if(data.terminated){

                forceSubmitExam(
                "Too many violations. Exam terminated."
                );

            } else {

                alert(
                "Fullscreen exited. Warning added!"
                );

            }

        });


    }


});



// ================================
// TAB SWITCH DETECTION (Part 5A)
// ================================
// Every switch away is logged the instant it happens (dynamic,
// server-authoritative counting -- see utils/tab_switch_policy.py):
//   - 1st switch : a 60-second grace timer starts. Returning within
//                  that window keeps the exam running. Staying away
//                  past 60 seconds triggers termination via the
//                  "tab_switch_timeout" event below.
//   - 2nd switch (or later): the server terminates immediately, as
//                  soon as this fetch call logs it -- no grace period.

let tabSwitchGraceTimeoutId = null;

document.addEventListener(
"visibilitychange",
function(){


    if(document.hidden){


        fetch("/update_warning",{

            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body: JSON.stringify({
                event_type:"tab_switch"
            })

        })

        .then(response=>response.json())


        .then(data=>{


            document.getElementById(
            "warningCount"
            ).innerHTML =
            data.warnings;



            document.getElementById(
            "deductedMarks"
            ).innerHTML =
            data.deducted;



            if(data.terminated){


                forceSubmitExam(
                "Multiple tab switches detected. Exam terminated."
                );

                return;


            }


            alert(
            "Tab switching detected! Return within 1 minute or the exam will be terminated."
            );


            // Start the 1-minute grace period for this (first) switch.
            tabSwitchGraceTimeoutId = setTimeout(function(){


                if(!document.hidden){
                    return;
                }


                fetch("/update_warning",{

                    method:"POST",
                    headers:{
                        "Content-Type":"application/json"
                    },
                    body: JSON.stringify({
                        event_type:"tab_switch_timeout"
                    })

                })

                .then(response=>response.json())

                .then(timeoutData=>{

                    if(timeoutData.warnings !== undefined){

                        document.getElementById(
                        "warningCount"
                        ).innerHTML =
                        timeoutData.warnings;

                        document.getElementById(
                        "deductedMarks"
                        ).innerHTML =
                        timeoutData.deducted;

                    }

                    if(timeoutData.terminated){

                        forceSubmitExam(
                        "Away from the exam tab for more than 1 minute. Exam terminated."
                        );

                    }

                });


            }, 60000);


        });


    } else {


        // Candidate returned in time -- cancel the pending 1-minute
        // timeout check for the switch that just ended.
        if(tabSwitchGraceTimeoutId){

            clearTimeout(tabSwitchGraceTimeoutId);
            tabSwitchGraceTimeoutId = null;

        }


    }


});



// ================================
// SESSION STATUS POLL (Milestone 2 / Part 10)
// ================================
// Catches camera-triggered terminations (no-face, multiple persons,
// phone/laptop/book, head-turn) which happen inside the video stream
// and wouldn't otherwise be noticed by the browser.
//
// Part 10: interval tightened from 5000ms to 1000ms so the sidebar
// warning/marks-deducted counters (the ONLY place they're shown --
// nothing is drawn inside the camera feed) reflect a camera-confirmed
// violation almost immediately instead of up to 5 seconds late.
// Browser-triggered events (tab switch, fullscreen exit) already
// update instantly via their own direct /update_warning calls above;
// this just brings camera-triggered events to the same standard,
// still reading from the same existing /session_status ->
// evaluate_session() proctoring/event pipeline.

setInterval(function(){

    fetch("/session_status")

    .then(response=>response.json())

    .then(data=>{

        if(!data.success){
            return;
        }

        document.getElementById(
        "warningCount"
        ).innerHTML =
        data.warnings;

        document.getElementById(
        "deductedMarks"
        ).innerHTML =
        data.deducted;

        if(data.terminated){

            forceSubmitExam(
            "Too many violations. Exam terminated."
            );

        }

    });

},1000);



// ================================
// TIMER
// ================================


let totalTime = 30 * 60;



let countdown = setInterval(function(){



    let minutes =
    Math.floor(totalTime/60);



    let seconds =
    totalTime % 60;



    timer.innerHTML =

    String(minutes).padStart(2,"0")
    +
    ":"
    +
    String(seconds).padStart(2,"0");



    if(totalTime <= 0){


        clearInterval(countdown);


        forceSubmitExam(
        "Time completed. Exam submitted."
        );


    }



    totalTime--;



},1000);