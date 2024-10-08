/* This is a basic workflow involving performing an API call to an LLM (OpenAI)
   in order to perform a context-aware probe on a respondent's previous response.
   
   For full context of its usage within a survey, see `taylor_probe_svy.qsf`.
*/

Qualtrics.SurveyEngine.addOnload(function() {
    /* Place your JavaScript here to run when the page loads */
    this.hideNextButton();
    document.getElementById('loadingMessage').style.display = 'block'; // Show loading message
    document.getElementById('loadingSpinner').style.display = 'block'; // Show loading spinner

    // Set variables containing user responses and model prompt
    var sys_prompt = " \
        You are a survey interviewer. \
        You will be provided a survey question and a response given by a respondent. \
        Your job is to probe further about the underlying motivations for their response.\
    ";

    var user_prompt = " \
        [Question]: ${q://QID1/QuestionText} \
        [Respondent Answer]: ${q://QID1/ChoiceTextEntryValue} \
    ";

    // Define function to call LLM
    function sendPromptToLLM(systemPrompt, userPrompt, onSuccess, onError) {
        // Extract global settings for LLM (must be stored in embedded Qualtrics data)
        var apiKey = Qualtrics.SurveyEngine.getEmbeddedData('apiKey');
        var baseModel = Qualtrics.SurveyEngine.getEmbeddedData('baseModel');
        var apiUrl = Qualtrics.SurveyEngine.getEmbeddedData('apiUrl');

        var xhr = new XMLHttpRequest();
        xhr.open("POST", apiUrl);

        // Prepare request
        var data = JSON.stringify({
            model: baseModel,
            temperature: 0.5,
            max_tokens: 100,
            messages: [
                { "role": "system", "content": systemPrompt },
                { "role": "user", "content": userPrompt }
            ]
        });
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.setRequestHeader("Authorization", apiKey);

        // Set a reasonable timeout for the request (e.g., 90 seconds)
        xhr.timeout = 90000; // 90,000 milliseconds = 90 seconds

        // Define action after response is received
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                document.getElementById('loadingSpinner').style.display = 'none'; // Always hide loading spinner
                document.getElementById('loadingMessage').style.display = 'none'; // Always hide loading message

                if (xhr.status === 200) {
                    try {
                        var response = JSON.parse(xhr.responseText);
                        var modelResponse = response.choices[0].message.content;
                        Qualtrics.SurveyEngine.setEmbeddedData('prompt1Completion', modelResponse);
                        onSuccess(modelResponse);
                    } catch (e) {
                        console.error("Error parsing response:", e);
                        onError("Failed to parse response");
                    }
                } else {
                    console.error("Error from OpenAI:", xhr.responseText);
                    onError("Error: " + xhr.status);
                    Qualtrics.SurveyEngine.setEmbeddedData('prompt1Completion', 'Error ' + xhr.status);
                }
                document.querySelector('.NextButton').style.display = ''; // Show the Next button
                document.getElementById('finishedMessage').style.display = 'block'; // Show finished loading message
            }
        };

        // Define action if request times out
        xhr.ontimeout = function() {
            console.error("Request timed out");
            onError("Request timed out");
            document.getElementById('loadingSpinner').style.display = 'none';
            document.getElementById('loadingMessage').style.display = 'none';
            document.getElementById('finishedMessage').style.display = 'block';
            document.querySelector('.NextButton').style.display = '';
        };

        console.log("Model: ", baseModel);

        // Fire off HTTP request
        xhr.send(data);
    }

    // Call function
    sendPromptToLLM(sys_prompt, user_prompt, function(response) {
        console.log("API says:", response);
        // Save request information (save any real action for next question)
        Qualtrics.SurveyEngine.setEmbeddedData('sys_prompt', sys_prompt);
        Qualtrics.SurveyEngine.setEmbeddedData('user_prompt', user_prompt);
    }, function(error) {
        console.log(error);
    });

});
