function getTime() {
    let now = new Date();
    return now.getHours() + ":" + (now.getMinutes() < 10 ? "0" : "") + now.getMinutes();
}

async function sendMessage() {
    let userInput = document.getElementById("userInput").value;
    let chatbox = document.getElementById("chatbox");

    if (userInput.trim() === "") return;

    // Create user message
    let userMsgWrapper = document.createElement("div");
    userMsgWrapper.classList.add("message-wrapper");
    userMsgWrapper.innerHTML = `<p class="chat-message user-message">
                                    <strong>You:</strong> ${userInput}
                                    <span class="timestamp">${getTime()}</span>
                                </p>`;

    chatbox.appendChild(userMsgWrapper);

    // Show typing indicator
    let botTypingWrapper = document.createElement("div");
    botTypingWrapper.classList.add("message-wrapper");
    let typingIndicator = document.createElement("p");
    typingIndicator.innerHTML = "<strong>Cop Chatbot:</strong> Typing...";
    typingIndicator.classList.add("chat-message", "bot-message");
    botTypingWrapper.appendChild(typingIndicator);
    chatbox.appendChild(botTypingWrapper);

    chatbox.scrollTop = chatbox.scrollHeight;

    try {
        let response = await fetch("http://localhost:5000/chat", {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            body: JSON.stringify({ message: userInput }),
        });

        let data = await response.json();

        // Remove typing indicator
        botTypingWrapper.remove();

        // Show bot response
        let botMsgWrapper = document.createElement("div");
        botMsgWrapper.classList.add("message-wrapper");
        botMsgWrapper.innerHTML = `<p class="chat-message bot-message">
                                        <strong>Cop Chatbot:</strong> ${data.reply}
                                        <span class="timestamp">${getTime()}</span>
                                    </p>`;
        chatbox.appendChild(botMsgWrapper);
    } catch (error) {
        console.error("Error:", error);
        botTypingWrapper.remove();
        let errorMsg = document.createElement("div");
        errorMsg.classList.add("message-wrapper");
        errorMsg.innerHTML = `<p class="chat-message bot-message">
                                <strong>Cop Chatbot:</strong> Error connecting to server: ${error.message}
                                <span class="timestamp">${getTime()}</span>
                              </p>`;
        chatbox.appendChild(errorMsg);
    }

    chatbox.scrollTop = chatbox.scrollHeight;
    document.getElementById("userInput").value = "";
}