const input = document.querySelector('input');
const button = document.querySelector('button');
const messages = document.querySelector('.messages');

button.addEventListener('click', sendMessage);
input.addEventListener('keydown', function(event) {
  if (event.keyCode === 13) {
    sendMessage();
  }
});

const endSessionBtn = document.getElementById('end-session-btn');
endSessionBtn.addEventListener('click', endSession);
document.addEventListener('DOMContentLoaded', createSession);


function createSession() {
    fetch('/create_session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
    })
    .then(response => response.json())
    .then(data => addMessage(data.text, 'bot'))
    .catch(error => console.error(error));
}

function endSession() {
    fetch('/delete', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
    })
    .then(response => response.json())
    .then(data => addMessage(data.text, 'me'))
    .catch(error => console.error(error));
}

function sendMessage() {
    const text = input.value;
    if (text !== '') {
        addMessage(text, 'me');
        input.value = '';
        getResponse(text);
    }
}

function getResponse(text) {
    fetch('/message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({text: text})
    })
    .then(response => response.json())
    .then(data => addMessage(data.text, 'bot'))
    .catch(error => console.error(error));
}

function addMessage(text, sender) {
    const message = document.createElement('div');
    message.innerText = text;
    message.classList.add('message', sender);
    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;
}
