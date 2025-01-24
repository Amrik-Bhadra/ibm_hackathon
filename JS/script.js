function checkLoginStatus() {
    fetch('/checkLoginStatus') // Change the URL to your server route that checks the login status
        .then(response => response.json())
        .then(data => {
            const avatar = document.getElementById("avatar");
            const loginButton = document.getElementById("loginButton");
            if (data.loggedIn) {
                avatar.classList.remove("profile-pic-hidden");
                avatar.classList.add("profile-pic-box");
                loginButton.classList.add("profile-pic-hidden");
            }
            else {
                document.getElementById('avatar').style.display = 'none';
            }
        });
}
checkLoginStatus();

// browse by category slider
const initSlider = () => {
    const cardList = document.querySelector(".category-slider-wrapper");
    const sliderButtons = document.querySelectorAll(".arrow-box");
    const cardWidth = document.querySelector(".category-card").offsetWidth;

    sliderButtons.forEach(button => {
        button.addEventListener("click", () => {
            const direction = button.id === "left-arrow-btn" ? -1 : 1;
            const scrollAmount = cardWidth * direction;
            cardList.scrollBy({ left: scrollAmount, behavior: "smooth" });
        });
    });
}
window.addEventListener("load", initSlider);


// chatbot section
document.addEventListener('DOMContentLoaded', function () {
    const chatOpenBox = document.querySelector('.chat-open-box');
    const chatContainer = document.querySelector('.chat-container');

    // Initially hide the chat container
    chatContainer.classList.add('chathidden');

    // Toggle visibility of chat container and change icon class on click
    chatOpenBox.addEventListener('click', function () {
        chatContainer.classList.toggle('chathidden');
        const icon = chatOpenBox.querySelector('i');
        icon.classList.toggle('fa-message');
        icon.classList.toggle('fa-x');
    });
});

const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const chatBox = document.querySelector(".chatbox");

// create a chat <li> element
const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent = className === "outgoing" ? `<p>${message}</p>` : `<span><img src="/ASSETS/IMAGES/chatbot.png" alt=""></span>
            <p>${message}</p>`;

    chatLi.innerHTML = chatContent;
    return chatLi;
}

const handleChat = () => {
    const userMessage = chatInput.value.trim();
    if (!userMessage) return;  // if chatfield is empty

    // append user's message to chat section
    chatBox.appendChild(createChatLi(userMessage, "outgoing"));

    // Clear the textarea
    chatInput.value = "";

    setTimeout(() => {
        chatBox.appendChild(createChatLi("...", "incoming"));
    }, 600);
}

sendChatBtn.addEventListener("click", handleChat);

function fetchCartItemCount() {
    fetch('/cartItemCount')
        .then(response => response.json())
        .then(data => {
            itemCount = data.itemCount;

            let cartCountElement = document.querySelector('.cart-count');

            if (cartCountElement) {
                cartCountElement.textContent = itemCount;
            }
        })
        .catch(error => {
            console.error('Error fetching cart item count:', error);
        });
}
fetchCartItemCount();