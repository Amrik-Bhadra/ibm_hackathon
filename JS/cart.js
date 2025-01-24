document.addEventListener("DOMContentLoaded", function() {
    const products = document.querySelectorAll(".product");
    const cartTotal = document.getElementById("cart-total");

    function generateQRCode() {
        // Update cart total before generating QR code
        updateCartTotal();

        // Retrieve the updated cart total
        const totalAmount = parseFloat(cartTotal.textContent.trim());

        // Replace 'your-upi-id@provider' with your actual UPI ID
        const upiId = 'shivam.pokharkar18@okaxis';
        const paymentDetails = `upi://pay?pa=${upiId}&pn=Your%20Name&am=${totalAmount}&cu=INR`;

        // Generate QR code
        const qr = new QRious({
            value: paymentDetails,
            size: 300
        });

        // Convert QR code to data URL
        const qrDataURL = qr.toDataURL();

        // Open QR code in a new window
        const qrWindow = window.open("qrcode.html", "_blank");

        if (qrWindow) {
            qrWindow.onload = function() {
                // Draw QR code onto the canvas in qrcode.html
                const qrCanvas = qrWindow.document.getElementById('qrcode');
                const ctx = qrCanvas.getContext('2d');
                const img = new Image();
                img.src = qrDataURL;
                img.onload = function() {
                    ctx.drawImage(img, 0, 0, qrCanvas.width, qrCanvas.height);
                };
                // Display total amount in QR code window
                const totalAmountParagraph = qrWindow.document.getElementById('total-amount');
                totalAmountParagraph.textContent += totalAmount.toFixed(2);

                // Set the canvas size to match its container
                qrCanvas.width = qrCanvas.parentElement.clientWidth;
                qrCanvas.height = qrCanvas.parentElement.clientHeight;
            };
        } else {
            alert("Popup blocked! Please allow popups to see the QR code.");
        }
    }

    document.querySelector(".checkout").addEventListener("click", function() {
        generateQRCode();
    });

    const removeProduct = function(event) {
        const product = event.target.closest(".product");
        product.remove();
        updateCartTotal();
    }

    const updateCartTotal = function() {
        let subtotal = 0;
        const products = document.querySelectorAll(".product");
        
        products.forEach(product => {
            const price = parseFloat(product.querySelector(".product-price").textContent);
            const quantity = parseFloat(product.querySelector(".product-quantity input").value);
            const linePrice = price * quantity;
            product.querySelector(".product-line-price").textContent = linePrice.toFixed(2);
            subtotal += linePrice;
        });
    
        const tax = subtotal * 0.05;
        const shipping = 15.00;
        const total = subtotal + tax + shipping;
    
        document.getElementById("cart-subtotal").textContent = subtotal.toFixed(2);
        document.getElementById("cart-tax").textContent = tax.toFixed(2);
        document.getElementById("cart-shipping").textContent = shipping.toFixed(2);
        document.getElementById("cart-total").textContent = total.toFixed(2);
    };
    
    document.querySelectorAll(".product-quantity input").forEach(input => {
        input.addEventListener("input", updateCartTotal);
    });
    
    document.querySelectorAll(".remove-product").forEach(button => {
        button.addEventListener("click", removeProduct);
    });
    
    updateCartTotal();
    
});
