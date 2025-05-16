const videoStream = document.getElementById('videoStream');
const gestureNameElement = document.getElementById('gestureName');
const confidenceElement = document.getElementById('confidence');

async function getVideoStream() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const videoTrack = stream.getVideoTracks()[0];
        window.imageCapture = new ImageCapture(videoTrack); // 将 imageCapture 存储到 window
        captureFrame(window.imageCapture);
    } catch (error) {
        console.error("Error accessing camera:", error);
        videoStream.src = "{% static 'recognizer/images/camera_error.png' %}"; // Optional error image
    }
}

async function captureFrame(imageCapture) {
    try {
        const frameBitmap = await imageCapture.grabFrame();

        // 创建 OffscreenCanvas
        const offscreenCanvas = new OffscreenCanvas(frameBitmap.width, frameBitmap.height);
        const ctx = offscreenCanvas.getContext('2d');

        // 将 ImageBitmap 绘制到 OffscreenCanvas 上
        ctx.drawImage(frameBitmap, 0, 0);

        // 将 OffscreenCanvas 转换为 Blob
        offscreenCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 })
            .then(blob => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64data = reader.result.split(',')[1];
                    sendFrame(base64data);
                };
                reader.readAsDataURL(blob);
            });

        frameBitmap.close(); // 释放 ImageBitmap 资源

    } catch (error) {
        console.error("Error capturing frame:", error);
        setTimeout(() => captureFrame(window.imageCapture), 50); // 使用 window.imageCapture，帧率调高到约 20 FPS
        return;
    }
}

function sendFrame(base64data) {
    fetch('/recognize/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: `image_data=data:image/jpeg;base64,${encodeURIComponent(base64data)}`
    })
    .then(response => response.json())
    .then(data => {
        videoStream.src = data.frame;
        gestureNameElement.textContent = `Gesture: ${data.gesture}`;
        confidenceElement.textContent = data.confidence > 0.8 ? `(Confidence: ${(data.confidence * 100).toFixed(2)}%)` : '';
        setTimeout(() => captureFrame(window.imageCapture), 50); // 使用 window.imageCapture，帧率调高到约 20 FPS
    })
    .catch(error => {
        console.error("Error sending frame:", error);
        setTimeout(() => getVideoStream(), 1000); // Retry camera access
    });
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.startsWith(name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Start video stream
getVideoStream();