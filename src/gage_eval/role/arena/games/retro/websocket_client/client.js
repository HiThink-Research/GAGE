const statusEl = document.getElementById("status");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.binaryType = "blob";

function sendInput(payload) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

ws.onopen = () => {
  statusEl.textContent = "WS connected";
  console.log("ws open");
};

ws.onmessage = async (event) => {
  if (typeof event.data === "string") {
    console.log("ws text", event.data);
    return;
  }
  const blob = event.data;
  const bitmap = await createImageBitmap(blob);
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }
  ctx.drawImage(bitmap, 0, 0);
};

ws.onerror = (event) => {
  statusEl.textContent = "WS error";
  console.error("ws error", event);
};

ws.onclose = () => {
  statusEl.textContent = "WS closed";
  console.log("ws closed");
};

window.addEventListener("keydown", (event) => {
  if (event.repeat) {
    return;
  }
  sendInput({ type: "keydown", key: event.key });
});

window.addEventListener("keyup", (event) => {
  sendInput({ type: "keyup", key: event.key });
});
