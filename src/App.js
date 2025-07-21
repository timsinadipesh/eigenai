import React, { useRef, useState } from "react";
import "./App.css";

function App() {
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [attachedFiles, setAttachedFiles] = useState(new Map());
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const chatInputRef = useRef(null);
  const fileInputRef = useRef(null);

  // Sidebar toggle
  const toggleSidebar = () => {
    setSidebarHidden((h) => !h);
  };

  // Auto-resize textarea to a max of 3 lines
  const autoResizeTextarea = () => {
    const textarea = chatInputRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    const computedStyle = getComputedStyle(textarea);
    const lineHeight = parseFloat(computedStyle.lineHeight);
    const paddingTop = parseFloat(computedStyle.paddingTop);
    const paddingBottom = parseFloat(computedStyle.paddingBottom);

    // Calculate the height for a maximum of 3 lines of text
    const maxThreeLinesHeight = lineHeight * 3 + paddingTop + paddingBottom;

    if (textarea.scrollHeight > maxThreeLinesHeight) {
      textarea.style.height = maxThreeLinesHeight + "px";
      textarea.style.overflowY = "auto";
    } else {
      textarea.style.height = textarea.scrollHeight + "px";
      textarea.style.overflowY = "hidden";
    }
  };

  // Send button state
  const hasContent = chatInput.trim().length > 0 || attachedFiles.size > 0;

  // Render attachments
  const renderAttachments = () => {
    if (attachedFiles.size === 0) return null;
    return (
      <div id="attachment-display" style={{ display: "flex" }}>
        {[...attachedFiles.entries()].map(([name, file]) => (
          <div className="attachment-pill" key={name}>
            <span>{name}</span>
            <button
              className="remove-attachment-btn"
              onClick={() => {
                const newFiles = new Map(attachedFiles);
                newFiles.delete(name);
                setAttachedFiles(newFiles);
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    );
  };

  // File input
  const handleFileChange = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      const newFiles = new Map(attachedFiles);
      for (const file of files) {
        if (!newFiles.has(file.name)) {
          newFiles.set(file.name, file);
        }
      }
      setAttachedFiles(newFiles);
    }
    e.target.value = "";
  };

  // Audio recording
  const toggleRecording = async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new window.MediaRecorder(stream);
        setAudioChunks([]);
        recorder.ondataavailable = (event) => {
          setAudioChunks((prev) => [...prev, event.data]);
        };
        recorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
          const audioName = `recording-${Date.now()}.webm`;
          const newFiles = new Map(attachedFiles);
          newFiles.set(audioName, audioBlob);
          setAttachedFiles(newFiles);
          stream.getTracks().forEach((track) => track.stop());
        };
        recorder.start();
        setMediaRecorder(recorder);
        setIsRecording(true);
      } catch (err) {
        alert("Could not access microphone. Please check permissions.");
      }
    } else {
      if (mediaRecorder) {
        mediaRecorder.stop();
      }
      setIsRecording(false);
    }
  };

  // Send
  const handleSend = () => {
    if (!hasContent) return;
    // Replace with your send logic
    console.log("Send:", chatInput, attachedFiles);
    setChatInput("");
    setAttachedFiles(new Map());
    setTimeout(autoResizeTextarea, 0);
  };

  // Keyboard send
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && hasContent) {
      e.preventDefault();
      handleSend();
    }
  };

  // Effects
  React.useEffect(() => {
    autoResizeTextarea();
  }, [chatInput]);

  return (
    <div className="container">
      {/* Toggle button */}
      <button className="toggle-btn" id="toggle-btn" onClick={toggleSidebar}>
        <svg viewBox="0 0 24 24">
          <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z" />
        </svg>
      </button>

      <div className={`sidebar${sidebarHidden ? " hidden" : ""}`} id="sidebar">
        <button className="app-btn" onClick={() => console.log("New Chat clicked")}>
          New Chat
        </button>
        <button className="app-btn" onClick={() => console.log("Interpreter clicked")}>
          Interpreter
        </button>
        <button className="app-btn" onClick={() => console.log("FolderMode clicked")}>
          FolderMode
        </button>
        <div className="separator"></div>
        <div className="search-bar">
          <input type="text" id="history-search" placeholder="Search History" />
        </div>
      </div>
      <div className="main-content" id="main-content">
        <div className="centered-content">
          <div className="header header-up">
            <h1>frende</h1>
            <p className="single-line">private on-device ai companion</p>
          </div>
          <div className="input-container input-container-up">
            {renderAttachments()}
            <div className="input-box-wrapper input-box-wrapper-up">
              <textarea
                ref={chatInputRef}
                id="chat-input"
                className="input-box input-box-up"
                placeholder="hello frende..."
                rows={1}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={handleKeyDown}
                autoFocus
              />
              <button
                id="attachment-btn"
                className="icon-btn attachment-btn"
                onClick={() => fileInputRef.current.click()}
                type="button"
              >
                <svg viewBox="0 0 24 20">
                  <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v11.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" />
                </svg>
              </button>
              <input
                type="file"
                id="file-input"
                hidden
                multiple
                ref={fileInputRef}
                onChange={handleFileChange}
              />
              <button
                id="mic-btn"
                className={`icon-btn mic-btn${isRecording ? " mic-recording" : ""}`}
                onClick={toggleRecording}
                type="button"
              >
                <svg id="mic-icon" viewBox="0 0 24 18">
                  <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z" />
                </svg>
              </button>
              <button
                id="send-btn"
                className="icon-btn send-btn"
                disabled={!hasContent}
                onClick={handleSend}
                type="button"
              >
                <svg id="send-icon" viewBox="0 0 24 18">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
              </button>
            </div>
            {/* Removed mode-buttons here */}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;