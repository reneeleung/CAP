let annotations = [];
let wordData = []; // Save information for each word
let tokenSeq = 0; // Record the sequence order of each token
const socket = io();
let current_filename = "";
let selectedWords = [];
let manuallySelectedWord = null;
let first_prediction = 0
// Global configuration file data
let loadedConfigFolder = null;
let annotationConfig = {};
let styleConfig = {};
let currentIndex = 0;
let fileList = [];
let currentDir = "preprocessing";
let tempDir = currentDir;
let showAnnotations = false;
let disablePauseTraining = true;

// Load annotation_config.json from a specific directory
function loadAnnotationConfig(dir) {
    return $.getJSON(`/get_file/${encodeURIComponent(dir + '/annotation_config.json')}`)
        .done(function (data) {
            annotationConfig = data;
            console.log('annotationConfig loaded:', annotationConfig);
            buildAllAnnotationUI();
        });
}

// Load and apply style_config.json
function loadStyleConfig() {
    return $.getJSON('/static/conf/style_config.json').done(function (data) {
        styleConfig = data;
        applyStyleConfig(data);
    });
}

// Load the contents of the specified directory
function loadDirectory(dir) {
    $.getJSON(`/list_directory?dir=${encodeURIComponent(dir)}`, function (response) {
        const data = response.items;
        const hasConfig = response.has_annotation_config;
    
        if (hasConfig) {
            loadAnnotationConfig(dir).then(() => {
                console.log("Annotation config updated from", dir, annotationConfig);
            });
        }
        // Clear the directory list
        $('#file-browser').empty();

        // If not at root directory, add ".." option to go back to parent directory
        if (dir !== "preprocessing") {
        // Use lastIndexOf to find the position of the last '/'
        let parentDir = "";
        const lastSlashIndex = dir.lastIndexOf('/');
        if (lastSlashIndex !== -1) {
            parentDir = dir.substring(0, lastSlashIndex);
        }
        const parentLi = $('<li>')
            .text('..')
            .css({ 'font-weight': 'bold', 'cursor': 'pointer' });
        // Double-click to go back to parent directory
        parentLi.on("dblclick", function() {
            tempDir = parentDir;
            loadDirectory(tempDir);
        });
        $('#file-browser').append(parentLi);
        }

        // Update directory display text
        $('#current-dir').text('Current directory: /' + dir);
        // Clear the current file list
        tempFileList = [];

        // Process the returned directory and file items
        data.forEach(item => {
            if (item.name != '__pycache__') {
                const folderSymbol = '<i class="fas fa-folder model-section-item-icon" style="color: #DAA520;"></i>'; // Yellow folder icon
                const fileSymbol = '<i class="fas fa-file model-section-item-icon"></i>'; // File icon
    
                const li = $('<li>')
                    .html((item.type === 'directory' ? folderSymbol : item.type === 'file_annotated' ? fileSymbol.replace("></i>", ' style="color: #0056b3"></i>') : fileSymbol) + '<span class="modal-section-item-name">' + item.name + '</span>')
                    .css('cursor', 'pointer');
                li.data("item", item);
                if (item.type === "directory") {
                    // Double-click folder to enter that directory
                    li.on("dblclick", function() {
                    tempDir = item.path;
                    loadDirectory(tempDir);
                    });
                } else if (item.type.startsWith("file")) {
                    // Add file to global fileList array, load the file on click
                    tempFileList.push(item);
                    li.on("dblclick", function() {
                        fileList = tempFileList;
                        currentIndex = fileList.findIndex(f => f.path === item.path);
                        console.log("item path: ",item.path);
                        fetchTextFile(item.path, false).then(() => {
                            currentDir = tempDir;
                            closeModal();
                            getTimerState();
                            const wasRunning = isRunning;
                            if (wasRunning) { // reset start time without saving current file
                                isRunning = false;
                                startTimer();
                            }
                        });
                    });
                }
                $('#file-browser').append(li);
        }});
    });
}

function buildAllAnnotationUI() {
    buildEntityArea();
    buildEventArea();
    buildAttributeArea(document.getElementById("attribute-area"));
    bindModalEvents();       // (re)attach change handlers
    // Default-select the first event if any
    const firstEvent = document.querySelector('input[name="event"]');
    if (firstEvent) {
        firstEvent.checked = true;
        firstEvent.dispatchEvent(new Event("change"));
    }
}

function buildEntityArea() {
    const c = document.getElementById("entity-area");
    c.innerHTML = "<h4>Entity</h4>";
    (annotationConfig.entityTypes || []).forEach(type => {
        const div = document.createElement("div");
        const display = entityDisplayName(type);
        div.innerHTML = `
            <input type="radio" name="entity" value="${type}" id="entity-${type}">
            <label for="entity-${type}">${display}</label>`;
        c.appendChild(div);
    });
}

function buildEventArea() {
    const c = document.getElementById("event-area");
    const types = annotationConfig.eventTypes || [];
    if (types.length === 0) {
        // Hide the whole Event section
        c.style.display = "none";
        c.innerHTML = "";
        return;
    }
    c.style.display = "";
    c.innerHTML = "<h4>Event</h4>";
    types.forEach(type => {
        const id = "event-" + type.replace(/\s+/g, '');
        const div = document.createElement("div");
        div.innerHTML = `
            <input type="radio" name="event" value="${type}" id="${id}">
            <label for="${id}">${type}</label>`;
        c.appendChild(div);
    });
}

function buildAttributeArea(container) {
    const attrTypes = annotationConfig.attributeTypes || {};
    if (Object.keys(attrTypes).length === 0) {
        container.style.display = "none";
        container.innerHTML = "";
        return;
    }
    container.style.display = "";
    container.innerHTML = "<h4>Attributes</h4>";
    Object.entries(attrTypes).forEach(([prefix, values]) => {
        const rowDiv = document.createElement("div");
        rowDiv.classList.add("attribute-row");
        rowDiv.dataset.attributePrefix = prefix;

        const prefixLabel = document.createElement("label");
        prefixLabel.classList.add("attr-prefix");
        prefixLabel.textContent = prefix + ":";
        rowDiv.appendChild(prefixLabel);

        Object.entries(values).forEach(([displayValue, mappedValue]) => {
            // Use mappedValue as the radio value — guaranteed unique across groups
            const radioId = `attribute-${prefix}-${mappedValue}`.replace(/\s+/g, '');

            const radio = document.createElement("input");
            radio.type = "radio";
            radio.name = prefix;              // still group by prefix
            radio.value = mappedValue;         // UNIQUE value
            radio.id = radioId;
            radio.dataset.displayValue = displayValue;
            radio.dataset.prefix = prefix;

            // Only special_entity-style uncheckable behavior, keep if needed
            if (prefix.toLowerCase() === "special_entity") {
                radio.dataset.wasChecked = "false";
                radio.addEventListener("click", function () {
                    if (this.dataset.wasChecked === "true") {
                        this.checked = false;
                        this.dataset.wasChecked = "false";
                    } else {
                        document.querySelectorAll(`input[name="${prefix}"]`).forEach(r => {
                            r.dataset.wasChecked = "false";
                        });
                        this.dataset.wasChecked = "true";
                    }
                });
            }

            const valLabel = document.createElement("label");
            valLabel.setAttribute("for", radioId);
            valLabel.textContent = displayValue || "(none)";

            const wrapper = document.createElement("span");
            wrapper.classList.add("radio-wrapper");
            wrapper.appendChild(radio);
            wrapper.appendChild(valLabel);

            rowDiv.appendChild(wrapper);
        });

        container.appendChild(rowDiv);
    });
}

let timerInterval = null;
let startTime = 0;
let elapsedTime = 0;
let isRunning = false;
let lastSaveTime = 0;
const saveInterval = 60000; // Auto save every minute

// Format time HH:MM:SS
function formatTime(milliseconds) {
    const totalSeconds = Math.floor(milliseconds / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    
    return [
        hours.toString().padStart(2, '0'),
        minutes.toString().padStart(2, '0'),
        seconds.toString().padStart(2, '0')
    ].join(':');
}

// Update timer display
function updateTimerDisplay() {
    const currentTime = isRunning ? (Date.now() - startTime + elapsedTime) : elapsedTime;
    document.getElementById('timer-display').textContent = formatTime(currentTime);
    
    // Periodically save timer status
    if (isRunning && current_filename && (Date.now() - lastSaveTime) > saveInterval) {
        if (document.getElementById("mode-general").classList.contains("active")) {
            saveTimerState(elapsedTime);
        }
        lastSaveTime = Date.now();
    }
    if (isRunning && current_filename) {
        document.getElementById('start-timer').disabled = true;
        document.getElementById('pause-timer').disabled = false;
    } else if (current_filename) {
        document.getElementById('start-timer').disabled = false;
        document.getElementById('pause-timer').disabled = true;
    }
}

function resetTimerDisplay() {
    clearInterval(timerInterval);
    isRunning = false;
    document.getElementById('start-timer').disabled = true;
    document.getElementById('pause-timer').disabled = true;
    elapsedTime = 0;
    updateTimerDisplay();
}

// Start timer
function startTimer() {
    if (!isRunning) {
        startTime = Date.now();
        isRunning = true;
        document.getElementById('start-timer').disabled = true;
        document.getElementById('pause-timer').disabled = false;
        
        timerInterval = setInterval(updateTimerDisplay, 1000);
        updateTimerDisplay(); // Update time display immediately
        
        // Record start time
        lastSaveTime = Date.now();
    }
}

// Pause timer
function pauseTimer() {
    if (isRunning) {
        clearInterval(timerInterval);
        elapsedTime += Date.now() - startTime;
        isRunning = false;
        document.getElementById('start-timer').disabled = false;
        document.getElementById('pause-timer').disabled = true;
        if (document.getElementById("mode-general").classList.contains("active")) {
            saveTimerState(elapsedTime);
        }
    }
}

// For general annotation
async function saveTimerState(elapsedTime) {
    console.log("Saving timer state for file: "+current_filename);
    $.ajax({
        url: '/save_timer',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            file: current_filename,
            elapsed_time: elapsedTime,
        }),
        success: function(response) {
            console.log("Timer state saved:", response.message);
        },
        error: function(xhr) {
            console.error("Error saving timer state:", xhr.responseText);
        }
    });
}
// For general annotation
function getTimerState() {
    console.log("Getting timer state for file: ", current_filename);
    $.ajax({
        url: `/get_timer?file=${current_filename}`,
        type: 'GET',
        contentType: 'application/json',
        success: function(response) {
            elapsedTime = response.elapsedTime;
            updateTimerDisplay();
        },
        error: function(xhr) {
            console.error("Error getting timer state:", xhr.responseText);
        }
    });
}

// 綁定計時器按鈕事件
document.getElementById('start-timer').addEventListener('click', startTimer);
document.getElementById('pause-timer').addEventListener('click', pauseTimer);

function refreshProgressBar(numAnnotated, total) {
    const progressBar = document.getElementById('progress-bar');
    const percentage = (numAnnotated / total) * 100;
    progressBar.style.width = percentage + '%';
    progressBar.textContent = Math.round(percentage) + '%';
}

// Make modal draggable
const modal = document.getElementById("modal");
modal.onmousedown = (e) => {
    const offsetX = e.clientX - modal.offsetLeft;
    const offsetY = e.clientY - modal.offsetTop;

    document.onmousemove = (e) => {
        modal.style.left = `${e.clientX - offsetX}px`;
        modal.style.top = `${e.clientY - offsetY}px`;
    };

    document.onmouseup = () => {
        document.onmousemove = null;
        document.onmouseup = null;
    };
};

$(document).ready(function () {
    $.when(loadStyleConfig()).then(function () {
        console.log("All configurations have been loaded", styleConfig);
    });

    // Initial mode set to General Annotation
    // Hide "Start Training" button, show upload area
    $("#startTraining, #resumeTraining, #confirmPauseTraining").hide();
    $("#upload-folder-form, #uploadForm").show();
    $('#prevButton, #nextButton').show();
    $("#saveAnnotations").hide();
    $("#progress-container").hide();
    // Mode switching: General Annotation
    $("#mode-general").on("click", function () {
        if (document.getElementById("startTraining").classList.contains("active") || document.getElementById("resumeTraining").classList.contains("active")) {
            const wasRunning = suspendTimerForDialog();

            // warn that this will pause training when switching mode
            const switchToGeneral = confirm("This will pause ongoing training. Are you sure you want to switch to General Annotation mode?")
            if (!switchToGeneral) {
                if (wasRunning) startTimer();   // resume where we left off
                return;
            }
    
            // Attempt to pause. If it fails, resume the timer and abort.
            if (!tryPauseTraining()) {          // tryPauseTraining from previous fix
                if (wasRunning) startTimer();   // ← resume after "not allowed" alert
                return;
            }
            // else: real pause already called pauseTimer()/resetTimerDisplay() inside
        }
        // proceed with mode switch (only reached when pause succeeded / no training)
        $(this).addClass("active");
        $("#mode-al").removeClass("active");
        $("#saveAnnotations").hide();
        $("#progress-container").hide();
        // Show upload area
        $("#open-file-modal").show();
        // Hide Start Training button
        $("#startTraining, #resumeTraining, #confirmPauseTraining").hide();
        $("#navigation-buttons").show();
        pauseTimer(); // save timer state for AL mode
        document.getElementById('start-timer').disabled = true;
        document.getElementById('pause-timer').disabled = true;
        clearFileContent();
        clearRelationships();
    });

    // Mode switching: Active Learning Annotation
    $("#mode-al").on("click", function () {
        if (current_filename !== '') {
            const wasRunning = suspendTimerForDialog();
            const switchToAL = confirm("The current annotations will not be saved. Are you sure you want to switch to Active Learning mode?");
            if (!switchToAL) {
                if (wasRunning) startTimer();
                return;
            }
        }
        $(this).addClass("active");
        $("#mode-general").removeClass("active");
        // Hide upload area
        $("#open-file-modal").hide();
        // Show Start Training button
        $("#saveAnnotations").show();
        $("#progress-container").show();
        $("#startTraining, #resumeTraining, #confirmPauseTraining").show();
        $("#navigation-buttons").hide();
        clearFileContent();
        clearRelationships();
    });

    $('#open-file-modal').on('click', openModal);
    $('#close-file-modal').on('click', closeModal);
    // Click on overlay area also closes Modal
    $('#file-modalOverlay').on('click', function(e) {
        if(e.target === this) closeModal();
     });

    // -----------------------------
    // Save Selection
    // -----------------------------
    document.getElementById("saveSelection").addEventListener("click", function () {
        const selectedEntityRadio = document.querySelector('input[name="entity"]:checked');
        const selectedEventRadio = document.querySelector('input[name="event"]:checked');
        const selectedAttributes = {};
        document.querySelectorAll('#attribute-area input[type="radio"]:checked').forEach(attr => {
            if (attr.dataset.prefix) {
                selectedAttributes[attr.dataset.prefix] = attr.dataset.displayValue;
            }
        });
        const selectedEntity = selectedEntityRadio ? selectedEntityRadio.value : "";
        const selectedEvent = selectedEventRadio ? selectedEventRadio.value : "";

        // Below update annotation of selectedWords based on selection ...

        console.log(selectedWords)
        selectedWords.forEach((word, index) => {
            const tokenSeqId = parseInt(word.id);
            const annotation = annotations.find(ann => ann.token_seq === tokenSeqId);
            if (annotation) {
                if (selectedEntityRadio) {
                    /// When Entity is selected, update the entity field,
                    // and save the selected attribute (which may be auto-selected as default) into the annotation
                    annotation.annotation.entity = index === 0 ? `B-${selectedEntity}` : `I-${selectedEntity}`;
                    annotation.annotation.attribute = selectedAttributes;  
                    annotation.annotation.event = "none";
                    $(word).removeClass("event").addClass("entity");
                    $(word).removeClass("b").removeClass("i");
                    if (annotation.annotation.entity.startsWith('B-')) {
                        $(word).addClass("b");
                    }
                    if (annotation.annotation.entity.startsWith('I-')) {
                        $(word).addClass("i");
                    }
                    console.log(selectedAttributes);
                    // remove 'predicted' class once press save
                    $(word).removeClass("predicted");
                } else if (selectedEventRadio) {
                    // When Event is selected, only update the event field,
                    // and clear attribute, because event does not need to display attribute
                    annotation.annotation.event = index === 0 ? `B-${selectedEvent}` : `I-${selectedEvent}`;
                    annotation.annotation.entity = "O";
                    annotation.annotation.attribute = {};
                    $(word).removeClass("entity").addClass("event");
                    $(word).removeClass("b").removeClass("i");
                    if (annotation.annotation.event.startsWith('B-')) {
                        $(word).addClass("b");
                    }
                    if (annotation.annotation.event.startsWith('I-')) {
                        $(word).addClass("i");
                    }
                }
            }
            $(word).addClass("annotated");
        });

        $('#modal-overlay, #modal').fadeOut();
        preventAnnotation = false;
        selectedWords = [];
        updateAnnotationLabels();
    });
    // Allow keyboard shortcut
    document.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && $('#modal-overlay, #modal').is(":visible")) {
            document.getElementById("saveSelection").click();
        }
    });

    // -----------------------------
    // Cancel Selection
    // -----------------------------
    document.getElementById("cancelSelection").addEventListener("click", function () {
        $('#modal-overlay, #modal').fadeOut();
        preventAnnotation = false;
        // Reset: Select the default Event (the first Event)
        const defaultEventRadio = document.querySelector('input[name="event"]');
        if (defaultEventRadio) {
            defaultEventRadio.checked = true;
            defaultEventRadio.dispatchEvent(new Event("change"));
        }
        // Clear selectedWords
        selectedWords.forEach(token => $(token).removeClass("selected"));
        selectedWords = [];
    });
    // Allow keyboard shortcut
    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            closeModal();
            if ($('#modal-overlay, #modal').is(":visible")) {
                document.getElementById("cancelSelection").click();
            }
        }
    });

    loadDirectory(currentDir)

});

// Socket
socket.on('connect', () => {
    console.log('Connected to server.');
});
socket.on('disconnect', () => {
    console.log('Disconnected from server.');
});
socket.on('annotation_request', async (data) => {   // async
    if (data && data.txt_file_path) {
        const folder = data.data_folder;
        if (folder && folder !== loadedConfigFolder) {
            loadedConfigFolder = folder;
            currentDir = `preprocessing/${folder}`;
            await loadAnnotationConfig(currentDir);   // await so subsequent code runs after config is ready
        }
        console.log(data.txt_file_path);
        fetchTextFile(data.txt_file_path, true);
        if (data.json_file_path && first_prediction !== 0) {
            console.log(data.json_file_path)
            console.log("Request: ",data.json_file_path)
            // fetchAnnotationFile(data.json_file_path, true);
            
        } else if (first_prediction == 0){
            first_prediction = first_prediction+1
        }
        if (data.sent_id) {
            console.log(data);
        }
        refreshProgressBar(data.num_annotated, data.total)
        disablePauseTraining = data.disable_pause
    } else {
        console.error('Invalid data format:', data);
    }
    if (first_prediction == 0){
        first_prediction = first_prediction+1
    }
    startTimer();
});
socket.on('annotation_process', (data) => {
    console.log('Process:');
    console.log(data.annotation_process);
});

// Attempts to pause training. Returns true on success, false if not allowed.
// NOTE: this does NOT touch the timer — callers handle suspend/resume.
function tryPauseTraining() {
    if (disablePauseTraining) {
        alert('Pause training not allowed. Please finish a few more reports before pausing.');
        return false;
    }
    socket.emit("end training");
    alert("Pause training signal emitted!");
    $("#startTraining").removeClass("active");
    $("#resumeTraining").removeClass("active");
    $("#confirmPauseTraining").addClass("active");
    pauseTimer();
    resetTimerDisplay();
    return true;
}

// Suspend the timer around a blocking confirm()/alert() dialog.
// Returns true if the timer was running (so caller can resume on cancel).
function suspendTimerForDialog() {
    if (isRunning) {
        pauseTimer();
        return true;
    }
    return false;
}

window.onload = function () {
    // When the user clicks the "Confirm End Training" button, emit the "end training" signal
    document.getElementById("confirmPauseTraining").addEventListener("click", function () {
        tryPauseTraining();
    });
};

function clearFileContent() {
    $('#fileContent').empty();
    wordData = [];
    annotations = [];
    tokenSeq = 0;
    updateFilename('');
}

function normalizePath(path) {
    const parts = path.split('SLE_ANN/');
    return parts.length > 1 ? parts[parts.length - 1] : path;
}

// Fetch the text file content
function fetchTextFile(txtFilePath, predicted) {
    clearFileContent();
    clearRelationships();
    let normalizedPath = txtFilePath.replace(/\\/g, '/');
    normalizedPath = normalizePath(normalizedPath);
    
    // 3. Get filename
    console.log("fetchTextFile:", normalizedPath);

    return fetch(`/get_file/${encodeURIComponent(normalizedPath)}`)
        .then(response => response.text())
        .then(text => {
            // Tokenize .txt content using the existing processLine logic
            const lines = text.split('\n');
            lines.forEach((line, index) => processLine(line, index + 1));
            console.log("Text file loaded successfully.");

            // Attempt to read the .json file with the same name
            const jsonFilename = normalizedPath.replace(/\.txt$/i, '.json');
            console.log("fetch2: ", jsonFilename)
            fetch(`/get_file/${encodeURIComponent(jsonFilename)}`)
                .then(response => {
                    if(response.ok) {
                        return response.json();
                    } else {
                        // If .json doesn't exist, return empty
                        return {};
                    }
                })
                .then(data => {
                    if (data.annotations && Array.isArray(data.annotations) && data.annotations.length > 0) {
                        // Check if annotation span starts with I-; If so, correct to B-
                        data.annotations.forEach(ann => {
                            if (ann.annotation.entity.startsWith('I-')) {
                                const entityName = ann.annotation.entity.slice(2)
                                if (ann.token_seq === 0 || (data.annotations[ann.token_seq-1].annotation.entity !== 'I-'+entityName && data.annotations[ann.token_seq-1].annotation.entity !== 'B-'+entityName)) {
                                    ann.annotation.entity = 'B-'+entityName
                                }
                            }
                        })
                        applySavedAnnotations(data.annotations, predicted);
                    }
                    if (data.relationships && Array.isArray(data.relationships) && data.relationships.length > 0) {
                        relationships = data.relationships;
                        updateRelationshipList();
                        // Update styles
                        relationships.forEach(function (rel) {
                            rel.sourceTokens.forEach(function (tokenId) {
                                let tokenElem = document.getElementById(tokenId);
                                if (tokenElem) {
                                    tokenElem.classList.add("relationship", "relationship-source");
                                    tokenElem.style.border = "2px dashed red";
                                }
                            });
                            rel.targetTokens.forEach(function (tokenId) {
                                let tokenElem = document.getElementById(tokenId);
                                if (tokenElem) {
                                    tokenElem.classList.add("relationship", "relationship-target");
                                    tokenElem.style.border = "2px dashed blue";
                                }
                            });
                        });
                    }
                })
                .catch(error => console.error('Error fetching annotation JSON:', error));
            updateFilename(txtFilePath);
            fetchNotes();
        })
        .catch(error => console.error('Error fetching text file:', error));
}

function updateFilename(filePath) {
    const fileName = filePath.split(/[/\\]/).pop();
    current_filename = filePath;
    document.getElementById("filename").textContent = 'Current File: ' + fileName;
}

function fetchAnnotationFile(jsonFilePath, predicted) {
    const fileName = jsonFilePath.split(/[/\\]/).pop();
    console.log("fetchAnnotationFile:", fileName);
    fetch(`/get_file/${encodeURIComponent(fileName)}`)
        .then(response => response.json())
        .then(data => {
            console.log("Annotations loaded:", data.annotations);
            applySavedAnnotations(data.annotations, predicted);
        })
        .catch(error => console.error('Error fetching annotation file:', error));
}

function resetAnnotations() {
    annotations.forEach((annotation, index) => {
        annotation.annotation.entity = "O";
        annotation.annotation.attribute = {};
        annotation.annotation.event = "none";
        annotation.annotation.relationship = "none";
        annotation.annotation.sourceRelationships = [];
        annotation.annotation.targetRelationships = [];
        const span = document.getElementById(index);
        if (span) {
            span.classList.remove("annotated", "entity", "event", "relationship", "relationship-source", "relationship-target");
            span.style.border = "";
        }
    });

    clearRelationships();
    updateAnnotationLabels();
    showNotification("All Annotation Reset to 'O' and relationships cleared");
    console.log("Relationships after reset:", relationships)
}

function showNotification(message) {
    const notification = document.getElementById("notification");
    notification.textContent = message;
    notification.classList.add("show");
    setTimeout(() => {
        notification.classList.remove("show");
    }, 2000);
}

socket.on('training_status', (status) => {
    console.log('Training status:', status);
    if (status.status == 'warning') {
        const wasRunning = suspendTimerForDialog();
        const forceStart = confirm(status.message)
        if (forceStart) {
            socket.emit('start_training', { force: forceStart });
            refreshProgressBar(0,1);
        } else if (wasRunning) {
            startTimer();   // user backed out — keep timer going
        }
    }
    if (status.status == 'error') {
        alert(status.message);
    }
    if (status.status == 'running' && status.mode == 'start') {
        try {
            elapsedTime = 0;
            updateTimerDisplay();
            startTimer();
        } catch (error) {
            console.error("Error starting timer, but training has started:", error);
        }
        $("#resumeTraining").removeClass("active");
        $("#confirmPauseTraining").removeClass("active");
        $("#startTraining").addClass("active");
    }
    if (status.end_training === true) {
        refreshProgressBar(1,1);
        $('#fileContent').text('Training completed!');
        pauseTimer();
    }
    if (status.status === 'completed') {
        alert('Training completed!');
    }
});

function startTraining() {
    socket.emit('start_training', {});
    console.log("Start training emitted, waiting for confirmation");
    // Wait for confirmation in training_status listener
}

function resumeTraining() {
    // First show loading status, but do not start the timer immediately
    $('#fileContent').text('Loading training data...');
    console.log("Training resume");
    socket.emit("continue_training");
    resetTimerDisplay();
    resumeTimer(); // will start timer during annotation_request
    $("#startTraining").removeClass("active");
    $("#confirmPauseTraining").removeClass("active");
    $("#resumeTraining").addClass("active");
}

function resumeTimer() {
    $.ajax({
        url: '/get_resume_time',
        type: 'GET',
        contentType: 'application/json',
        success: function(response) {
            elapsedTime = response.elapsedTime;
            updateTimerDisplay();
        },
        error: function(xhr) {
            console.error("Error getting timer state:", xhr.responseText);
        }
    });
}

function openModal() {
    $('#file-modalOverlay').css('display', 'flex');
    loadDirectory(currentDir); // load directory
}

function closeModal() {
    $('#file-modalOverlay').hide();
}

// Next button
document.getElementById("nextButton").addEventListener("click", async function () {
    if (fileList.length === 0) {
        alert("No file is currently open.");
        return;
    }
    // Update index: move to next item (loop)
    currentIndex = (currentIndex + 1) % fileList.length;
    const nextFile = fileList[currentIndex];
    current_filename = fileList[(currentIndex-1+fileList.length) % fileList.length].path
    
    const wasRunning = isRunning;
    pauseTimer();
    const outputData = {
        filename: current_filename,
        annotations: annotations,
        relationships: relationships,
    };
    current_filename = nextFile.path;
    await saveGeneralAnnotations(outputData, currentIndex, wasRunning);
});

// Prev button
document.getElementById("prevButton").addEventListener("click", async function () {
    if (fileList.length === 0) {
        alert("No file is currently open.");
        return;
    }
    // Update index: move to previous item (loop)
    currentIndex = (currentIndex - 1 + fileList.length) % fileList.length;
    const prevFile = fileList[currentIndex];
    current_filename = fileList[(currentIndex+1) % fileList.length].path

    const wasRunning = isRunning;
    pauseTimer();
    const outputData = {
        filename: current_filename,
        annotations: annotations,
        relationships: relationships,
    };
    current_filename = prevFile.path;
    await saveGeneralAnnotations(outputData, currentIndex, wasRunning);
});

async function saveGeneralAnnotations(outputData, currentIndex, wasRunning) {
    try {
        const response = await fetch('/general_annotations', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(outputData),
        });

        const data = await response.json();
        showNotification(data.message);
        clearFileContent();
        clearRelationships();
        await getNextFile(currentIndex);
        if (wasRunning) {
            startTimer();
        }
    } catch (error) {
        alert("Error saving annotations.");
    }
}
const delay = ms => new Promise(res => setTimeout(res, ms));

async function getNextFile(currentIndex) {
    const data = await $.getJSON(`/get_file_by_index?dir=${encodeURIComponent(currentDir)}&index=${currentIndex}`);
    updateFilename(currentDir+'/'+data.filename);
    fetchNotes();
    await getTimerState();

    const lines = data.content.split('\n');
    lines.forEach((line, index) => processLine(line, index + 1));

    let savedAnnotations = data.annotations || [];
    if (typeof savedAnnotations === 'string') {
        try {
            savedAnnotations = JSON.parse(savedAnnotations);
        } catch (e) {
            console.error("Failed to parse annotations", e);
            savedAnnotations = [];
        }
    }
    if (Array.isArray(savedAnnotations) && savedAnnotations.length > 0) {
        applySavedAnnotations(savedAnnotations, false);
    }

    let savedRelationships = data.relationships || [];
    if (typeof savedRelationships === 'string') {
        try {
            savedRelationships = JSON.parse(savedRelationships);
        } catch (e) {
            console.error("Failed to parse annotations", e);
            savedRelationships = [];
        }
    }
    if (Array.isArray(savedRelationships) && savedRelationships.length > 0) {
        relationships = savedRelationships;
        updateRelationshipList();
        savedRelationships.forEach(function (rel) {
            rel.sourceTokens.forEach(function (tokenId) {
                let tokenElem = document.getElementById(tokenId);
                if (tokenElem) {
                    tokenElem.classList.add("relationship", "relationship-source");
                    tokenElem.style.border = "2px dashed red";
                }
            });
            rel.targetTokens.forEach(function (tokenId) {
                let tokenElem = document.getElementById(tokenId);
                if (tokenElem) {
                    tokenElem.classList.add("relationship", "relationship-target");
                    tokenElem.style.border = "2px dashed blue";
                }
            });
        });
    }
}

// Return the display label for an internal entity type code
function entityDisplayName(code) {
    if (!code) return code;
    return (annotationConfig.entityDisplayNames && annotationConfig.entityDisplayNames[code]) || code;
}

// Given a B-/I- prefixed tag like "B-QYL", return "B-起始地"; for "O" or null, return as-is.
function entityTagDisplayName(tag) {
    if (!tag || tag === "O") return tag;
    const m = tag.match(/^([BI])-(.+)$/);
    if (!m) return tag;
    return `${m[1]}-${entityDisplayName(m[2])}`;
}

// Detect CJK (Chinese/Japanese/Korean) characters by Unicode ranges
function isCJK(char) {
    const cp = char.codePointAt(0);
    return (
        (cp >= 0x4E00  && cp <= 0x9FFF)  ||  // CJK Unified Ideographs
        (cp >= 0x3400  && cp <= 0x4DBF)  ||  // CJK Extension A
        (cp >= 0x20000 && cp <= 0x2A6DF) ||  // CJK Extension B
        (cp >= 0xF900  && cp <= 0xFAFF)  ||  // CJK Compatibility Ideographs
        (cp >= 0x2F800 && cp <= 0x2FA1F) ||  // CJK Compatibility Ideographs Supplement
        (cp >= 0x3000  && cp <= 0x303F)  ||  // CJK Symbols and Punctuation
        (cp >= 0xFF00  && cp <= 0xFFEF)      // Halfwidth and Fullwidth Forms
    );
}

function processLine(line, line_number) {
    const lineContainer = document.createElement('div');
    lineContainer.className = 'line'

    const lineNumber = document.createElement('span');
    lineNumber.className = 'line-number'
    lineNumber.textContent = line_number;
    lineContainer.appendChild(lineNumber);

    try {
        let tokens = [];
        
        // 1. First split by spaces and some explicit punctuation marks
        let initialTokens = line.split(/(\s+|[\%\*\:\(\)\,\;\#\>\"\'])/);
        
        // Filter out empty strings
        initialTokens = initialTokens.filter(token => token !== '');
        
        // 2. Further process each initially split token
        initialTokens.forEach(initialToken => {
            // Perform all splitting processing first to get preliminary tokens
            let tempTokens = [initialToken];  // Start with the original token
            
            // Process arrow symbols
            if (initialToken.includes('↑') || initialToken.includes('↓') || initialToken.includes('→') || initialToken.includes('←') || initialToken.includes('/') || initialToken.includes('+') ||
                initialToken.includes('?') || initialToken.includes('<') || initialToken.includes('>') || initialToken.includes('"') || initialToken.includes("'")) {
                let newTempTokens = [];
                for (let token of tempTokens) {
                    // Only process tokens containing arrows
                    if (token.includes('↑') || token.includes('↓') || token.includes('→') || token.includes('←') || token.includes('/') || token.includes('+') ||
                        token.includes('?') || token.includes('<') || token.includes('>')|| token.includes('"') || token.includes("'")) {
                        let currentToken = "";
                        for (let char of token) {
                            if (char === '↑' || char === '↓' || char === '→' || char === '←' || char === '/' || char === '+' ||
                                char === '?' || char === '<' || char === '>' || char === '"' || char === "'") {
                                if (currentToken) {
                                    newTempTokens.push(currentToken);
                                    currentToken = "";
                                }
                                newTempTokens.push(char);
                            } else {
                                currentToken += char;
                            }
                        }
                        if (currentToken) {
                            newTempTokens.push(currentToken);
                        }
                    } else {
                        // Tokens without arrows remain unchanged
                        newTempTokens.push(token);
                    }
                }
                tempTokens = newTempTokens;
            }
            
            // Further process each tempToken
            // ── Step 0: split CJK characters into individual tokens ──
            let cjkSplitTokens = [];
            tempTokens.forEach(token => {
                if ([...token].some(ch => isCJK(ch))) {
                    let buf = "";
                    for (const ch of token) {
                        if (isCJK(ch)) {
                            if (buf) { cjkSplitTokens.push(buf); buf = ""; }
                            cjkSplitTokens.push(ch);
                        } else {
                            buf += ch;
                        }
                    }
                    if (buf) cjkSplitTokens.push(buf);
                } else {
                    cjkSplitTokens.push(token);
                }
            });
            tempTokens = cjkSplitTokens;

            tempTokens.forEach(token => {
                // Single CJK char → pass through unchanged (skip English regexes)
                if ([...token].length === 1 && isCJK(token)) {
                    tokens.push(token);
                    return;
                }

                // Check for patterns like letters followed by hyphen/plus, or letters followed by ending period
                if (/[a-zA-Z][\-\+\.\"\']+/.test(token) || /[0-9]\.$/.test(token) || /[a-zA-Z]\.$/.test(token) || /[a-zA-Z]\/$/.test(token) ||
                    /[0-9][a-zA-Z]/.test(token) || /^C3[0-9]/.test(token) || /^C4[0-9]/.test(token) || /RBC[0-9]/.test(token) ||
                    /(?<!^)[a-z][A-Z]/.test(token) || /(?<!^)[A-Z][a-z]/.test(token) ) {
                    let newTokens = [];
                    let currentToken = "";

                    // Process character by character
                    for (let i = 0; i < token.length; i++) {
                        const char = token[i];

                        // Check if current character is a letter and next is -, +, or ending period
                        if ((i < token.length - 1 && /[a-zA-Z]/.test(char) && ['-', '+', '.', '"', '"'].includes(token[i+1])) || 
                            (i === token.length - 2 && (token[i+1] === '.' || token[i+1] === '/')) ||
                            (i < token.length - 1 && /[0-9]/.test(char) && /[a-zA-Z]/.test(token[i+1])) ||
                            (i == 1 && (char == '3' || char == '4') && token[i-1] == 'C' && i < token.length-1 && /[0-9]/.test(token[i+1])) ||
                            (i >= 2 && char == 'C' && token[i-1] == 'B' && token[i-2] == 'R' && i < token.length-1 && /[0-9]/.test(token[i+1])) ||
                            (i > 0 && i < token.length - 1 && /[a-z]/.test(char) && /[A-Z]/.test(token[i+1])) ||
                            (i > 0 && i < token.length - 1 && /[A-Z]/.test(char) && /[a-z]/.test(token[i+1]))) {

                            // Add accumulated token
                            currentToken += char;
                            newTokens.push(currentToken);
                            currentToken = "";

                            // Collect all consecutive special characters
                            let specialChars = "";
                            let j = i + 1;
                            while (j < token.length && ['-', '+', '.', '"', '"'].includes(token[j])) {
                                specialChars += token[j];
                                j++;
                            }

                            // If consecutive - or +, treat each as a separate token
                            if (specialChars && [...specialChars].every(c => ['-', '+', '.', '"', '"'].includes(c))) {
                                [...specialChars].forEach(c => {
                                    newTokens.push(c);
                                });
                            }
                            // If ending period, treat as a separate token
                            else if (specialChars === "." && j === token.length) {
                                newTokens.push(".");
                            }
                            else {
                                // Other cases, keep as is
                                newTokens.push(specialChars);
                            }
                            i = j - 1;  // Adjust index
                        } else {
                            // Normal case, accumulate characters
                            currentToken += char;
                        }
                    }
                    // Add the last token (if any)
                    if (currentToken) {
                        newTokens.push(currentToken);
                    }
                    tokens = tokens.concat(newTokens);
                }
                // Check if it contains punctuation between letters or between numbers
                else if (/[a-zA-Z0-9][\/\-\_\@\&\+\.][a-zA-Z0-9]/.test(token)) {
                    // Use new method to correctly handle punctuation between letters or numbers
                    let newTokens = [];
                    let currentToken = "";

                    // Check character by character
                    for (let i = 0; i < token.length; i++) {
                        const char = token[i];

                        // Check if it's a delimiter
                        if (/[\/\-\_\@\&\+\.]/.test(char)) {
                            // Check if both sides are letters or both are numbers
                            const prevChar = i > 0 ? token[i-1] : null;
                            const nextChar = i < token.length - 1 ? token[i+1] : null;

                            if (prevChar && nextChar && 
                                ((/[a-zA-Z]/.test(prevChar) && /[a-zA-Z]/.test(nextChar)) || 
                                 (/[0-9]/.test(prevChar) && /[0-9]/.test(nextChar))||
                                 (/[a-zA-Z]/.test(prevChar) && /[0-9]/.test(nextChar))||
                                 (/[0-9]/.test(prevChar) && /[a-zA-Z]/.test(nextChar)))) {
                                // If both sides are letters or both are numbers, add accumulated characters as a token
                                if (currentToken) {
                                    newTokens.push(currentToken);
                                    currentToken = "";
                                }
                                // Add delimiter as a separate token
                                newTokens.push(char);
                            } else {
                                // Otherwise, treat delimiter as part of the current token
                                currentToken += char;
                            }
                        } else {
                            // Non-delimiter, directly add to current token
                            currentToken += char;
                        }
                    }
                    // Add the last token
                    if (currentToken) {
                        newTokens.push(currentToken);
                    }
                    tokens = tokens.concat(newTokens);
                } else {
                    // If token has no special patterns, keep as is
                    tokens.push(token);
                }
            });
        });
        // Debug output
        console.log("Tokenization result:", tokens);
        tokens.forEach((token) => {
            // Ensure token is not an empty string
            if (token && token.trim && token.trim() !== '') {
                const tokenObject = {
                    text: token,
                    token_seq: tokenSeq
                };
                wordData.push(tokenObject);
                annotations.push({
                    text: token,
                    token_seq: tokenSeq,
                    annotation: {
                        entity: "O",
                        attribute: {},
                        event: "none",
                        sourceRelationships: [],
                        targetRelationships: []
                    }
                });
                const span = document.createElement('span');
                span.textContent = token;
                span.id = tokenSeq;
                span.className = 'word-span';
                lineContainer.appendChild(span);
                const space = document.createTextNode(' ');
                lineContainer.appendChild(space);
                tokenSeq++;
            }
        });
    } catch (error) {
        console.error("Error in processLine:", error);
        // Use simple tokenization method as fallback in case of error
        const tokens = line.split(/\s+/);
        tokens.forEach(token => {
            if (token && token.trim() !== '') {
                const span = document.createElement('span');
                span.textContent = token;
                span.id = tokenSeq;
                span.className = 'word-span';
                lineContainer.appendChild(span);
                const space = document.createTextNode(' ');
                lineContainer.appendChild(space);
                
                wordData.push({ text: token, token_seq: tokenSeq });
                annotations.push({
                    text: token,
                    token_seq: tokenSeq,
                    annotation: {
                        entity: "O",
                        attribute: {},
                        event: "none",
                        sourceRelationships: [],
                        targetRelationships: []
                    }
                });
                tokenSeq++;
            }
        });
    }
    $('#fileContent').append(lineContainer);
}

function applySavedAnnotations(savedAnnotations, predicted) {
    console.log(savedAnnotations);
    savedAnnotations.forEach(function (savedAnn) {
        const idx = savedAnn.token_seq;
        if (annotations[idx]) {
            annotations[idx].annotation = savedAnn.annotation;
        }
        const span = document.getElementById(idx);
        if (span) {
            if (savedAnn.annotation.entity !== "O") {
                span.classList.add("annotated", "entity");
                if (savedAnn.annotation.entity.startsWith('B-')) {
                    span.classList.add("b");
                }
                if (savedAnn.annotation.entity.startsWith('I-')) {
                    span.classList.add("i");
                }
                if (predicted) {
                    span.classList.add("predicted");
                }
            } else if (savedAnn.annotation.event !== "none") {
                span.classList.add("event", "annotated");
                if (savedAnn.annotation.event.startsWith('B-')) {
                    span.classList.add("b");
                }
                if (savedAnn.annotation.event.startsWith('I-')) {
                    span.classList.add("i");
                }
            }
            if ((savedAnn.annotation.sourceRelationships && savedAnn.annotation.sourceRelationships.length > 0) ||
                (savedAnn.annotation.targetRelationships && savedAnn.annotation.targetRelationships.length > 0)) {
                span.classList.add("relationship");
            }
        }
    });
    updateAnnotationLabels();
}

$('#textDisplay').on('contextmenu', '.annotated', function (event) {
    event.preventDefault();
    const group = getAnnotationGroup(this);
    const groupTokenIds = group.map(token => token.id.toString());
    const removedRelationships = relationships.filter(rel =>
        groupTokenIds.some(id => rel.sourceTokens.includes(id) || rel.targetTokens.includes(id))
    );
    const removedRelIds = removedRelationships.map(rel => rel.id);
    relationships = relationships.filter(rel => !removedRelIds.includes(rel.id));
    annotations.forEach(function (ann) {
        if (ann.annotation.sourceRelationships) {
            ann.annotation.sourceRelationships = ann.annotation.sourceRelationships.filter(rel => !removedRelIds.includes(rel.id));
        }
        if (ann.annotation.targetRelationships) {
            ann.annotation.targetRelationships = ann.annotation.targetRelationships.filter(rel => !removedRelIds.includes(rel.id));
        }
    });
    annotations.forEach(function (ann) {
        const tokenElem = document.getElementById(ann.token_seq);
        if (tokenElem) {
            if ((!ann.annotation.sourceRelationships || ann.annotation.sourceRelationships.length === 0) &&
                (!ann.annotation.targetRelationships || ann.annotation.targetRelationships.length === 0)) {
                tokenElem.classList.remove("relationship", "relationship-source", "relationship-target");
                tokenElem.style.border = "";
            }
        }
    });
    group.forEach(function (token) {
        // Remove label
        const label = token.parentElement.querySelector(".annotation-label");
        if (label) {
            label.remove();
        }
        // Remove annotation
        token.classList.remove("annotated", "b", "i", "entity", "event", "relationship", "relationship-source", "relationship-target", "selected", "highlighted");
        token.style.border = "";
        let ann = annotations.find(a => a.token_seq === parseInt(token.id));
        if (ann) {
            ann.annotation.entity = "O";
            ann.annotation.attribute = {};
            ann.annotation.event = "none";
            ann.annotation.sourceRelationships = [];
            ann.annotation.targetRelationships = [];
        }
    });
    updateRelationshipList();
    updateAnnotationLabels();
    console.log("Removed relationships with IDs: " + removedRelIds.join(", ") +
        " for tokens: " + groupTokenIds.join(", "));
});

$('#textDisplay').on('mousedown', '.word-span', function (event) {
    if (preventAnnotation) {
        return;
    }
    selectedWords = [];
    manuallySelectedWord = this;
});

$('#textDisplay').on('mouseup', function () {
    if (preventAnnotation) {
        return;
    }
    let selection = window.getSelection();
    selectedWords = [];
    if (selection.rangeCount > 0 && selection.toString().trim().length > 0) {
        let range = selection.getRangeAt(0);
        document.querySelectorAll("#textDisplay .word-span").forEach(span => {
            if (range.intersectsNode(span)) {
                selectedWords.push(span);
            }
        });
    }
    if (selectedWords.length === 0 && manuallySelectedWord) {
        selectedWords.push(manuallySelectedWord);
    }
    manuallySelectedWord = null;
    if (selectedWords.length > 0) {
        $('#modal-selected-annotation-text').text(selectedWords.map(w => w.innerHTML).join(' '));
        $('#modal-overlay').fadeIn();
        $("#modal").css({
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)"
        }).fadeIn();
        preventAnnotation = true
    }
});

const resizer = document.getElementById("resizer");
const relationshipParent = document.getElementById("relationshipParent");
let startX, startWidth;
resizer.addEventListener("mousedown", initResize, false);
function initResize(e) {
    const container = document.getElementById("relationship-container");
    if (getComputedStyle(container).display === "none") {
        return;
    }
    startX = e.clientX;
    startWidth = relationshipParent.offsetWidth;
    window.addEventListener("mousemove", resizeParent, false);
    window.addEventListener("mouseup", stopResize, false);
}
function resizeParent(e) {
    let newWidth = startWidth + (startX - e.clientX);
    if (newWidth < 100) newWidth = 100;
    if (newWidth > 600) newWidth = 600;
    relationshipParent.style.width = newWidth + "px";
}
function stopResize(e) {
    window.removeEventListener("mousemove", resizeParent, false);
    window.removeEventListener("mouseup", stopResize, false);
}

$('#saveAnnotations').on('click', function () {
    if (!current_filename) {
        const filename = $('#fileInput')[0].files[0]?.name;
        if (!filename) {
            alert('No file is currently open.');
            return;
        }
    }
    if (isRunning) {
        // If the timer is running, calculate the current time
        const currentTime = Date.now() - startTime + elapsedTime;
        annotationTime = currentTime;
    } else {
        // If the timer is paused, use the accumulated time
        annotationTime = elapsedTime;
    }

    const outputData = {
        annotations: annotations,
        relationships: relationships,
        filename: current_filename,
    };

    $.ajax({
        url: '/annotations',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(outputData),
        success: function (response) {
            showNotification(response.message);

            // 保存檔案標註時間
            saveFileAnnotationTime(current_filename, annotationTime);

            clearFileContent();
            clearRelationships();
            $('#fileContent').text('Loading...');
        },
        error: function (xhr) {
            alert('Error saving annotations: ' + xhr.responseJSON.error);
        }
    });
    sleep(1000).then(() => socket.emit('Annotation Finished'))
    // pause the timer first while waiting
    pauseTimer();
});

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function bindModalEvents() {
    const entityRadios = document.querySelectorAll('input[name="entity"]');
    const eventRadios  = document.querySelectorAll('input[name="event"]');
    const attributeArea = document.getElementById("attribute-area");

    // ---- EVENT SELECTED ----
    eventRadios.forEach(radio => {
        radio.onchange = function () {
            if (!this.checked) return;
            entityRadios.forEach(e => e.checked = false);
            attributeArea.style.display = "none";
            attributeArea.querySelectorAll('input[type="radio"]')
                .forEach(r => r.checked = false);
        };
    });

    // ---- ENTITY SELECTED ----
    entityRadios.forEach(radio => {
        radio.onchange = function () {
            if (!this.checked) return;
            eventRadios.forEach(e => e.checked = false);

            const entity = this.value;
            const map  = annotationConfig.entityAttributeMap?.[entity];
            const mode = annotationConfig.entityAttributeMapMode || "explicit";

            // "all" mode OR no map at all → show all attributes, unrestricted
            if (mode === "all" || !annotationConfig.entityAttributeMap) {
                // No attributes configured at all → nothing to show
                if (Object.keys(annotationConfig.attributeTypes || {}).length === 0) {
                    attributeArea.style.display = "none";
                    attributeArea.querySelectorAll('input[type="radio"]')
                        .forEach(r => r.checked = false);
                    return;
                }
                attributeArea.style.display = "block";
                attributeArea.querySelectorAll('.attribute-row').forEach(row => {
                    row.style.display = "";
                    row.querySelectorAll('input[type="radio"]').forEach(r => {
                        r.disabled = false;
                        r.parentElement.style.display = "";
                    });
                });
                return;
            }

            // New config: entity has no mapped attributes → hide whole panel
            if (!map || Object.keys(map).length === 0) {
                attributeArea.style.display = "none";
                attributeArea.querySelectorAll('input[type="radio"]')
                    .forEach(r => r.checked = false);
                return;
            }

            attributeArea.style.display = "block";

            attributeArea.querySelectorAll('.attribute-row').forEach(row => {
                const prefix = row.dataset.attributePrefix;
                const allowedDisplays = map[prefix];

                if (!allowedDisplays) {
                    row.style.display = "none";
                    row.querySelectorAll('input[type="radio"]').forEach(r => {
                        r.checked = false;
                        r.disabled = true;
                        r.parentElement.style.display = "none";
                    });
                    return;
                }

                row.style.display = "";
                row.querySelectorAll('input[type="radio"]').forEach(r => {
                    const displayVal = r.dataset.displayValue;
                    const allowed = allowedDisplays.includes(displayVal);
                    r.disabled = !allowed;
                    r.parentElement.style.display = allowed ? "" : "none";
                    if (!allowed) r.checked = false;
                });
            });

            // Apply default attributes
            const defaults = annotationConfig.entityDefaultAttributes?.[entity] || [];
            defaults.forEach(defAttr => {
                const [prefix, val] = defAttr.split(':').map(s => s.trim());
                if (!val) return;
                document.querySelectorAll(`#attribute-area input[name="${prefix}"]`).forEach(r => {
                    if (r.dataset.displayValue === val && !r.disabled) r.checked = true;
                });
            });
        };
    });
}

(function initTooltip() {
    const tooltip = document.createElement("div");
    tooltip.className = "annotation-tooltip";
    document.body.appendChild(tooltip);
    document.body.addEventListener("mouseenter", function (event) {
        if (showAnnotations) return;
        if (event.target.classList.contains("annotated")) {
            const tokenElem = event.target;
            const tokenSeqId = parseInt(tokenElem.id);
            const annData = annotations.find(ann => ann.token_seq === tokenSeqId);
            if (!annData) return;
            let tooltipText = "";
            if (tokenElem.classList.contains("entity")) {
                // Only entity shows attributes, if there are attributes then include them in display
                const a = annData.annotation.attribute || {};
                const vals = Object.values(a).filter(Boolean);
                const attrText = vals.length > 0 ? vals.map(attributeDisplayName).join(", ") : "none";
                tooltipText = "Entity: " + entityTagDisplayName(annData.annotation.entity) + " | Attributes: " + attrText;
            } else if (tokenElem.classList.contains("event")) {
                // Event only shows event, does not show attribute
                tooltipText = "Event: " + annData.annotation.event;
            } else if (tokenElem.classList.contains("relationship")) {
                tooltipText = "Relationship annotation";
            } else {
                tooltipText = "No annotation";
            }
            tooltip.textContent = tooltipText;
            tooltip.style.left = event.pageX + "px";
            tooltip.style.top = (event.pageY - 30) + "px";
            tooltip.style.visibility = "visible";
            tooltip.style.opacity = "1";
        }
    }, true);
    document.body.addEventListener("mouseleave", function (event) {
        tooltip.style.visibility = "hidden";
        tooltip.style.opacity = "0";
    }, true);
})();

// toggle show full annotations
document.getElementById("toggleAnnotations").addEventListener("change", function() {
    showAnnotations = this.checked
    updateAnnotationLabels();
});

// Annotation label
function createAnnotationLabel(tokenElem, annotationText, predicted) {
    // console.log('creating annotation label: '+annotationText)
    // Remove any existing label for this annotation
    const existingLabel = tokenElem.querySelector(".annotation-label");
    if (existingLabel) {
        existingLabel.remove();
    }

    // Create a new label
    const label = document.createElement("div");
    label.className = "annotation-label";
    label.textContent = annotationText;
    label.addEventListener('click', function(event) {
        event.stopPropagation();
        tokenElem.click();
    })
    if (predicted) {
        label.classList.add("predicted");
    }
    tokenElem.parentElement.appendChild(label);
    // Check if the label overflows outside of 'fileContent'
    const fileContentRect = document.getElementById("fileContent").getBoundingClientRect();
    function updateLabelPosition() {
        label.style.left = `${tokenElem.offsetLeft}px`;
        label.style.top = `${tokenElem.offsetTop - label.offsetHeight}px`; // Adjust above token

        const labelRect = label.getBoundingClientRect();
        if (labelRect.right > fileContentRect.right + 10) {
            // Not enough space, move token to next line
            tokenElem.style.removeProperty("margin-left"); // reset token margin
            const prevTokenElem = document.getElementById(`${parseInt(tokenElem.id)-1}`);
            const shiftAmount = fileContentRect.right - prevTokenElem.getBoundingClientRect().right;
            prevTokenElem.style.marginRight = `${shiftAmount-4}px`
            // Also move the annotation label
            label.style.left = `${tokenElem.offsetLeft}px`;
            label.style.top = `${tokenElem.offsetTop - label.offsetHeight}px`;
        }
    }
    updateLabelPosition();
    // Find the next token with class "b" in the same line
    let nextToken = tokenElem.nextElementSibling;
    while (nextToken && !nextToken.classList.contains("b")) {
        // Ensure we stay within the same line
        if (nextToken.closest(".line") !== tokenElem.closest(".line")) {
            break; // Stop if we reach a different line
        }
        nextToken = nextToken.nextElementSibling;
    }
    // If we found the annotation span, check the space between
    if (nextToken && nextToken.classList.contains("b")) {
        const tokenTop = tokenElem.getBoundingClientRect().top;
        const nextTokenTop = nextToken.getBoundingClientRect().top;
        const spaceBetween = nextToken.offsetLeft - (tokenElem.offsetLeft + label.offsetWidth);
        // Shift only the required amount to prevent overlap
        if (Math.abs(nextTokenTop - tokenTop) <= 5 && spaceBetween < 0) {
            const margin = Math.abs(spaceBetween) + 5;
            // If token doesn't fit in remaining space, increase right margin of the last token instead
            if (nextToken.offsetWidth + nextToken.offsetLeft + margin > fileContentRect.right) {
                const nextPrevToken = document.getElementById(`${parseInt(nextToken.id)-1}`);
                nextPrevToken.style.marginRight = `${fileContentRect.right - nextPrevToken.getBoundingClientRect().right - nextPrevToken.offsetWidth -4}px`;
            } else {
                nextToken.style.marginLeft = `${margin}px`;
            }
        }
    }
}

function attributeDisplayName(value) {
    const attrTypes = annotationConfig.attributeTypes || {};
    for (const group of Object.values(attrTypes)) {
        for (const [display, mapped] of Object.entries(group)) {
            if (mapped === value || display === value) return display;
        }
    }
    return value;
}

function updateAnnotationLabels() {
    // Save current scroll position
    const scrollY = window.scrollY;
    // Temporarily disable scrolling to prevent visible jumps
    document.body.style.overflow = "hidden";

    document.querySelectorAll(".annotation-label").forEach(label => label.remove()); // Clear previous labels
    document.querySelectorAll(".word-span").forEach(token => {
        token.style.removeProperty("margin-left"); // Reset token position
        token.style.removeProperty("margin-right"); // Reset token position
    });
    document.querySelectorAll(".line").forEach(line => {
        line.style.removeProperty("line-height"); // Reset line height
    });
    // Reset token positioning when annotations are hidden
    if (!showAnnotations) {
        document.body.style.overflow = ""; // Restore scrolling
        return;
    }
    document.querySelectorAll(".line").forEach(line => {
        let hasLabel = false;
        line.querySelectorAll(".annotated").forEach(tokenElem => {
            const tokenSeqId = parseInt(tokenElem.id);
            const annData = annotations.find(ann => ann.token_seq === tokenSeqId);
            if (!annData) return;

            let annotationText = "";
            let attrText = "";
            let predicted = false;
            if (tokenElem.classList.contains("entity") && annData.annotation.entity.startsWith('B-')) {
                const a = annData.annotation.attribute || {};
                const vals = Object.values(a).filter(Boolean);
                if (vals.length > 0) attrText = vals.map(attributeDisplayName).join(", ");
                const code = annData.annotation.entity.slice(2);       // e.g. "QYL"
                annotationText = `${entityDisplayName(code)} [${attrText}]`;
                predicted = tokenElem.classList.contains("predicted");
            } else if (tokenElem.classList.contains("event") && annData.annotation.event.startsWith('B-')) {
                annotationText = annData.annotation.event.slice(2);
            }
            if (annotationText) {
                hasLabel = true
                line.style.lineHeight = "5.5";
                createAnnotationLabel(tokenElem, annotationText, predicted);
            }
        });
    });
    // Restore scroll position **without visible movement**
    window.scrollTo({ top: scrollY, behavior: "instant" });
    // Re-enable scrolling
    document.body.style.overflow = "";
}
window.addEventListener("resize", updateAnnotationLabels);

// Edit annotations
document.addEventListener("dblclick", function (event) {
    if (event.target.classList.contains("annotated")) {
        document.body.style.userSelect = "none";

        // Get the annotation for the clicked word
        const tokenSeqId = parseInt(event.target.id);
        const annotation = annotations.find(ann => ann.token_seq === tokenSeqId);
        if (annotation) {
            const entityName = annotation.annotation.entity.replace(/^B-|^I-/, '');
            const eventName = annotation.annotation.event.replace(/^B-|^I-/, '');
            const is_entity = annotation.annotation.entity.startsWith("B-") || annotation.annotation.entity.startsWith("I-") // otherwise it's an event
            
            // Find the start of the annotation span
            let startTokenSeqId = tokenSeqId;
            while (startTokenSeqId > 0) {
                const startAnnotation = annotations.find(ann => ann.token_seq === startTokenSeqId);
                if (startAnnotation && is_entity && startAnnotation.annotation.entity.startsWith("B-")) {
                    break;
                } else if (startAnnotation && !is_entity && startAnnotation.annotation.event.startsWith("B-")) {
                    break;
                } else if (startTokenSeqId == 0) {
                    break;
                } else if (!annotations.find(ann => ann.token_seq === startTokenSeqId-1).annotation.entity.endsWith(entityName)) {
                    break;
                }
                startTokenSeqId--;
            }

            // Collect all spans that belong to the same annotation
            selectedWords = [];
            let currentTokenSeqId = startTokenSeqId;
            while (true) {
                const currentToken = document.getElementById(currentTokenSeqId);
                const spanAnnotation = annotations.find(ann => ann.token_seq === currentTokenSeqId);
                if (spanAnnotation && is_entity && spanAnnotation.annotation.entity.slice(2) === entityName &&
                ((currentTokenSeqId === startTokenSeqId && spanAnnotation.annotation.entity.startsWith("B-")) || (currentTokenSeqId > startTokenSeqId && spanAnnotation.annotation.entity.startsWith("I-")))) {
                    selectedWords.push(currentToken);
                    currentTokenSeqId++;
                } else if (spanAnnotation && !is_entity && spanAnnotation.annotation.event.slice(2) === eventName &&
                ((currentTokenSeqId === startTokenSeqId && spanAnnotation.annotation.event.startsWith("B-")) || (currentTokenSeqId > startTokenSeqId && spanAnnotation.annotation.event.startsWith("I-")))) {
                    selectedWords.push(currentToken);
                    currentTokenSeqId++;
                } else {
                    break;
                }
            }
            if (!selectedWords.length) {
                showNotification("Invalid annotation. Please delete and redo annotation.");
                return;
            }

            // Set the radio buttons based on the annotation
            if (annotation.annotation.entity !== "O") {
                const entityRadio = document.querySelector(`input[name="entity"][value="${entityName}"]`);
                if (!entityRadio) {
                    console.warn("No entity radio for", entityName, "— is annotationConfig loaded?");
                    return;
                }
                entityRadio.checked = true;
                document.querySelectorAll('input[name="event"]').forEach(eradio => eradio.checked = false);
                document.querySelectorAll('#attribute-area input[type="radio"]').forEach(attr => {
                    attr.checked = false;
                });
                entityRadio.dispatchEvent(new Event("change"));

                const a = annotation.annotation.attribute || {};
                Object.values(a).filter(Boolean).forEach(attr => {
                    document.querySelectorAll('#attribute-area input[type="radio"]').forEach(r => {
                        if (!r.disabled && r.dataset.displayValue === attr) r.checked = true;
                    });
                });
            } else if (annotation.annotation.event !== "none") {
                document.querySelector(`input[name="event"][value="${eventName}"]`).checked = true;
                document.querySelectorAll('input[name="entity"]').forEach(eradio => eradio.checked = false);
                document.getElementById("attribute-area").style.display = "none";
                document.querySelectorAll('#attribute-area input[type="radio"]').forEach(attr => {
                    attr.checked = false;
                });
            }

            // Update the modal with the annotation details
            $('#modal-selected-annotation-text').text(selectedWords.map(w => w.innerHTML).join(' '));
            $('#modal-overlay').fadeIn();
            $("#modal").css({
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)"
            }).fadeIn();
            preventAnnotation = true;
        }
    }
});

let selectedTokens = [];
let relationships = [];
let isDragging = false;
let dragStartToken = null;
let preventAnnotation = false; 

// Define drag threshold and starting coordinates
let dragThreshold = 5;
let drag_startX = 0, drag_startY = 0;

document.addEventListener("mousedown", function (event) {
    // Only detect dragging for annotated tokens
    if (event.target.classList.contains("annotated")) {
    isDragging = true;
    dragStartToken = event.target; // Record the starting token
    // Record starting mouse coordinates
    drag_startX = event.clientX;
    drag_startY = event.clientY;
    // Disable text selection (remove highlight effect)
    document.body.style.userSelect = "none";
    }
});

document.addEventListener("mouseup", function (event) {
    // Restore text selection
    document.body.style.userSelect = "";
        
    if (!isDragging) return; // If dragging hasn't started, return directly

    // End dragging, reset isDragging
    isDragging = false;
    let dragEndToken = event.target;

    // Calculate mouse movement distance
    let diffX = Math.abs(event.clientX - drag_startX);
    let diffY = Math.abs(event.clientY - drag_startY);

    // If movement distance exceeds threshold, treat as dragging
    if (diffX > dragThreshold || diffY > dragThreshold) {
        // If drag start and end points are different
        if (dragStartToken !== dragEndToken) {
            // Get annotation groups of the drag start and end tokens
            const groupA = getAnnotationGroup(dragStartToken);
            const groupB = getAnnotationGroup(dragEndToken);
            selectedTokens = [groupA, groupB];  // Save the selected token groups
            
            // Directly check and save relationship (without popping up modal)
            onRelationshipDrag(dragStartToken, dragEndToken);
            
            // preventAnnotation = true; // Set preventAnnotation to prevent subsequent click from triggering annotation modal
            manuallySelectedWord = null; // Clear manual selection variable
            event.stopImmediatePropagation();
            return;
        }
    }
    // If dragging distance does not exceed threshold, treat as simple click, do not proceed with drag handling
    manuallySelectedWord = null;
}, true);

// Helper：Remove B-/I- prefixes
function normalize(tag) {
    if (!tag) return "";
    if (tag.startsWith("B-") || tag.startsWith("I-")) {
        return tag.slice(2);
    }
    return tag;
}

function isRelationshipAllowed(startToken, targetToken) {
    let startAnn = annotations.find(a => a.token_seq === parseInt(startToken.id));
    let targetAnn = annotations.find(a => a.token_seq === parseInt(targetToken.id));
    if (!startAnn || !targetAnn) return { allowed: false };

    // Get normalized values (remove B-/I- prefixes)
    let startEvent = normalize(startAnn.annotation.event);
    let targetEvent = normalize(targetAnn.annotation.event);
    let startEntity = normalize(startAnn.annotation.entity);
    let targetEntity = normalize(targetAnn.annotation.entity);

    if (startEvent && startEvent.toLowerCase() !== "none" &&
        startEvent !== "Unit") {
        // Then the target token must have an entity annotation (and normalized value is not "O")
        if (targetEntity && targetEntity !== "O") {
        return { allowed: true, relationshipType: "link to" };
        } else {
        return { allowed: false };
        }
    }

    // Rule 2: If the starting token's event is "Unit" (normalized)
    if (startEvent === "Unit") {
        if (targetEvent === "Value") {
        return { allowed: true, relationshipType: "is unit of" };
        } else {
        return { allowed: false };
        }
    }

    // Rule 3: If the starting token's entity is "Time" (normalized)
    if (startEntity === "Time") {
        if (targetEntity && targetEntity !== "O") {
        return { allowed: true, relationshipType: "is time of" };
        } else {
        return { allowed: false };
        }
    }

    return { allowed: false };
}

function onRelationshipDrag() {
    // Ensure selectedTokens has stored two sets of tokens, and each set has at least one token
    if (selectedTokens.length !== 2 || selectedTokens[0].length === 0 || selectedTokens[1].length === 0) {
        console.log("Drag failed: insufficient tokens selected.");
        return;
    }

    // Use the first token of each set as the check reference
    let startToken = selectedTokens[0][0];
    let targetToken = selectedTokens[1][0];

    // Check if relationship rules are met (note isRelationshipAllowed already handles B-/I- prefixes)
    const check = isRelationshipAllowed(startToken, targetToken);
    if (check.allowed) {
        // Merge the text of the two sets of tokens as token1 and token2
        let token1Text = selectedTokens[0].map(token => token.textContent).join(" ");
        let token2Text = selectedTokens[1].map(token => token.textContent).join(" ");
        
        // Create new relationship object (format consistent with original)
        let newRel = {
        id: "rel_" + new Date().getTime(),
        token1: token1Text,             // Consistent with token1 used in the list
        relationship: check.relationshipType, // Get relationship type directly from check result
        token2: token2Text,             // Consistent with token2 used in the list
        sourceTokens: selectedTokens[0].map(token => token.id),
        targetTokens: selectedTokens[1].map(token => token.id)
        };
        
        // Add to global relationships array
        relationships.push(newRel);
        console.log("Saved relationship:", newRel);
        
        // Update relationship list UI (based on your implementation of updateRelationshipList())
        updateRelationshipList();
        
        // Update source token: add red dashed border, update annotation's sourceRelationships
        selectedTokens[0].forEach(function(token) {
        let tokenSeqId = parseInt(token.id);
        let ann = annotations.find(a => a.token_seq === tokenSeqId);
        if (ann) {
            if (!ann.annotation.sourceRelationships) {
            ann.annotation.sourceRelationships = [];
            }
            ann.annotation.sourceRelationships.push(newRel);
        }
        token.classList.add("relationship", "relationship-source");
        token.style.border = "4px dashed red";
        });
        
        // Update target token: add blue dashed border, update annotation's targetRelationships
        selectedTokens[1].forEach(function(token) {
        let tokenSeqId = parseInt(token.id);
        let ann = annotations.find(a => a.token_seq === tokenSeqId);
        if (ann) {
            if (!ann.annotation.targetRelationships) {
            ann.annotation.targetRelationships = [];
            }
            ann.annotation.targetRelationships.push(newRel);
        }
        token.classList.add("relationship", "relationship-target");
        token.style.border = "4px dashed blue";
        });

        showNotification("Relationship Created!");
        updateAnnotationLabels();
    } else {
        showNotification("Please select a valid relationship pair.");
    }

    // Clear selected tokens, regardless of success or failure
    selectedTokens = [];
}

// Toggle relationship panel show/hide
function toggleRelationshipPanel() {
    const relationshipContainer = document.getElementById("relationship-container");
    if (relationshipContainer.style.display === "flex") {
    relationshipContainer.style.display = "none";
    } else {
    relationshipContainer.style.display = "flex";
    }
}

function updateRelationshipList() {
    const list = document.getElementById("relationship-list");
    list.innerHTML = ""; // Clear the list

    relationships.forEach((rel, index) => {
        // Create li item
        let listItem = document.createElement("li");

        // Create flex container, all content on the same row
        let rowDiv = document.createElement("div");
        rowDiv.className = "relationship-item-row";

        // Create delete button, placed at the front
        let deleteBtn = document.createElement("button");
        deleteBtn.textContent = "❌";
        deleteBtn.className = "rel-delete-btn";
        deleteBtn.onclick = function () {
            const relId = rel.id;
            // Remove this relationship from the global relationships array
            relationships.splice(index, 1);
            
            // Remove the relationship record from all annotations
            annotations.forEach(function(ann) {
                if (ann.annotation.sourceRelationships) {
                    ann.annotation.sourceRelationships = ann.annotation.sourceRelationships.filter(r => r.id !== relId);
                }
                if (ann.annotation.targetRelationships) {
                    ann.annotation.targetRelationships = ann.annotation.targetRelationships.filter(r => r.id !== relId);
                }
            });
            
            // Remove UI styles from all tokens involved in this relationship
            const allTokenIds = rel.sourceTokens.concat(rel.targetTokens);
            allTokenIds.forEach(id => {
                let tokenElem = document.getElementById(id);
                if (tokenElem) {
                    tokenElem.classList.remove("relationship", "relationship-source", "relationship-target");
                    tokenElem.style.border = "";
                }
            });
            
            updateRelationshipList();
            showNotification("A relationship is deleted");
            updateAnnotationLabels();
        };

        // Create span to display token1
        let spanToken1 = document.createElement("span");
        spanToken1.className = "rel-part rel-token1";
        spanToken1.textContent = rel.token1;

        // Create span to display relationship type
        let spanRelationship = document.createElement("span");
        spanRelationship.className = "rel-part rel-relationship";
        spanRelationship.textContent = rel.relationship;

        // Create span to display token2
        let spanToken2 = document.createElement("span");
        spanToken2.className = "rel-part rel-token2";
        spanToken2.textContent = rel.token2;

        // Place delete button at the front, then add other spans in order
        rowDiv.appendChild(deleteBtn);
        rowDiv.appendChild(spanToken1);
        rowDiv.appendChild(spanRelationship);
        rowDiv.appendChild(spanToken2);

        // Add this row to li, then add to the list
        listItem.appendChild(rowDiv);
        list.appendChild(listItem);
    });

    if (relationships.length > 0) {
        document.getElementById("relationship-container").style.display = "flex";
    }
}

function clearRelationships() {
    $('#relationship-list').empty();
    $('#textDisplay .relationship')
        .removeClass('relationship relationship-source relationship-target')
        .css('border', '');
    relationships = [];
}
/**
* Pass in the DOM element of a token, return the entire annotation group that the token belongs to
*/
function getAnnotationGroup(tokenElement) {
    const tokenSeqId = parseInt(tokenElement.id);
    const currentAnn = annotations.find(ann => ann.token_seq === tokenSeqId);
    if (!currentAnn) return [tokenElement];

    // If the token is annotated as an entity (entity is not "O"), use entity grouping logic
    if (currentAnn.annotation.entity !== "O") {
        let parts = currentAnn.annotation.entity.split("-");
        if (parts.length < 2) return [tokenElement];
        const currentPrefix = parts[0];  // "B" or "I"
        const entityType = parts[1];     
        
        let endIndex = tokenSeqId;
        while (endIndex < annotations.length - 1) {
            const nextAnn = annotations[endIndex + 1];
            if (nextAnn && nextAnn.annotation.entity !== "O") {
                let nextParts = nextAnn.annotation.entity.split("-");
                if (nextParts.length >= 2) {
                    const nextPrefix = nextParts[0];
                    const nextEntity = nextParts[1];
                    if (nextEntity === entityType && nextPrefix === "I") {
                        endIndex++;
                        continue;
                    }
                }
            }
            break;
        }
        let startIndex = tokenSeqId;
        if (currentPrefix === "I") {
            while (startIndex > 0) {
                const prevAnn = annotations[startIndex - 1];
                if (prevAnn && prevAnn.annotation.entity !== "O") {
                    let prevParts = prevAnn.annotation.entity.split("-");
                    if (prevParts.length >= 2) {
                        const prevPrefix = prevParts[0];
                        const prevEntity = prevParts[1];
                        if (prevEntity === entityType) {
                            if (prevPrefix === "I") {
                                startIndex--;
                                continue;
                            }
                            if (prevPrefix === "B") {
                                startIndex--;
                            }
                        }
                    }
                }
                break;
            }
        }
        let group = [];
        for (let i = startIndex; i <= endIndex; i++) {
            let elem = document.getElementById(i);
            if (elem) group.push(elem);
        }
        return group;
    }
    // If the token is annotated as an event (event is not "none"), use similar logic
    else if (currentAnn.annotation.event !== "none") {
        let parts = currentAnn.annotation.event.split("-");
        if (parts.length < 2) return [tokenElement];
        const currentPrefix = parts[0];  // "B" or "I"
        const eventType = parts[1];       
        
        let endIndex = tokenSeqId;
        while (endIndex < annotations.length - 1) {
            const nextAnn = annotations[endIndex + 1];
            if (nextAnn && nextAnn.annotation.event !== "none") {
                let nextParts = nextAnn.annotation.event.split("-");
                if (nextParts.length >= 2) {
                    const nextPrefix = nextParts[0];
                    const nextEvent = nextParts[1];
                    if (nextEvent === eventType && nextPrefix === "I") {
                        endIndex++;
                        continue;
                    }
                }
            }
            break;
        }
        let startIndex = tokenSeqId;
        if (currentPrefix === "I") {
            while (startIndex > 0) {
                const prevAnn = annotations[startIndex - 1];
                if (prevAnn && prevAnn.annotation.event !== "none") {
                    let prevParts = prevAnn.annotation.event.split("-");
                    if (prevParts.length >= 2) {
                        const prevPrefix = prevParts[0];
                        const prevEvent = prevParts[1];
                        if (prevEvent === eventType) {
                            if (prevPrefix === "I") {
                                startIndex--;
                                continue;
                            }
                            if (prevPrefix === "B") {
                                startIndex--;
                            }
                        }
                    }
                }
                break;
            }
        }
        let group = [];
        for (let i = startIndex; i <= endIndex; i++) {
            let elem = document.getElementById(i);
            if (elem) group.push(elem);
        }
        return group;
    }
    // If neither is annotated, just return the token
    else {
        return [tokenElement];
    }
}

function applyStyles() {
    // Get user input values
    const fontSize = document.getElementById("fontSize").value + "px";
    const annotated_bgColor = document.getElementById("annotated_bgColor").value;
    const annotated_textColor = document.getElementById("annotated_textColor").value;
    const selectionBgColor = document.getElementById("selectionBgColor").value;
    const selectionTextColor = document.getElementById("selectionTextColor").value;

    // Update word-span font size
    document.getElementById("textDisplay").style.fontSize = fontSize

    updateCSSRule('.annotated', 'background-color', annotated_bgColor)
    updateCSSRule('.annotated', 'color', annotated_textColor)

    removeCSSRule('.word-span:hover');
    removeCSSRule('#textDisplay span::selection');
    addCSSRule('.word-span:hover', 
    `
        background-color: ${selectionBgColor}; 
        color: ${selectionTextColor};
        transform: scale(1.1);
        font-size: 1.05em;
        transition: transform 0.3s ease, font-size 0.3s ease, background-color 0.3s ease;
        border-radius: 8px; 
    `);
    addCSSRule('#textDisplay span::selection', 
    `
        background-color: ${selectionBgColor}; 
        color: ${selectionTextColor};
        transform: scale(1.1);
        font-size: 1.05em;
        transition: transform 0.3s ease, font-size 0.3s ease, background-color 0.3s ease;
    `);

}

function updateCSSRule(selector, property, value) {
    const styleSheet = document.styleSheets[0]; // Get the first CSS stylesheet
    for (let i = 0; i < styleSheet.cssRules.length; i++) {
        let rule = styleSheet.cssRules[i];
        console.log(value)
        if (rule.selectorText === selector) { // Find the specified CSS selector
            console.log("Before: ",rule.style[property])
            rule.style[property]= value; // Directly modify CSS property
            console.log("After: " + rule.style[property])
            return;
        }
    }
    console.log(styleSheet.cssRules.selectorText)
}

function removeCSSRule(selector) {
    const styleSheet = document.styleSheets[0];
    for (let i = 0; i < styleSheet.cssRules.length; i++) {
        if (styleSheet.cssRules[i].selectorText === selector) {
            styleSheet.deleteRule(i);
            return;
        }
    }
}

function addCSSRule(selector, ruleContent) {
    const styleSheet = document.styleSheets[0];
    styleSheet.insertRule(`${selector} { ${ruleContent} }`, styleSheet.cssRules.length);
}

// Use data from style_config.json to update CSS rules
function applyStyleConfig(styleConfig) {
    // Set font size of textDisplay
    const textDisplay = document.getElementById("textDisplay");
    if (textDisplay) {
        textDisplay.style.fontSize = styleConfig.textDisplayFontSize;
    }
    
    // Update background color and text color of .annotated
    updateCSSRule('.annotated', 'background-color', styleConfig.annotatedHighlightColor);
    updateCSSRule('.annotated', 'color', styleConfig.annotatedTextColor);
    
    // Update ::selection style (since ::selection is a pseudo-element, recommend removing first, then adding)
    removeCSSRule("::selection");
    addCSSRule("::selection", `background-color: ${styleConfig.selectionBgColor}; color: ${styleConfig.selectionTextColor};`);
    
    // Update border color of relationship token1 and token2 (assuming using .rel-token1 and .rel-token2 CSS selectors)
    updateCSSRule('.rel-token1', 'border-color', styleConfig.relationshipToken1BorderColor);
    updateCSSRule('.rel-token2', 'border-color', styleConfig.relationshipToken2BorderColor);
}

function toggleToolbox() {
    const toolbox = document.getElementById("settings-panel");
    const container = document.getElementById("toolbox-container");

    if (container.classList.contains("open")) {
        // Collapse toolbox
        toolbox.style.right = "-250px";
        container.classList.remove("open");
    } else {
        // Expand toolbox
        toolbox.style.right = "0";
        container.classList.add("open");
    }
}

async function fetchNotes() {
    // Disable save button if filename is empty or undefined
    const saveButton = document.getElementById("save-notes");
    if (current_filename === '') {
        saveButton.disabled = true;
        return;
    } else {
        saveButton.disabled = false;
    }
    // Fetch saved notes
    try {
        const notesArea = document.getElementById("note-textarea");
        const shortFilename = current_filename.split(/[/\\]/).pop();
        const response = await fetch(`/get_notes?filename=${shortFilename}&current_dir=${currentDir}`);
        const data = await response.json();
        notesArea.value = data.notes || ""; // Set notes if found
    } catch (error) {
    }
}

function showNoteBox() {
    const noteBox = document.getElementById("note-box");
    const button = document.getElementById("toggle-notes");
    const noteBoxWidth = parseInt(window.getComputedStyle(noteBox).width, 10);
    const buttonRight = parseInt(window.getComputedStyle(button).right, 10);
    noteBox.style.right = "0px"; // Show the box
    button.style.right = `${buttonRight + noteBoxWidth}px`;
    fetchNotes();
}

function hideNoteBox() {
    const noteBox = document.getElementById("note-box");
    const button = document.getElementById("toggle-notes");
    const noteBoxWidth = parseInt(window.getComputedStyle(noteBox).width, 10);
    const buttonRight = parseInt(window.getComputedStyle(button).right, 10);
    noteBox.style.right = "-300px"; // Hide the box
    button.style.right = `${buttonRight - noteBoxWidth}px`;
}

document.getElementById("toggle-notes").addEventListener("click", async function() {
    const noteBox = document.getElementById("note-box");
    const noteBoxRight = parseInt(window.getComputedStyle(noteBox).right, 10);
    if (noteBoxRight === -300) {
        showNoteBox();
    } else if (noteBoxRight === 0){
        hideNoteBox();
    }
});

document.getElementById("save-notes").addEventListener("click", function() {
    const notesContent = document.getElementById("note-textarea").value;
    const shortFilename = current_filename.split(/[/\\]/).pop();
    fetch("/save_notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: shortFilename, notes: notesContent, current_dir: currentDir })
    }).then(response => response.json())
    .then(data => {
        alert(data.message); // Notify user upon success
        hideNoteBox();
    }).catch(error => {
        console.error("Error saving notes:", error);
    });
});

function saveFileAnnotationTime(filename, annotationTime) {
    $.ajax({
        url: '/save_file_time',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            filename: filename,
            annotation_time: annotationTime,
        }),
        success: function(response) {
            console.log("File annotation time saved:", response.message);
        },
        error: function(xhr) {
            console.error("Error saving file annotation time:", xhr.responseText);
        }
    });
}
