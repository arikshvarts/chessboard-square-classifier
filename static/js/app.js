// Chess Board Position Classifier - Frontend JavaScript

let currentResult = null;

// Chess piece Unicode symbols
const PIECE_SYMBOLS = {
    'white_king': '♔',
    'white_queen': '♕',
    'white_rook': '♖',
    'white_bishop': '♗',
    'white_knight': '♘',
    'white_pawn': '♙',
    'black_king': '♚',
    'black_queen': '♛',
    'black_rook': '♜',
    'black_bishop': '♝',
    'black_knight': '♞',
    'black_pawn': '♟',
    'empty': ''
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeUpload();
    initializeButtons();
});

// Initialize file upload functionality
function initializeUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });
    
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
}

// Initialize button handlers
function initializeButtons() {
    document.getElementById('clearBtn')?.addEventListener('click', clearImage);
    document.getElementById('newAnalysisBtn')?.addEventListener('click', resetApp);
    document.getElementById('retryBtn')?.addEventListener('click', resetApp);
    document.getElementById('copyFenBtn')?.addEventListener('click', copyFEN);
    document.getElementById('downloadBtn')?.addEventListener('click', downloadResults);
}

// Handle file selection
function handleFileSelect() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        showImagePreview(e.target.result);
        processImage(file);
    };
    reader.readAsDataURL(file);
}

// Show image preview
function showImagePreview(imageSrc) {
    const uploadArea = document.getElementById('uploadArea');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    
    uploadArea.classList.add('hidden');
    imagePreview.classList.remove('hidden');
    previewImg.src = imageSrc;
}

// Clear image and reset
function clearImage() {
    resetApp();
}

// Process uploaded image
async function processImage(file) {
    showLoading(true);
    hideError();
    hideResults();
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            showError(result.error);
        } else if (result.success) {
            currentResult = result;
            displayResults(result);
        } else {
            showError('Unexpected response from server');
        }
    } catch (error) {
        showError(`Network error: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Display results
function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.remove('hidden');
    
    // Display chess board
    displayChessBoard(result.board_state);
    
    // Display FEN
    displayFEN(result.fen);
    
    // Display statistics
    displayStatistics(result.board_state);
}

// Display chess board visualization
function displayChessBoard(boardState) {
    const chessBoard = document.getElementById('chessBoard');
    chessBoard.innerHTML = '';
    
    for (let rank = 0; rank < 8; rank++) {
        for (let file = 0; file < 8; file++) {
            const square = document.createElement('div');
            square.className = 'chess-square';
            
            // Alternate light/dark squares
            const isLight = (rank + file) % 2 === 0;
            square.classList.add(isLight ? 'light' : 'dark');
            
            const squareData = boardState[rank][file];
            const piece = squareData.piece;
            const confidence = squareData.confidence;
            
            // Add piece symbol
            if (piece !== 'empty') {
                const pieceSpan = document.createElement('span');
                pieceSpan.className = 'piece';
                pieceSpan.textContent = PIECE_SYMBOLS[piece] || '?';
                square.appendChild(pieceSpan);
            }
            
            // Add confidence badge
            const badge = document.createElement('div');
            badge.className = 'confidence-badge';
            badge.textContent = `${(confidence * 100).toFixed(0)}%`;
            square.appendChild(badge);
            
            // Mark low confidence squares
            if (confidence < 0.7) {
                square.classList.add('low-confidence');
            }
            
            // Add tooltip
            square.title = `${squareData.square}: ${piece} (${(confidence * 100).toFixed(1)}% confidence)`;
            
            chessBoard.appendChild(square);
        }
    }
}

// Display FEN notation
function displayFEN(fen) {
    const fenText = document.getElementById('fenText');
    fenText.textContent = fen;
}

// Display statistics
function displayStatistics(boardState) {
    let totalConfidence = 0;
    let lowConfidenceCount = 0;
    let piecesCount = 0;
    let squareCount = 0;
    
    for (let rank = 0; rank < 8; rank++) {
        for (let file = 0; file < 8; file++) {
            const squareData = boardState[rank][file];
            totalConfidence += squareData.confidence;
            squareCount++;
            
            if (squareData.confidence < 0.7) {
                lowConfidenceCount++;
            }
            
            if (squareData.piece !== 'empty') {
                piecesCount++;
            }
        }
    }
    
    const avgConfidence = (totalConfidence / squareCount * 100).toFixed(1);
    
    document.getElementById('avgConfidence').textContent = `${avgConfidence}%`;
    document.getElementById('lowConfCount').textContent = lowConfidenceCount;
    document.getElementById('piecesCount').textContent = piecesCount;
}

// Copy FEN to clipboard
function copyFEN() {
    const fenText = document.getElementById('fenText').textContent;
    
    navigator.clipboard.writeText(fenText).then(() => {
        showNotification('FEN copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showNotification('Failed to copy FEN', 'error');
    });
}

// Download results
function downloadResults() {
    if (!currentResult) return;
    
    const data = {
        fen: currentResult.fen,
        board_state: currentResult.board_state,
        predictions: currentResult.predictions,
        timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chess-position-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    showNotification('Results downloaded!');
}

// Show loading indicator
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (show) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}

// Show error message
function showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
}

// Hide error message
function hideError() {
    const errorSection = document.getElementById('errorSection');
    errorSection.classList.add('hidden');
}

// Hide results
function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.add('hidden');
}

// Show notification
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        background: ${type === 'success' ? '#27ae60' : '#e74c3c'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Reset app to initial state
function resetApp() {
    const uploadArea = document.getElementById('uploadArea');
    const imagePreview = document.getElementById('imagePreview');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    hideError();
    hideResults();
    showLoading(false);
    
    fileInput.value = '';
    currentResult = null;
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
