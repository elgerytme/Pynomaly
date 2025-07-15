/**
 * Voice Commands Module for Pynomaly
 * Provides speech recognition and voice navigation capabilities
 */

class VoiceCommandManager {
  constructor() {
    this.isListening = false;
    this.recognition = null;
    this.synthesis = window.speechSynthesis;
    this.isSupported = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    this.commands = new Map();
    this.context = 'global'; // global, dashboard, detection, etc.
    
    this.init();
  }

  init() {
    if (!this.isSupported) {
      console.warn('Speech recognition not supported in this browser');
      return;
    }

    this.setupRecognition();
    this.registerCommands();
    this.setupUI();
  }

  setupRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognition = new SpeechRecognition();
    
    this.recognition.continuous = true;
    this.recognition.interimResults = false;
    this.recognition.lang = 'en-US';
    this.recognition.maxAlternatives = 3;

    this.recognition.onstart = () => {
      this.isListening = true;
      this.updateUI();
      this.speak('Voice commands activated');
    };

    this.recognition.onend = () => {
      this.isListening = false;
      this.updateUI();
    };

    this.recognition.onresult = (event) => {
      const results = event.results[event.resultIndex];
      const command = results[0].transcript.toLowerCase().trim();
      
      console.log('Voice command received:', command);
      this.processCommand(command);
    };

    this.recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      this.handleError(event.error);
    };
  }

  registerCommands() {
    // Navigation Commands
    this.addCommand(['go to dashboard', 'show dashboard', 'navigate to dashboard'], () => {
      window.location.href = '/';
    });

    this.addCommand(['go to datasets', 'show datasets', 'navigate to datasets'], () => {
      window.location.href = '/datasets';
    });

    this.addCommand(['go to detectors', 'show detectors', 'navigate to detectors'], () => {
      window.location.href = '/detectors';
    });

    this.addCommand(['run detection', 'start detection', 'detect anomalies'], () => {
      window.location.href = '/detection';
    });

    this.addCommand(['show visualizations', 'go to charts', 'view charts'], () => {
      window.location.href = '/visualizations';
    });

    this.addCommand(['show monitoring', 'real time monitoring', 'monitor'], () => {
      window.location.href = '/monitoring';
    });

    // Accessibility Commands
    this.addCommand(['toggle high contrast', 'high contrast mode'], () => {
      window.accessibilityManager?.toggleHighContrast();
    });

    this.addCommand(['increase font size', 'larger text', 'bigger text'], () => {
      window.accessibilityManager?.adjustFontSize(10);
    });

    this.addCommand(['decrease font size', 'smaller text'], () => {
      window.accessibilityManager?.adjustFontSize(-10);
    });

    this.addCommand(['toggle dark mode', 'dark theme', 'switch theme'], () => {
      const themeToggle = document.getElementById('theme-toggle');
      themeToggle?.click();
    });

    // Search Commands
    this.addCommand(['search for', 'find', 'look for'], (command) => {
      const searchTerm = command.replace(/^(search for|find|look for)\s+/, '');
      const searchInput = document.getElementById('main-search');
      if (searchInput && searchTerm) {
        searchInput.value = searchTerm;
        searchInput.focus();
        this.speak(`Searching for ${searchTerm}`);
      }
    });

    // Form Commands
    this.addCommand(['upload file', 'upload dataset', 'add file'], () => {
      const fileInput = document.querySelector('input[type="file"]');
      fileInput?.click();
    });

    this.addCommand(['submit form', 'submit', 'send form'], () => {
      const submitButton = document.querySelector('button[type="submit"], input[type="submit"]');
      submitButton?.click();
    });

    // Help Commands
    this.addCommand(['help', 'what can i say', 'voice commands', 'show commands'], () => {
      this.showHelp();
    });

    this.addCommand(['stop listening', 'turn off voice', 'disable voice'], () => {
      this.stopListening();
    });

    // Page-specific commands based on context
    this.registerContextualCommands();
  }

  registerContextualCommands() {
    // Dashboard specific commands
    if (window.location.pathname === '/' || window.location.pathname.includes('dashboard')) {
      this.addCommand(['refresh data', 'update dashboard', 'reload'], () => {
        const refreshButton = document.querySelector('[data-action="refresh"]');
        refreshButton?.click();
      });

      this.addCommand(['new detection', 'create detection'], () => {
        const newButton = document.querySelector('.btn-primary');
        newButton?.click();
      });
    }

    // Detection page commands
    if (window.location.pathname.includes('detection')) {
      this.addCommand(['select algorithm', 'choose algorithm'], () => {
        const algorithmSelect = document.querySelector('select[name="algorithm"]');
        algorithmSelect?.focus();
      });

      this.addCommand(['start analysis', 'begin detection', 'analyze data'], () => {
        const startButton = document.querySelector('.start-detection');
        startButton?.click();
      });
    }

    // Dataset page commands
    if (window.location.pathname.includes('datasets')) {
      this.addCommand(['upload new dataset', 'add dataset'], () => {
        const uploadButton = document.querySelector('.upload-dataset');
        uploadButton?.click();
      });

      this.addCommand(['view dataset', 'show dataset details'], () => {
        const firstDataset = document.querySelector('.dataset-item a');
        firstDataset?.click();
      });
    }
  }

  addCommand(phrases, action) {
    if (Array.isArray(phrases)) {
      phrases.forEach(phrase => {
        this.commands.set(phrase.toLowerCase(), action);
      });
    } else {
      this.commands.set(phrases.toLowerCase(), action);
    }
  }

  processCommand(command) {
    // Try exact match first
    if (this.commands.has(command)) {
      this.commands.get(command)(command);
      return;
    }

    // Try partial matches for commands with parameters
    for (const [phrase, action] of this.commands) {
      if (command.startsWith(phrase)) {
        action(command);
        return;
      }
    }

    // If no match found, provide feedback
    this.speak("Sorry, I didn't understand that command. Say 'help' for available commands.");
  }

  startListening() {
    if (!this.isSupported || this.isListening) return;
    
    try {
      this.recognition.start();
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      this.speak('Sorry, voice commands are not available right now.');
    }
  }

  stopListening() {
    if (!this.isListening) return;
    
    this.recognition.stop();
    this.speak('Voice commands deactivated');
  }

  toggleListening() {
    if (this.isListening) {
      this.stopListening();
    } else {
      this.startListening();
    }
  }

  speak(text, options = {}) {
    if (!this.synthesis) return;

    // Cancel any ongoing speech
    this.synthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = options.rate || 1;
    utterance.pitch = options.pitch || 1;
    utterance.volume = options.volume || 0.8;

    // Find a natural-sounding voice
    const voices = this.synthesis.getVoices();
    const preferredVoice = voices.find(voice => 
      voice.lang.startsWith('en') && voice.name.includes('Natural')
    ) || voices.find(voice => voice.lang.startsWith('en'));
    
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    this.synthesis.speak(utterance);
  }

  setupUI() {
    const voiceToggle = document.getElementById('voice-commands-toggle');
    if (voiceToggle) {
      voiceToggle.addEventListener('click', () => {
        this.toggleListening();
      });
    }

    // Add keyboard shortcut for voice commands
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + Shift + V
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'V') {
        e.preventDefault();
        this.toggleListening();
      }
    });
  }

  updateUI() {
    const voiceToggle = document.getElementById('voice-commands-toggle');
    if (voiceToggle) {
      voiceToggle.setAttribute('aria-pressed', this.isListening.toString());
      
      const icon = voiceToggle.querySelector('span[aria-hidden="true"]');
      if (icon) {
        icon.textContent = this.isListening ? '🔴' : '🎤';
      }
      
      const text = voiceToggle.querySelector('.sr-only');
      if (text) {
        text.textContent = this.isListening ? 'Stop voice commands' : 'Start voice commands';
      }
    }

    // Update accessibility toolbar
    const toolbar = document.getElementById('accessibility-toolbar');
    if (toolbar) {
      toolbar.classList.toggle('voice-active', this.isListening);
    }

    // Announce to screen readers
    if (window.accessibilityManager) {
      const message = this.isListening ? 
        'Voice commands are now listening' : 
        'Voice commands stopped';
      window.accessibilityManager.announce(message);
    }
  }

  showHelp() {
    const helpCommands = [
      'Navigation: "Go to dashboard", "Show datasets", "Run detection"',
      'Accessibility: "Toggle high contrast", "Increase font size"',
      'Search: "Search for [term]", "Find [item]"',
      'Actions: "Upload file", "Submit form", "Refresh data"',
      'Control: "Stop listening", "Help"'
    ];
    
    const helpText = `Available voice commands: ${helpCommands.join('. ')}`;
    this.speak(helpText);
    
    // Also show visual help
    this.showVisualHelp(helpCommands);
  }

  showVisualHelp(commands) {
    // Create or update help modal
    let helpModal = document.getElementById('voice-help-modal');
    if (!helpModal) {
      helpModal = document.createElement('div');
      helpModal.id = 'voice-help-modal';
      helpModal.className = 'modal';
      helpModal.setAttribute('role', 'dialog');
      helpModal.setAttribute('aria-labelledby', 'voice-help-title');
      
      helpModal.innerHTML = `
        <div class="modal-content">
          <div class="modal-header">
            <h2 id="voice-help-title">Voice Commands Help</h2>
            <button class="modal-close" aria-label="Close help">×</button>
          </div>
          <div class="modal-body">
            <p>You can use the following voice commands:</p>
            <ul id="voice-commands-list" class="list-disc pl-6 space-y-2"></ul>
            <p class="mt-4 text-sm text-gray-600">
              Press <kbd>Ctrl+Shift+V</kbd> to toggle voice commands on/off.
            </p>
          </div>
        </div>
      `;
      
      document.body.appendChild(helpModal);
    }
    
    const commandsList = helpModal.querySelector('#voice-commands-list');
    commandsList.innerHTML = commands.map(cmd => `<li>${cmd}</li>`).join('');
    
    // Show modal
    helpModal.classList.add('show');
    helpModal.setAttribute('aria-hidden', 'false');
    helpModal.querySelector('.modal-content').focus();
    
    // Close modal functionality
    const closeButton = helpModal.querySelector('.modal-close');
    closeButton.addEventListener('click', () => {
      helpModal.classList.remove('show');
      helpModal.setAttribute('aria-hidden', 'true');
    });
    
    // Close on escape
    helpModal.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeButton.click();
      }
    });
  }

  handleError(error) {
    let message = 'Voice command error occurred';
    
    switch (error) {
      case 'no-speech':
        message = 'No speech detected. Please try again.';
        break;
      case 'audio-capture':
        message = 'Microphone not available.';
        break;
      case 'not-allowed':
        message = 'Microphone permission denied.';
        break;
      case 'network':
        message = 'Network error for voice recognition.';
        break;
    }
    
    this.speak(message);
    
    if (window.accessibilityManager) {
      window.accessibilityManager.announce(message, 'assertive');
    }
  }

  // Public API
  isSupported() {
    return this.isSupported;
  }

  getCurrentContext() {
    return this.context;
  }

  setContext(context) {
    this.context = context;
    this.registerContextualCommands();
  }
}

// Initialize voice commands
export function initVoiceCommands() {
  if (!window.voiceCommandManager) {
    window.voiceCommandManager = new VoiceCommandManager();
  }
  return window.voiceCommandManager;
}

// Export for direct use
export { VoiceCommandManager };